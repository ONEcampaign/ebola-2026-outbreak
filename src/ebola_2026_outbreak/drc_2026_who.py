"""Scrape the WHO country table for the 2026 Bundibugyo virus disease outbreak.

The table sits under the "Ebola disease (Bundibugyo virus)" heading on the WHO
alert-and-response page. It shows only the latest figures, so each run upserts
into a tracked CSV to build a history. Each country has its own "Data as of"
date, which becomes the `date` column.
"""

import logging
import re
from datetime import UTC, date, datetime
from difflib import get_close_matches
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup, Tag

from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTBREAK = "drc_2026"
SOURCE_URL = "https://www.who.int/emergencies/alert-and-response"
SNAPSHOT_DIR = ROOT / "data" / "raw" / OUTBREAK / "who"
OUTPUT_PATH = ROOT / "data" / "processed" / f"{OUTBREAK}_who_by_country.csv"

SECTION_HEADING = "Bundibugyo"
FIRST_HEADER = "Affected countries"
TOTAL_LABEL = "Total"
DATE_ROW = re.compile(r"Data as of\s*(\d{1,2} \w+ \d{4})")
DATE_FORMAT = "%d %B %Y"
MISSING = "-"
# Footnote markers WHO appends to country names, e.g. "France ***", "Kenya#".
FOOTNOTE_MARKERS = re.compile(r"[\s*#†]+$")

# Labels other modules select on. A rename here reaches them all.
DRC_COUNTRY = "Democratic Republic of the Congo"
CONFIRMED_CASES = "Confirmed cases"
CONFIRMED_DEATHS = "Confirmed deaths"
RECOVERED = "Recovered"

# Header text -> indicator label. Suspected deaths and CFR suspected appear
# only in the May 2026 table images.
INDICATORS = {
    "Suspected cases": "Suspected cases",
    "Suspected deaths": "Suspected deaths",
    "CFR (%) suspected": "CFR suspected (%)",
    "Confirmed cases": CONFIRMED_CASES,
    "Confirmed deaths": CONFIRMED_DEATHS,
    "Probable cases": "Probable cases",
    "Probable deaths": "Probable deaths",
    "CFR (%) confirmed": "CFR confirmed (%)",
    "Recovered": RECOVERED,
}
NON_ADDITIVE = frozenset({"CFR confirmed (%)", "CFR suspected (%)"})
# Headers match ignoring spacing and case, and OCR typos ("Confrmed deaths")
# match the closest known header. Distinct headers score at most 0.76.
HEADER_MATCH_CUTOFF = 0.9
# Indicators used downstream: a Total mismatch here rejects the capture.
# WHO's own Total row is occasionally off for the others (Recovered sums to
# 1 970 against 1 960 on 24 September 2026), which only logs a warning.
STRICT_INDICATORS = frozenset({CONFIRMED_CASES, CONFIRMED_DEATHS})

KEY = ["date", "country", "indicator"]
SCHEMA = pl.Schema(
    {
        "outbreak": pl.String,
        "date": pl.Date,
        "country": pl.String,
        "indicator": pl.String,
        "value": pl.Float64,
        "retrieved_at": pl.Datetime("us", "UTC"),
        "source_url": pl.String,
    }
)


def _cell_text(cell: Tag) -> str:
    """Return a cell's text with all whitespace (incl. NBSP) collapsed."""
    return " ".join(cell.get_text(" ").split())


def parse_value(text: str) -> float | None:
    """Parse a WHO table value such as "9 552", "48%" or "-".

    Args:
        text: Cell text.

    Returns:
        The number, or None when WHO reports "-".

    Raises:
        ValueError: If the text is not a recognised number.
    """
    cleaned = re.sub(r"\s", "", text).removesuffix("%")
    if cleaned == MISSING:
        return None
    if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
        raise ValueError(f"Unrecognised WHO table value: {text!r}")
    return float(cleaned)


def _find_section_content(soup: BeautifulSoup) -> tuple[Tag | None, str | None]:
    """Return the Bundibugyo section's table, or failing that its table image URL.

    Raises:
        ValueError: If the section heading is missing.
    """
    heading = soup.find(
        lambda tag: tag.name in {"h2", "h3"} and SECTION_HEADING in tag.get_text()
    )
    if heading is None:
        raise ValueError(f"No heading containing {SECTION_HEADING!r}")
    # Search only this section: stop at the next top-level heading.
    image_src = None
    for tag in heading.find_all_next(["table", "img", "h2"]):
        if tag.name == "table":
            return tag, None
        if tag.name == "h2":
            break
        src = str(tag.get("src", ""))
        if SECTION_HEADING.lower() in src.lower():
            image_src = image_src or src
    return None, image_src


def find_table(soup: BeautifulSoup) -> Tag:
    """Locate the table that follows the Bundibugyo section heading.

    Raises:
        ValueError: If the heading or table is missing, or the section shows
            the table as an image.
    """
    table, image_src = _find_section_content(soup)
    if table is not None:
        return table
    if image_src:
        # Until early September 2026 WHO published the table as a PNG.
        raise ValueError(f"Table published as an image: {image_src}")
    raise ValueError(f"No table after heading {SECTION_HEADING!r}")


def find_table_image(html: str) -> str | None:
    """Return the URL of the table image when the section has no HTML table."""
    table, image_src = _find_section_content(BeautifulSoup(html, "html.parser"))
    return None if table is not None else image_src


def _normalise_header(text: str) -> str:
    """Drop whitespace and case so "CFR(%) confirmed" matches "CFR (%) confirmed"."""
    return re.sub(r"\s", "", text).lower()


_HEADERS = {_normalise_header(k): v for k, v in INDICATORS.items()}


def _parse_header(cells: list[str]) -> list[str]:
    """Map header cells to indicator labels, raising on unknown columns."""
    if cells[0] != FIRST_HEADER:
        raise ValueError(f"Unexpected first header cell: {cells[0]!r}")
    labels: list[str] = []
    unknown: list[str] = []
    for cell in cells[1:]:
        match = get_close_matches(
            _normalise_header(cell), _HEADERS, n=1, cutoff=HEADER_MATCH_CUTOFF
        )
        if match:
            labels.append(_HEADERS[match[0]])
        else:
            unknown.append(cell)
    if unknown:
        raise ValueError(f"Unknown WHO table columns: {unknown}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate WHO table columns: {cells[1:]}")
    return labels


def parse_rows(
    rows: list[list[str]],
    *,
    retrieved_at: datetime,
    source_url: str,
    strict_indicators: frozenset[str] = STRICT_INDICATORS,
) -> pl.DataFrame:
    """Turn the table's cell texts into long rows, one per country and indicator.

    Shared by the HTML parser and the image OCR. The Total row is used to
    validate country sums and is not returned.

    Args:
        rows: Cell texts per table row, header row included.
        retrieved_at: When the page was captured (UTC).
        source_url: Where the table came from, stored for provenance.
        strict_indicators: Indicators whose Total mismatch rejects the table.

    Returns:
        DataFrame with columns in `SCHEMA`.

    Raises:
        ValueError: If the table layout is unrecognised or totals do not match.
    """
    header_idx = next(
        (i for i, r in enumerate(rows) if r and r[0].startswith(FIRST_HEADER)),
        None,
    )
    if header_idx is None:
        raise ValueError(f"No header row starting with {FIRST_HEADER!r}")
    indicators = _parse_header(rows[header_idx])

    records: list[dict[str, object]] = []
    totals: dict[str, float | None] = {}
    as_of: date | None = None
    for cells in rows[header_idx + 1 :]:
        if not any(cells):
            continue
        if match := DATE_ROW.search(" ".join(cells)):
            as_of = datetime.strptime(match.group(1), DATE_FORMAT).date()
            continue
        if len(cells) != len(indicators) + 1:
            raise ValueError(f"Row has {len(cells)} cells, expected: {cells}")

        country = FOOTNOTE_MARKERS.sub("", cells[0])
        values = [parse_value(v) for v in cells[1:]]
        if country == TOTAL_LABEL:
            totals = dict(zip(indicators, values, strict=True))
            continue
        if as_of is None:
            raise ValueError(f"No 'Data as of' date before row for {country}")
        records.extend(
            {"date": as_of, "country": country, "indicator": ind, "value": val}
            for ind, val in zip(indicators, values, strict=True)
        )

    if not records:
        raise ValueError("WHO table has no country rows")
    df = pl.DataFrame(records).with_columns(
        outbreak=pl.lit(OUTBREAK),
        retrieved_at=pl.lit(retrieved_at),
        source_url=pl.lit(source_url),
    )
    df = df.select(list(SCHEMA)).cast(SCHEMA)
    validate(df, totals=totals, strict_indicators=strict_indicators)
    return df


def parse_who_table(
    html: str, *, retrieved_at: datetime, source_url: str = SOURCE_URL
) -> pl.DataFrame:
    """Parse the Bundibugyo HTML table into long rows.

    Args:
        html: Full HTML of the WHO page.
        retrieved_at: When the page was captured (UTC).
        source_url: Where the page came from, stored for provenance.

    Returns:
        DataFrame with columns in `SCHEMA`.

    Raises:
        ValueError: If the table is missing or invalid; see `parse_rows`.
    """
    table = find_table(BeautifulSoup(html, "html.parser"))
    rows = [
        [_cell_text(cell) for cell in tr.find_all(["td", "th"])]
        for tr in table.find_all("tr")
    ]
    return parse_rows(rows, retrieved_at=retrieved_at, source_url=source_url)


def validate(
    df: pl.DataFrame,
    *,
    totals: dict[str, float | None],
    strict_indicators: frozenset[str] = STRICT_INDICATORS,
) -> None:
    """Check keys are unique, log nulls, and check country sums match the Total row.

    Raises:
        ValueError: On duplicate keys, a missing Total row or a sum mismatch
            for an indicator in `strict_indicators`.
    """
    if df.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate (date, country, indicator) rows")

    nulls = df.filter(pl.col("value").is_null())
    if nulls.height:
        logger.info(
            "%d values reported as '-' (null): %s",
            nulls.height,
            nulls.group_by("country").len().sort("country").rows(),
        )

    if not totals:
        raise ValueError("WHO table has no Total row")
    sums = (
        df.filter(~pl.col("indicator").is_in(NON_ADDITIVE))
        .group_by("indicator")
        .agg(pl.col("value").sum())
    )
    for indicator, total in sums.iter_rows():
        expected = totals.get(indicator) or 0.0
        if total == expected:
            continue
        message = f"{indicator}: countries sum to {total}, Total row says {expected}"
        if indicator in strict_indicators:
            raise ValueError(message)
        logger.warning(message)


def upsert(new: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Merge new rows into the CSV at `path`, keeping the latest capture per key.

    WHO revises figures, so a later `retrieved_at` replaces an earlier one.
    A re-capture of an unchanged value keeps the stored row, so a run that
    finds nothing new rewrites the CSV byte for byte and makes no commit.
    The batch is reduced to its latest capture per key first, so a batch
    holding a revision and then a revert back to the stored value keeps the
    stored value.

    Args:
        new: Rows from one or more snapshots.
        path: Tracked CSV to merge into; created if absent.

    Returns:
        The merged DataFrame that was written.
    """
    new = new.sort("retrieved_at", maintain_order=True).unique(
        subset=KEY, keep="last", maintain_order=True
    )
    frames = [new]
    if path.exists():
        existing = pl.read_csv(path, schema=SCHEMA)
        frames = [
            existing,
            new.join(
                existing.select(*KEY, "value"),
                on=[*KEY, "value"],
                how="anti",
                nulls_equal=True,
            ),
        ]
    merged = (
        pl.concat(frames)
        .sort("retrieved_at")
        .unique(subset=KEY, keep="last", maintain_order=True)
        .sort(["date", "country", "indicator"])
    )
    merged.write_csv(path)
    logger.info(
        "Wrote %d rows to %s (%d dates)",
        merged.height,
        path.name,
        merged["date"].n_unique(),
    )
    return merged


def main() -> None:
    """Fetch the WHO page, save a snapshot and upsert the parsed table."""
    retrieved_at = datetime.now(UTC)
    response = get(SOURCE_URL)
    html = response.text
    # When WHO last edited the page, to learn when it updates the table.
    logger.info(
        "Page last modified: %s", response.headers.get("last-modified", "not sent")
    )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = SNAPSHOT_DIR / f"{retrieved_at:%Y%m%dT%H%M%SZ}.html"
    snapshot.write_text(html, encoding="utf-8")
    logger.info("Saved snapshot %s", snapshot.relative_to(ROOT))

    df = parse_who_table(html, retrieved_at=retrieved_at)
    logger.info(
        "Parsed %d rows for %s",
        df.height,
        df.select("country", "date").unique().sort("country").rows(),
    )
    upsert(df, OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
