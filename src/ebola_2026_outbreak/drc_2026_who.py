"""Scrape the WHO country table for the 2026 Bundibugyo virus disease outbreak.

The table sits under the "Ebola disease (Bundibugyo virus)" heading on the WHO
alert-and-response page. It shows only the latest figures, so each run upserts
into a tracked CSV to build a history. Each country has its own "Data as of"
date, which becomes the `date` column.
"""

import logging
import re
from datetime import UTC, date, datetime
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
DATE_ROW = re.compile(r"Data as of (\d{1,2} \w+ \d{4})")
DATE_FORMAT = "%d %B %Y"
MISSING = "-"
# Footnote markers WHO appends to country names, e.g. "France ***", "Kenya#".
FOOTNOTE_MARKERS = re.compile(r"[\s*#†]+$")

# Header text (whitespace-normalised) -> indicator label.
INDICATORS = {
    "Suspected cases": "Suspected cases",
    "Confirmed cases": "Confirmed cases",
    "Confirmed deaths": "Confirmed deaths",
    "Probable cases": "Probable cases",
    "Probable deaths": "Probable deaths",
    "CFR (%) confirmed": "CFR confirmed (%)",
    "Recovered": "Recovered",
}
NON_ADDITIVE = frozenset({"CFR confirmed (%)"})
# Indicators used downstream: a Total mismatch here rejects the capture.
# WHO's own Total row is occasionally off for the others (Recovered sums to
# 1 970 against 1 960 on 24 September 2026), which only logs a warning.
STRICT_INDICATORS = frozenset({"Confirmed cases", "Confirmed deaths"})

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


def find_table(soup: BeautifulSoup) -> Tag:
    """Locate the table that follows the Bundibugyo section heading.

    Raises:
        ValueError: If the heading or table is missing, or the section shows
            the table as an image.
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
            return tag
        if tag.name == "h2":
            break
        image_src = image_src or tag.get("src")
    if image_src:
        # Until early September 2026 WHO published the table as a PNG.
        raise ValueError(f"Table published as an image: {image_src}")
    raise ValueError(f"No table after heading {SECTION_HEADING!r}")


def _parse_header(row: Tag) -> list[str]:
    """Map header cells to indicator labels, raising on unknown columns."""
    cells = [_cell_text(c) for c in row.find_all(["td", "th"])]
    if cells[0] != FIRST_HEADER:
        raise ValueError(f"Unexpected first header cell: {cells[0]!r}")
    unknown = [c for c in cells[1:] if c not in INDICATORS]
    if unknown:
        raise ValueError(f"Unknown WHO table columns: {unknown}")
    return [INDICATORS[c] for c in cells[1:]]


def parse_who_table(
    html: str, *, retrieved_at: datetime, source_url: str = SOURCE_URL
) -> pl.DataFrame:
    """Parse the Bundibugyo table into long rows, one per country and indicator.

    The Total row is used to validate country sums and is not returned.

    Args:
        html: Full HTML of the WHO page.
        retrieved_at: When the page was captured (UTC).
        source_url: Where the page came from, stored for provenance.

    Returns:
        DataFrame with columns in `SCHEMA`.

    Raises:
        ValueError: If the table layout is unrecognised or totals do not match.
    """
    rows = find_table(BeautifulSoup(html, "html.parser")).find_all("tr")
    header_idx = next(
        (i for i, r in enumerate(rows) if _cell_text(r).startswith(FIRST_HEADER)),
        None,
    )
    if header_idx is None:
        raise ValueError(f"No header row starting with {FIRST_HEADER!r}")
    indicators = _parse_header(rows[header_idx])

    records: list[dict[str, object]] = []
    totals: dict[str, float | None] = {}
    as_of: date | None = None
    for row in rows[header_idx + 1 :]:
        cells = [_cell_text(c) for c in row.find_all(["td", "th"])]
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
    validate(df, totals=totals)
    return df


def validate(df: pl.DataFrame, *, totals: dict[str, float | None]) -> None:
    """Check keys are unique, log nulls, and check country sums match the Total row.

    Raises:
        ValueError: On duplicate keys, a missing Total row or a sum mismatch
            for an indicator in `STRICT_INDICATORS`.
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
        if indicator in STRICT_INDICATORS:
            raise ValueError(message)
        logger.warning(message)


def upsert(new: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Merge new rows into the CSV at `path`, keeping the latest capture per key.

    WHO revises figures, so a later `retrieved_at` replaces an earlier one.

    Args:
        new: Rows from one or more snapshots.
        path: Tracked CSV to merge into; created if absent.

    Returns:
        The merged DataFrame that was written.
    """
    frames = [new]
    if path.exists():
        frames.insert(0, pl.read_csv(path, schema=SCHEMA))
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
    html = get(SOURCE_URL).text

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
