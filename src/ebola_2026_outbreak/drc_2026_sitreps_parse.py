"""Parse province case and death counts from DRC Ministry of Health sitreps.

Each sitrep since mid-June has a table of confirmed cases and deaths by
province, ending in a Total row. Its layout drifts between sitreps (column
order, headers, wrapped province names), so rows are found by province name
rather than by template. Numbers use a space as thousands separator, which
makes "44 5 659 2 583" ambiguous; the row's printed case fatality rate
(deaths / cases) settles which reading is right. Province rows must sum to the
Total row, so a misread or missing province fails loudly.
"""

import logging
import re
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import pdfplumber
import polars as pl

from ebola_2026_outbreak.drc_2026_sitreps_download import (
    MANIFEST_PATH,
    MANIFEST_SCHEMA,
    OUTBREAK,
    ROOT,
    SITREP_DIR,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = ROOT / "data" / "processed" / f"{OUTBREAK}_sitrep_by_province.csv"

# Spelling variants seen in the tables -> canonical province name.
PROVINCES = {
    r"Ituri": "Ituri",
    r"Nord[- ]Kivu": "Nord-Kivu",
    r"Sud[- ]Kivu": "Sud-Kivu",
    r"Haut[- ]U[ée]l[ée]": "Haut-Uélé",
    r"Bas[- ]U[ée]l[ée]": "Bas-Uélé",
    r"Tshopo": "Tshopo",
    r"Sud[- ]Ubangi": "Sud-Ubangi",
    r"Nord[- ]Ubangi": "Nord-Ubangi",
}
TOTAL = "Total"
LABEL = re.compile(
    r"^(?P<label>" + "|".join(PROVINCES) + r"|total(?: RDC)?)\b[\s*]*(?P<rest>.*)$",
    re.I,
)
# A province row wrapped over three lines, the name split at either end and
# CFR fragments on either side of the numbers:
# "Sud-" / "0 1 1 100,0 % 1/16 —" / "Ubangi**"
# "Sud- 100,0" / "0 1 1 1/16 —" / "Ubangi** %"
# "Sud-Ubangi 100,0" / "0 1 1 1/16 —" / "** %"
WRAP_HEAD = re.compile(r"^(Sud|Nord|Haut|Bas)-(Ubangi|Kivu|U[ée]l[ée])?\s*([\d,]*)$")
WRAP_TAIL = re.compile(r"^(Ubangi|Kivu|U[ée]l[ée])?[\s*]*(%?)$")
NUMBERS_ONLY = re.compile(r"^[\d\s]+\d+(?:,\d+)?\s*%")
PERCENT = re.compile(r"(\d+(?:,\d+)?)\s*%")
# Columns that are not counts: "28/36", "(77,8 %)", "20 sur 36", "—".
NON_COUNT = re.compile(r"\d+\s*(?:/|sur)\s*\d+|\(\s*[\d,]+\s*%\s*\)|—|-")
MISSING = "ND"
# Non-row lines (footnotes, wrap fragments) tolerated inside one table.
MAX_GAP_LINES = 3
# Half a unit of the printed CFR's one decimal place.
CFR_TOLERANCE = 0.05 + 1e-9

INDICATORS = {
    "new_cases": "New confirmed cases (24h)",
    "cases": "Cumulative confirmed cases",
    "deaths": "Cumulative confirmed deaths",
}
# Early sitreps without a province table of cumulative counts.
NO_PROVINCE_TABLE = {
    date(2026, 5, 19): "alerts summary only",
    date(2026, 5, 21): "new counts by health zone only",
    date(2026, 8, 14): "one-off layout with no CFR or Total row to validate",
}

SCHEMA = pl.Schema(
    {
        "outbreak": pl.String,
        "date": pl.Date,
        "province": pl.String,
        "indicator": pl.String,
        "value": pl.Int64,
        "sitrep_number": pl.Int64,
        "media_id": pl.Int64,
    }
)


class ProvinceCounts(NamedTuple):
    """One row of the province table."""

    new_cases: int | None
    cases: int
    deaths: int


def canonical_province(label: str) -> str:
    """Map a table label to its canonical province name, or "Total"."""
    for pattern, name in PROVINCES.items():
        if re.fullmatch(pattern, label, re.I):
            return name
    if label.lower().startswith("total"):
        return TOTAL
    raise ValueError(f"Unknown province label: {label!r}")


def join_split_rows(lines: list[str]) -> list[str]:
    """Rejoin rows that PDF extraction split across lines.

    Handles a wrapped province name (see `WRAP_HEAD`) and a
    Total row whose numbers sit on the line above ("86 6 843 ... 48,4%" /
    "Total 61/151").
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        head = WRAP_HEAD.match(line)
        tail = WRAP_TAIL.match(lines[i + 2]) if i + 2 < len(lines) else None
        if head and tail and bool(head.group(2)) != bool(tail.group(1)):
            name = f"{head.group(1)}-{head.group(2) or tail.group(1)}"
            out.append(f"{name} {lines[i + 1]} {head.group(3)} {tail.group(2)}")
            i += 3
            continue
        if (
            NUMBERS_ONLY.match(line)
            and i + 1 < len(lines)
            and re.match(r"(?i)^total\b", lines[i + 1])
            and not PERCENT.search(NON_COUNT.sub(" ", lines[i + 1]))
        ):
            out.append(f"Total {line} {lines[i + 1][len('Total') :]}")
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def readings(tokens: list[str]) -> list[list[int]]:
    """List every way to group digit tokens into numbers.

    A number is one token, or a 1-3 digit token without a leading zero
    followed by 3-digit tokens ("5 659" -> 5659).
    """
    if not tokens:
        return [[]]
    results: list[list[int]] = []
    for k in range(1, len(tokens) + 1):
        head = tokens[:k]
        if k > 1 and (
            len(head[0]) > 3 or head[0].startswith("0") or len(head[-1]) != 3
        ):
            break
        value = int("".join(head))
        results.extend([value, *rest] for rest in readings(tokens[k:]))
    return results


def cfr_matches(cases: int, deaths: int, cfr: float) -> bool:
    """Check deaths / cases rounds to the printed case fatality rate."""
    return cases > 0 and abs(deaths / cases * 100 - cfr) <= CFR_TOLERANCE


def parse_counts(rest: str) -> ProvinceCounts:
    """Read new cases, cumulative cases and deaths from a row's text.

    Handles both layouts: new cases before the cumulative columns, or as the
    last column after the CFR and health-zone fractions.

    Raises:
        ValueError: If the row has no CFR or no single consistent reading.
    """
    cfr_match = PERCENT.search(rest)
    if cfr_match is None:
        raise ValueError(f"No CFR in row: {rest!r}")
    cfr = float(cfr_match.group(1).replace(",", "."))
    before = NON_COUNT.sub(" ", rest[: cfr_match.start()]).split()
    after = NON_COUNT.sub(" ", rest[cfr_match.end() :]).split()

    new_missing = bool(before) and before[0] == MISSING
    digits = before[1:] if new_missing else before
    if not all(t.isdigit() for t in digits):
        raise ValueError(f"Non-numeric tokens before CFR: {rest!r}")

    candidates: set[ProvinceCounts] = set()
    for values in readings(digits):
        if len(values) == 3 and not new_missing:
            new, cases, deaths = values
        elif len(values) == 2 and len(after) <= 1:
            cases, deaths = values
            new = int(after[0]) if after and after[0].isdigit() else None
        else:
            continue
        if cfr_matches(cases, deaths, cfr):
            candidates.add(ProvinceCounts(new, cases, deaths))

    if len(candidates) != 1:
        raise ValueError(f"{len(candidates)} readings match CFR {cfr}: {rest!r}")
    return candidates.pop()


def find_province_table(text: str) -> dict[str, ProvinceCounts]:
    """Find the first run of province rows that ends in a parseable Total row.

    Args:
        text: Extracted text of the whole sitrep.

    Returns:
        {province or "Total": counts}, in table order.

    Raises:
        ValueError: If no such table is found.
    """
    lines = join_split_rows([line.strip() for line in text.splitlines()])
    errors: list[str] = []
    block: dict[str, ProvinceCounts] = {}
    gap = 0
    for line in lines:
        match = LABEL.match(line)
        if match is None:
            gap += 1
            if gap > MAX_GAP_LINES:
                block = {}
            continue
        gap = 0
        try:
            name = canonical_province(match.group("label"))
            counts = parse_counts(match.group("rest"))
        except ValueError as e:
            # Narrative sentences also start with province names; only rows
            # that look tabular are worth reporting.
            if match.group("rest")[:1].isdigit():
                errors.append(str(e))
            block = {}
            continue
        if name in block:
            block = {}
        block[name] = counts
        if name == TOTAL and len(block) > 1:
            return block
    raise ValueError(f"No province table found; row errors: {errors[:5]}")


def validate_table(table: dict[str, ProvinceCounts]) -> None:
    """Check provinces sum to the Total row for cases and deaths.

    New cases are checked only when every province reports them.

    Raises:
        ValueError: On any mismatch.
    """
    total = table[TOTAL]
    provinces = [c for name, c in table.items() if name != TOTAL]
    for field in ("cases", "deaths"):
        summed = sum(getattr(c, field) for c in provinces)
        if summed != getattr(total, field):
            raise ValueError(
                f"Provinces sum to {summed} {field}, Total row says "
                f"{getattr(total, field)}"
            )
    new = [c.new_cases for c in provinces]
    if total.new_cases is not None and None not in new:
        if sum(n for n in new if n is not None) != total.new_cases:
            raise ValueError(
                f"Provinces sum to {sum(n for n in new if n is not None)} new "
                f"cases, Total row says {total.new_cases}"
            )


def table_to_frame(
    table: dict[str, ProvinceCounts],
    *,
    report_date: date,
    sitrep_number: int | None,
    media_id: int,
) -> pl.DataFrame:
    """Turn a validated table into long rows, dropping the Total row."""
    records = [
        {
            "outbreak": OUTBREAK,
            "date": report_date,
            "province": province,
            "indicator": label,
            "value": getattr(counts, field),
            "sitrep_number": sitrep_number,
            "media_id": media_id,
        }
        for province, counts in table.items()
        if province != TOTAL
        for field, label in INDICATORS.items()
    ]
    return pl.DataFrame(records, schema=SCHEMA)


def read_text(path: Path) -> str:
    """Extract the text of every page of a PDF."""
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_sitrep(
    text: str, *, report_date: date, sitrep_number: int | None, media_id: int
) -> pl.DataFrame:
    """Parse and validate one sitrep's province table.

    Raises:
        ValueError: If the table is missing, unreadable or does not sum.
    """
    table = find_province_table(text)
    validate_table(table)
    return table_to_frame(
        table,
        report_date=report_date,
        sitrep_number=sitrep_number,
        media_id=media_id,
    )


def check_monotonic(df: pl.DataFrame) -> None:
    """Log any fall in a province's cumulative counts between sitreps."""
    cumulative = df.filter(pl.col("indicator") != INDICATORS["new_cases"]).sort("date")
    for (province, indicator), group in cumulative.group_by(
        "province", "indicator", maintain_order=True
    ):
        for (d0, v0), (d1, v1) in pairwise(group.select("date", "value").iter_rows()):
            if v1 < v0:
                logger.warning(
                    "%s %s falls from %d (%s) to %d (%s)",
                    province,
                    indicator.lower(),
                    v0,
                    d0,
                    v1,
                    d1,
                )


def main() -> None:
    """Parse every kept sitrep not yet in the output and update the CSV."""
    manifest = pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA).filter("is_latest")
    existing = (
        pl.read_csv(OUTPUT_PATH, schema=SCHEMA)
        if OUTPUT_PATH.exists()
        else pl.DataFrame(schema=SCHEMA)
    )
    done = set(existing.select("date", "media_id").unique().iter_rows())

    frames = [existing]
    failures: dict[date, str] = {}
    for row in manifest.sort("report_date").iter_rows(named=True):
        report_date = row["report_date"]
        if (report_date, row["media_id"]) in done:
            continue
        if report_date in NO_PROVINCE_TABLE:
            logger.info("Skipping %s: %s", report_date, NO_PROVINCE_TABLE[report_date])
            continue
        try:
            frames.append(
                parse_sitrep(
                    read_text(SITREP_DIR / f"{row['media_id']}.pdf"),
                    report_date=report_date,
                    sitrep_number=row["sitrep_number"],
                    media_id=row["media_id"],
                )
            )
        except ValueError as e:
            failures[report_date] = str(e)

    # A newer upload for a date replaces rows parsed from an older one.
    latest_media = manifest.select(pl.col("report_date").alias("date"), "media_id")
    out = (
        pl.concat(frames)
        .join(latest_media, on=["date", "media_id"], how="semi")
        .sort("date", "province", "indicator")
    )
    if out.select("date", "province", "indicator").is_duplicated().any():
        raise ValueError("Duplicate (date, province, indicator) rows")
    check_monotonic(out)
    out.write_csv(OUTPUT_PATH)
    logger.info(
        "Wrote %d rows for %d sitreps to %s",
        out.height,
        out["date"].n_unique(),
        OUTPUT_PATH.name,
    )

    if failures:
        for report_date, reason in failures.items():
            logger.error("Failed %s: %s", report_date, reason)
        raise ValueError(f"{len(failures)} sitreps could not be parsed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
