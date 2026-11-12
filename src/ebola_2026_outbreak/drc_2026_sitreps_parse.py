"""Parse province case and death counts from the DRC sitreps.

The sitreps come from INSP. Files taken from the Ministry of Health's site
before 28 October 2026 are kept for the dates that have one.

Each sitrep since mid-June has a table of confirmed cases and deaths by
province, ending in a Total row. Its layout drifts between sitreps (column
order, headers, wrapped province names), so rows are found by province name.
Numbers use a space as thousands separator, which makes "44 5 659 2 583"
ambiguous. The row's printed case fatality rate (deaths / cases) settles the
reading. Province rows must sum to the Total row, otherwise parsing raises.
Uploads known not to parse are listed in `PROVINCE_TABLE_EXCEPTIONS`.
"""

import logging
import re
from datetime import date
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import pdfplumber
import polars as pl

from ebola_2026_outbreak.drc_2026_sitreps_download import (
    MANIFEST_PATH,
    MANIFEST_SCHEMA,
    OUTBREAK,
    PDF_MIME,
    ROOT,
    MediaItem,
    download,
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
# Columns that are not counts: "28/36", "(77,8 %)", "20 sur 36", "—", and
# footnote markers ("3 769 * 51,1%").
NON_COUNT = re.compile(r"\d+\s*(?:/|sur)\s*\d+|\(\s*[\d,]+\s*%\s*\)|—|-|\*")
MISSING = "ND"
# Non-row lines (footnotes, wrap fragments) tolerated inside one table.
MAX_GAP_LINES = 3
# A wrapped province row spans this many lines (see `WRAP_HEAD`).
WRAPPED_ROW_LINES = 3
# Digits in a thousands group: "5 659" is 5 and 659.
THOUSANDS_GROUP_DIGITS = 3
# Count columns before the CFR when a row has new cases: new, cumulative, deaths.
COUNT_COLUMNS_WITH_NEW = 3
# Row errors quoted when no province table is found.
MAX_ERRORS_SHOWN = 5
# Null rows named in the log when the monotonic check skips them.
MAX_NULLS_LISTED = 5
# Half a unit of the printed CFR's one decimal place.
CFR_TOLERANCE = 0.05 + 1e-9

INDICATORS = {
    "new_cases": "New confirmed cases (24h)",
    "cases": "Cumulative confirmed cases",
    "deaths": "Cumulative confirmed deaths",
}


class SkipCategory(StrEnum):
    """Why an upload's province table is not parsed."""

    NO_TABLE = "no_table"  # no province table of cumulative counts to validate
    PARSER_MISREAD = "parser_misread"  # the table is there, the parser misreads it
    PUBLISHER_ERROR = "publisher_error"  # the printed table does not add up


class TableException(NamedTuple):
    """An upload whose province table is skipped, and why."""

    report_date: date
    category: SkipCategory
    reason: str


# Uploads skipped by parse, keyed by (media_id, modified). A new `modified` for
# an excepted media id fails the run until the entry is reviewed and rekeyed.
PROVINCE_TABLE_EXCEPTIONS: dict[tuple[int, str], TableException] = {
    (24659, "2026-05-16T12:26:49"): TableException(
        date(2026, 5, 14),
        SkipCategory.NO_TABLE,
        "first sitrep, Ituri narrative only",
    ),
    (24706, "2026-05-20T17:15:42"): TableException(
        date(2026, 5, 17),
        SkipCategory.NO_TABLE,
        "alerts by health zone only",
    ),
    (428, "2026-05-22T23:40:37"): TableException(
        date(2026, 5, 19),
        SkipCategory.NO_TABLE,
        "alerts summary only",
    ),
    (24728, "2026-05-21T13:42:07"): TableException(
        date(2026, 5, 20),
        SkipCategory.NO_TABLE,
        "new counts by health zone only",
    ),
    (429, "2026-05-22T23:40:43"): TableException(
        date(2026, 5, 21),
        SkipCategory.NO_TABLE,
        "new counts by health zone only",
    ),
    (24821, "2026-05-23T12:32:28"): TableException(
        date(2026, 5, 22),
        SkipCategory.NO_TABLE,
        "new counts by health zone only",
    ),
    (24848, "2026-05-24T10:54:14"): TableException(
        date(2026, 5, 23),
        SkipCategory.NO_TABLE,
        "new counts by health zone only",
    ),
    (24853, "2026-05-25T17:20:48"): TableException(
        date(2026, 5, 24),
        SkipCategory.NO_TABLE,
        "counts by health zone only, no CFR",
    ),
    (24874, "2026-05-27T04:58:29"): TableException(
        date(2026, 5, 25),
        SkipCategory.NO_TABLE,
        "counts by health zone only, no CFR",
    ),
    (24880, "2026-05-27T12:11:28"): TableException(
        date(2026, 5, 26),
        SkipCategory.NO_TABLE,
        "province counts in the narrative only",
    ),
    (24892, "2026-05-28T18:16:59"): TableException(
        date(2026, 5, 27),
        SkipCategory.NO_TABLE,
        "counts by health zone with province subtotals, no CFR",
    ),
    (24895, "2026-05-30T14:03:03"): TableException(
        date(2026, 5, 28),
        SkipCategory.NO_TABLE,
        "no case table",
    ),
    (24901, "2026-05-31T09:07:00"): TableException(
        date(2026, 5, 29),
        SkipCategory.NO_TABLE,
        "counts by health zone with province subtotals",
    ),
    (24912, "2026-05-31T20:25:01"): TableException(
        date(2026, 5, 30),
        SkipCategory.NO_TABLE,
        "province counts in the narrative only",
    ),
    (24994, "2026-06-12T18:14:15"): TableException(
        date(2026, 6, 11),
        SkipCategory.PARSER_MISREAD,
        'footnote mark in "Nord-Kivu1" hides the row: provinces sum to 649 cases, Total row says 689',
    ),
    (25024, "2026-06-18T12:27:44"): TableException(
        date(2026, 6, 16),
        SkipCategory.PARSER_MISREAD,
        'footnote mark in "Nord-Kivu1" hides the row: provinces sum to 802 cases, Total row says 875',
    ),
    (25028, "2026-06-18T19:14:38"): TableException(
        date(2026, 6, 17),
        SkipCategory.PARSER_MISREAD,
        "Total CFR printed 26,3 % for 232/896 (25.9 %): parser reads the contacts follow-up table instead (Ituri 4 659 / 3 297)",
    ),
    (25077, "2026-07-03T06:22:09"): TableException(
        date(2026, 7, 1),
        SkipCategory.PARSER_MISREAD,
        "Total CFR printed 30,9 % for 452/1460 (30.96 %): parser reads the contacts follow-up table instead (Ituri 8 376 / 6 973)",
    ),
    (25105, "2026-07-07T23:43:04"): TableException(
        date(2026, 7, 6),
        SkipCategory.PARSER_MISREAD,
        "Ituri CFR printed 31,5 % for 491/1556 (31.55 %) drops the row: provinces sum to 152 cases, Total row says 1708",
    ),
    (25133, "2026-07-12T11:09:42"): TableException(
        date(2026, 7, 10),
        SkipCategory.PUBLISHER_ERROR,
        "Total row leaves out Haut-Uélé and Tshopo: provinces sum to 1878 cases, Total row says 1873",
    ),
    (25144, "2026-07-14T07:14:55"): TableException(
        date(2026, 7, 12),
        SkipCategory.PUBLISHER_ERROR,
        "provinces sum to 719 deaths, Total row says 718",
    ),
    (25185, "2026-07-22T23:56:33"): TableException(
        date(2026, 7, 21),
        SkipCategory.PUBLISHER_ERROR,
        "Total row says 2 531 cases, provinces and the CFR give 2 536: no province table found",
    ),
    (25216, "2026-07-28T03:22:20"): TableException(
        date(2026, 7, 26),
        SkipCategory.PUBLISHER_ERROR,
        "Haut-Uélé CFR printed 52,6 % for 21/38; provinces sum to 3261 cases, Total row says 3262",
    ),
    (25219, "2026-07-29T06:55:06"): TableException(
        date(2026, 7, 27),
        SkipCategory.PARSER_MISREAD,
        "Haut-Uélé CFR printed 52,6 % for 21/39 drops the row (counts sum): provinces sum to 8 cases, Total row says 3360",
    ),
    (25269, "2026-08-04T20:34:24"): TableException(
        date(2026, 8, 3),
        SkipCategory.PARSER_MISREAD,
        "Total CFR printed 45,1 % for 1751/3874 (45.20 %): parser reads the contacts follow-up table instead (Ituri 11 544 / 9 255)",
    ),
    (528, "2026-08-17T00:16:19"): TableException(
        date(2026, 8, 14),
        SkipCategory.NO_TABLE,
        "one-off layout with no CFR or Total row to validate",
    ),
    (25373, "2026-08-23T19:05:07"): TableException(
        date(2026, 8, 21),
        SkipCategory.PARSER_MISREAD,
        'Total row wraps "(37,7 %)" over two lines: no province table found',
    ),
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
        "modified": pl.String,
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
            i += WRAPPED_ROW_LINES
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
            len(head[0]) > THOUSANDS_GROUP_DIGITS
            or head[0].startswith("0")
            or len(head[-1]) != THOUSANDS_GROUP_DIGITS
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
        if len(values) == COUNT_COLUMNS_WITH_NEW and not new_missing:
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
    raise ValueError(
        f"No province table found; row errors: {errors[:MAX_ERRORS_SHOWN]}"
    )


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
    new = [c.new_cases for c in provinces if c.new_cases is not None]
    if total.new_cases is not None and len(new) == len(provinces):
        if sum(new) != total.new_cases:
            raise ValueError(
                f"Provinces sum to {sum(new)} new cases, Total row says "
                f"{total.new_cases}"
            )


def table_to_frame(
    table: dict[str, ProvinceCounts],
    *,
    report_date: date,
    sitrep_number: int | None,
    media_id: int,
    modified: str,
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
            "modified": modified,
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
    text: str,
    *,
    report_date: date,
    sitrep_number: int | None,
    media_id: int,
    modified: str,
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
        modified=modified,
    )


def check_monotonic(
    df: pl.DataFrame, *, by: tuple[str, ...] = ("province",), label: str = ""
) -> None:
    """Log any fall in a cumulative count between dates.

    Shared with `drc_2026_inrb`. New-case rows and null values are skipped.

    Args:
        df: Long rows with `date`, `indicator`, `value` and the `by` columns.
        by: Columns that, with `indicator`, identify one series.
        label: Prefix for the log message, such as the source name.
    """
    series = df.filter(pl.col("indicator") != INDICATORS["new_cases"])
    nulls = series.filter(pl.col("value").is_null())
    if nulls.height:
        logger.info(
            "%s: skipping %d null values in the monotonic check, first %d: %s",
            label or "Monotonic check",
            nulls.height,
            MAX_NULLS_LISTED,
            nulls.select("date", *by, "indicator").head(MAX_NULLS_LISTED).rows(),
        )
    cumulative = series.drop_nulls("value").sort("date")
    for keys, group in cumulative.group_by(*by, "indicator", maintain_order=True):
        *entity, indicator = keys
        name = " ".join(
            str(part) for part in (label, *entity, str(indicator).lower()) if part
        )
        for (d0, v0), (d1, v1) in pairwise(group.select("date", "value").iter_rows()):
            if v1 < v0:
                logger.warning("%s falls from %d (%s) to %d (%s)", name, v0, d0, v1, d1)


def ensure_pdf(*, media_id: int, modified: str, source_url: str) -> Path:
    """Return a kept sitrep's local PDF, downloading it if absent.

    The CI runner starts with no raw files, so a sitrep that needs parsing
    may not have been downloaded in this run.
    """
    item: MediaItem = {
        "id": media_id,
        "modified": modified,
        "source_url": source_url,
        "mime_type": PDF_MIME,
    }
    return download(item)


def stale_exception_note(*, media_id: int, modified: str) -> str:
    """Name any exception for the same upload under a different `modified`.

    INSP sometimes touches `modified` on old files without changing them, which
    moves an excepted upload out of its exception.
    """
    stale = [
        f"{key_modified} ({skip.category}: {skip.reason})"
        for (key_id, key_modified), skip in PROVINCE_TABLE_EXCEPTIONS.items()
        if key_id == media_id and key_modified != modified
    ]
    if not stale:
        return ""
    return (
        f"; media {media_id} has an exception for modified {', '.join(stale)},"
        f" but the manifest says {modified}"
    )


def warn_stale_exceptions(manifest: pl.DataFrame) -> None:
    """Log any exception whose upload is not a latest sitrep in the manifest."""
    latest = set(manifest.select("media_id", "modified").iter_rows())
    for key, skip in PROVINCE_TABLE_EXCEPTIONS.items():
        if key not in latest:
            logger.warning(
                "Exception for media %d modified %s (%s) matches no latest sitrep",
                *key,
                skip.report_date,
            )


def main() -> None:
    """Parse every kept sitrep not yet in the output and update the CSV.

    Uploads in `PROVINCE_TABLE_EXCEPTIONS` are skipped. One whose media id is
    excepted under a different `modified` fails without being parsed. Writes
    nothing if any sitrep fails, so a partial run cannot drop dates.
    """
    manifest = pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA).filter("is_latest")
    existing = (
        pl.read_csv(OUTPUT_PATH, schema=SCHEMA)
        if OUTPUT_PATH.exists()
        else pl.DataFrame(schema=SCHEMA)
    )
    # Keyed on `modified` too, so an upload replaced in place is parsed again.
    done = set(existing.select("date", "media_id", "modified").unique().iter_rows())

    warn_stale_exceptions(manifest)

    frames = [existing]
    failures: dict[date, str] = {}
    for row in manifest.sort("report_date").iter_rows(named=True):
        report_date = row["report_date"]
        if (report_date, row["media_id"], row["modified"]) in done:
            continue
        key = (row["media_id"], row["modified"])
        # Some excepted uploads parse "successfully" from the wrong table, so one
        # with a touched `modified` fails whatever the parser does.
        if key not in PROVINCE_TABLE_EXCEPTIONS and (
            note := stale_exception_note(media_id=key[0], modified=key[1])
        ):
            failures[report_date] = f"not parsed{note}"
            continue
        if key in PROVINCE_TABLE_EXCEPTIONS:
            skip = PROVINCE_TABLE_EXCEPTIONS[key]
            logger.info(
                "Skipping %s (media %d, %s): %s",
                report_date,
                row["media_id"],
                skip.category,
                skip.reason,
            )
            continue
        try:
            frames.append(
                parse_sitrep(
                    read_text(
                        ensure_pdf(
                            media_id=row["media_id"],
                            modified=row["modified"],
                            source_url=row["source_url"],
                        )
                    ),
                    report_date=report_date,
                    sitrep_number=row["sitrep_number"],
                    media_id=row["media_id"],
                    modified=row["modified"],
                )
            )
        except ValueError as e:
            failures[report_date] = str(e) + stale_exception_note(
                media_id=row["media_id"], modified=row["modified"]
            )

    if failures:
        for report_date, reason in failures.items():
            logger.error("Failed %s: %s", report_date, reason)
        raise ValueError(f"{len(failures)} sitreps could not be parsed")

    # A newer or replaced upload for a date supersedes rows parsed earlier.
    latest = manifest.select(
        pl.col("report_date").alias("date"), "media_id", "modified"
    )
    out = (
        pl.concat(frames)
        .join(latest, on=["date", "media_id", "modified"], how="semi")
        .sort("date", "province", "indicator")
    )
    if out.select("date", "province", "indicator").is_duplicated().any():
        raise ValueError("Duplicate (date, province, indicator) rows")
    lost = set(existing["date"].to_list()) - set(out["date"].to_list())
    if lost:
        raise ValueError(f"Dates would drop out of the output: {sorted(lost)}")

    check_monotonic(out)
    out.write_csv(OUTPUT_PATH)
    logger.info(
        "Wrote %d rows for %d sitreps to %s",
        out.height,
        out["date"].n_unique(),
        OUTPUT_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
