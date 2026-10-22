"""Find and download DRC Ministry of Health sitreps for the 2026 outbreak.

The ministry's site (https://sante.gouv.cd/documents/sitreps) is a front end
over WordPress, whose media API lists more sitreps than the listing page.
Filenames are unreliable, so the report date comes from the PDF header. The
header has two dates: the printed report date and the date inside the sitrep
reference ("SitRep N°150 /MVE-BDBV/11/10/2026"). Usually they agree. When they
do not, the date that fits the nearest sitrep whose two dates agree, counted
forward by sitrep number, wins. Printed sitrep numbers are themselves
occasionally wrong (two files print "N°153"), so they are kept for reference
only. The manifest keeps one row per media item and flags the file kept for
each report date.
"""

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple, TypedDict
from urllib.parse import unquote

import pdfplumber
import polars as pl

from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTBREAK = "drc_2026"
MEDIA_API = "https://administration.sante.gouv.cd/wp-json/wp/v2/media"
MEDIA_QUERY = "?search=SitRep&per_page=100&_fields=id,modified,source_url,mime_type"
SITREP_DIR = ROOT / "data" / "raw" / OUTBREAK / "sitreps"
MANIFEST_PATH = ROOT / "data" / "processed" / f"{OUTBREAK}_sitrep_manifest.csv"

PDF_MIME = "application/pdf"

FRENCH_MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
    "decembre": 12,
}
FRENCH_DATE = r"(\d{1,2})(?:er)?\s+([a-zéû]+)\s+(\d{4})"
REPORT_DATE = re.compile(rf"Date d(?:u|e) rapport(?:age)?\s*:?\s*{FRENCH_DATE}", re.I)
PUBLISH_DATE = re.compile(rf"Date de publication\s*:?\s*{FRENCH_DATE}", re.I)
# "SitRep N°157 /MVE-BDBV/18/10/2026", "SitRep N°32/MVB_15/06/2026",
# "SitRep MVE N° 007/MVB_17/2026" (no date).
SITREP_REF = re.compile(r"SitRep\s+(?:MVE\s+)?N°\s*(\d+)([^\n]*)", re.I)
NUMERIC_DATE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")

MANIFEST_SCHEMA = pl.Schema(
    {
        "outbreak": pl.String,
        "report_date": pl.Date,
        "sitrep_number": pl.Int64,
        "printed_report_date": pl.Date,
        "reference_date": pl.Date,
        "publish_date": pl.Date,
        "is_latest": pl.Boolean,
        "in_api": pl.Boolean,
        "media_id": pl.Int64,
        "modified": pl.String,
        "file_name": pl.String,
        "source_url": pl.String,
    }
)


class MediaItem(TypedDict):
    """WordPress media API item, limited to the requested fields."""

    id: int
    modified: str
    source_url: str
    mime_type: str


class SitrepHeader(NamedTuple):
    """Fields read from a sitrep's first page; None when absent."""

    sitrep_number: int | None
    report_date: date | None
    reference_date: date | None
    publish_date: date | None


def parse_french_date(day: str, month: str, year: str) -> date:
    """Build a date from French day, month name and year strings.

    Raises:
        ValueError: If the month name is not recognised.
    """
    month_number = FRENCH_MONTHS.get(month.lower())
    if month_number is None:
        raise ValueError(f"Unrecognised French month: {month!r}")
    return date(int(year), month_number, int(day))


def parse_header(text: str) -> SitrepHeader:
    """Read the sitrep number and dates from first-page text.

    Args:
        text: Extracted text of the sitrep's first page.

    Returns:
        The header fields found; missing fields are None.
    """
    number = reference_date = report_date = publish_date = None
    if ref := SITREP_REF.search(text):
        number = int(ref.group(1))
        if numeric := NUMERIC_DATE.search(ref.group(2)):
            day, month, year = numeric.groups()
            reference_date = date(int(year), int(month), int(day))
    if match := REPORT_DATE.search(text):
        report_date = parse_french_date(*match.groups())
    if match := PUBLISH_DATE.search(text):
        publish_date = parse_french_date(*match.groups())
    return SitrepHeader(number, report_date, reference_date, publish_date)


def find_anchors(headers: list[SitrepHeader]) -> list[tuple[int, date]]:
    """Return (sitrep number, date) for headers whose two dates agree."""
    return [
        (h.sitrep_number, h.report_date)
        for h in headers
        if h.sitrep_number is not None
        and h.report_date is not None
        and h.report_date == h.reference_date
    ]


def resolve_report_date(
    header: SitrepHeader, *, anchors: list[tuple[int, date]]
) -> date:
    """Pick the report date for a sitrep, settling header conflicts by neighbours.

    Args:
        header: Parsed header fields.
        anchors: (sitrep number, date) pairs from sitreps whose header dates agree.

    Returns:
        The report date.

    Raises:
        ValueError: If the header has no date, or its two dates disagree and
            neither fits the nearest anchor.
    """
    printed, reference = header.report_date, header.reference_date
    if printed is None or reference is None or printed == reference:
        if resolved := printed or reference:
            return resolved
        raise ValueError(f"No date found in header: {header}")

    sitrep_number = header.sitrep_number
    if sitrep_number is None or not anchors:
        raise ValueError(f"Header dates disagree and no anchor to settle: {header}")
    number, anchor_date = min(anchors, key=lambda a: abs(a[0] - sitrep_number))
    expected = anchor_date + timedelta(days=sitrep_number - number)
    if expected not in {printed, reference}:
        raise ValueError(f"Neither header date fits sitrep #{number}: {header}")
    logger.warning(
        "Sitrep %s prints report date %s and reference date %s; using %s",
        header.sitrep_number,
        printed,
        reference,
        expected,
    )
    return expected


def read_first_page(path: Path) -> str:
    """Extract the text of a PDF's first page."""
    with pdfplumber.open(path) as pdf:
        return pdf.pages[0].extract_text() or ""


def list_media() -> list[MediaItem]:
    """List every sitrep PDF in the WordPress media library.

    Returns:
        Media items with id, modified, source_url and mime_type.
    """
    items: list[MediaItem] = []
    page, total_pages = 1, 1
    while page <= total_pages:
        response = get(f"{MEDIA_API}{MEDIA_QUERY}&page={page}")
        total_pages = int(response.headers["X-WP-TotalPages"])
        items.extend(response.json())
        page += 1

    pdfs = [item for item in items if item["mime_type"] == PDF_MIME]
    for item in items:
        if item["mime_type"] != PDF_MIME:
            logger.warning("Ignoring non-PDF media item: %s", item["source_url"])
    logger.info("Media API lists %d sitrep PDFs", len(pdfs))
    return pdfs


def download(item: MediaItem, *, refresh: bool = False) -> Path:
    """Download a media item to the sitrep folder unless already present.

    Files are named by media id because source filenames contain "N°" and
    duplicate each other across uploads.

    Args:
        item: Media item to fetch.
        refresh: Download even if a file exists, for an upload replaced in place.

    Returns:
        Path to the local PDF.
    """
    path = SITREP_DIR / f"{item['id']}.pdf"
    if refresh or not path.exists():
        path.write_bytes(get(item["source_url"]).content)
    return path


def manifest_row(item: MediaItem, header: SitrepHeader) -> dict[str, object]:
    """Combine a media item and its parsed header into an unresolved manifest row."""
    source_url = item["source_url"]
    return {
        "outbreak": OUTBREAK,
        "report_date": None,
        "sitrep_number": header.sitrep_number,
        "printed_report_date": header.report_date,
        "reference_date": header.reference_date,
        "publish_date": header.publish_date,
        "is_latest": False,
        "in_api": True,
        "media_id": item["id"],
        "modified": item["modified"],
        "file_name": unquote(source_url.rsplit("/", 1)[-1]),
        "source_url": source_url,
    }


def resolve_report_dates(manifest: pl.DataFrame) -> pl.DataFrame:
    """Fill `report_date` for every file, using the other files as anchors."""
    headers = [
        SitrepHeader(*row)
        for row in manifest.select(
            "sitrep_number", "printed_report_date", "reference_date", "publish_date"
        ).iter_rows()
    ]
    anchors = find_anchors(headers)
    resolved = [resolve_report_date(h, anchors=anchors) for h in headers]
    return manifest.with_columns(report_date=pl.Series(resolved, dtype=pl.Date))


def flag_latest(manifest: pl.DataFrame) -> pl.DataFrame:
    """Mark the most recently modified file for each report date as the one to use."""
    latest = (pl.col("modified") == pl.col("modified").max().over("report_date")) & (
        pl.col("media_id") == pl.col("media_id").max().over("report_date", "modified")
    )
    flagged = manifest.with_columns(is_latest=latest).sort("report_date", "media_id")

    duplicates = flagged.filter(pl.len().over("report_date") > 1)
    for (report_date,), group in duplicates.group_by(
        "report_date", maintain_order=True
    ):
        kept = group.filter("is_latest")["file_name"].item()
        logger.info("%s has %d files, keeping %s", report_date, group.height, kept)
    return flagged


def log_gaps(manifest: pl.DataFrame) -> None:
    """Log report dates with no sitrep between the first and last available."""
    dates = set(manifest["report_date"].to_list())
    first, last = min(dates), max(dates)
    missing = [
        first + timedelta(days=i)
        for i in range((last - first).days + 1)
        if first + timedelta(days=i) not in dates
    ]
    logger.warning(
        "%d of %d days between %s and %s have no sitrep",
        len(missing),
        (last - first).days + 1,
        first,
        last,
    )


def build_manifest(previous: pl.DataFrame | None) -> pl.DataFrame:
    """Build the manifest, downloading only media items that are new or changed.

    Rows for media the API no longer lists are kept with `in_api` False, so a
    removed upload never deletes its date from the outputs.

    Args:
        previous: The existing manifest, or None on a first run.

    Returns:
        One row per sitrep PDF with `is_latest` flagged per report date.
    """
    previous_rows = previous.to_dicts() if previous is not None else []
    known = {(row["media_id"], row["modified"]): row for row in previous_rows}
    known_ids = {row["media_id"] for row in previous_rows}

    rows: list[dict[str, object]] = []
    for item in list_media():
        if cached := known.get((item["id"], item["modified"])):
            rows.append(cached | {"in_api": True})
            continue
        # A known id with a new `modified` time is an upload replaced in place.
        replaced = item["id"] in known_ids
        path = download(item, refresh=replaced)
        rows.append(manifest_row(item, parse_header(read_first_page(path))))
        logger.info("%s %s", "Replaced" if replaced else "Added", rows[-1]["file_name"])

    listed_ids = {row["media_id"] for row in rows}
    for row in previous_rows:
        if row["media_id"] not in listed_ids:
            logger.warning(
                "%s (%s) is no longer listed by the API; keeping it",
                row["file_name"],
                row["report_date"],
            )
            rows.append(row | {"in_api": False})

    manifest = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
    return flag_latest(resolve_report_dates(manifest))


def main() -> None:
    """Refresh the sitrep manifest and download any new PDFs."""
    SITREP_DIR.mkdir(parents=True, exist_ok=True)
    previous = (
        pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA)
        if MANIFEST_PATH.exists()
        else None
    )
    manifest = build_manifest(previous)
    log_gaps(manifest.filter("is_latest"))
    manifest.write_csv(MANIFEST_PATH)
    logger.info(
        "Wrote %d files (%d report dates) to %s",
        manifest.height,
        manifest["is_latest"].sum(),
        MANIFEST_PATH.name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
