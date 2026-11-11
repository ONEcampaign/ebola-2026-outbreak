"""Find and download the DRC sitreps of the 2026 outbreak from INSP.

INSP (https://insp.cd/category/sitrep/) publishes each sitrep first. Its
site is a WordPress front end whose media API lists every upload, other
outbreaks' sitreps included, so only files marked "MVEBDB", or "MVE_RDC" from
sitrep 169 on, are kept. INSP renames its files often (sitrep 179 dropped
"MVE_RDC"), so a PDF attached to a post in INSP's SitRep category is kept
whatever its name.

Those posts are also the check that nothing was missed. Each is titled with
the sitrep's reference ("SitRep N°179/MVE-BDBV/09/11/2026"), so `check`
compares the newest posted sitrep number with the newest in the manifest and
fails when the manifest is behind.

The report date comes from the PDF header, since filenames are unreliable. The
header has two dates: the printed report date and the date inside the sitrep
reference ("SitRep N°150 /MVE-BDBV/11/10/2026"). Usually they agree. When they
do not, the date that fits the nearest sitrep whose two dates agree, counted
forward by sitrep number, wins. If equally near anchors disagree, the run
fails. A date is resolved once, when its upload is new or replaced, and kept
after that. Printed sitrep numbers are themselves occasionally wrong (two
files print "N°153"), so they are kept for reference only. The manifest keeps
one row per media item and flags the file kept for each report date.

The manifest also holds files from the Ministry of Health's media API
(https://sante.gouv.cd/documents/sitreps), which published sitreps a day or
more after INSP and was last read on 28 October 2026. A date that has a
ministry file keeps it, so no processed report changes file.
"""

import argparse
import html
import logging
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict, cast
from urllib.parse import unquote

import httpx
import pdfplumber
import polars as pl

from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTBREAK = "drc_2026"
INSP_MEDIA_API = "https://insp.cd/wp-json/wp/v2/media"
INSP_POSTS_API = "https://insp.cd/wp-json/wp/v2/posts"
# INSP's "SitRep" post category; every post in it is one of this outbreak's.
INSP_SITREP_CATEGORY_ID = 308
POSTS_QUERY = (
    f"?categories={INSP_SITREP_CATEGORY_ID}&per_page=100&_fields=id,date_gmt,title"
)
# INSP sometimes posts a sitrep shortly before or after attaching its PDF, so a
# post younger than this does not fail the check; the next run catches it.
POST_GRACE = timedelta(hours=3)
# Host of the ministry files kept from before INSP became the only source.
MINISTRY_SITE = "https://administration.sante.gouv.cd/"
HTTP_NOT_FOUND = 404
# INSP's library also holds other outbreaks' sitreps; this outbreak's carry
# "MVEBDB" in the file name.
INSP_FILE_MARKER = "MVEBDB"
# From sitrep 169 (30 October 2026) INSP names files "MVE_RDC" instead. Its
# uploads before August used that name too, and the ministry's files already
# cover those dates, so the marker, and attachment to a SitRep post, count
# only from the first renamed upload.
INSP_RENAMED_MARKER = "MVE_RDC"
INSP_RENAMED_FIRST_MEDIA_ID = 25808
MEDIA_QUERY = (
    "?search=SitRep&per_page=100&_fields=id,modified,source_url,mime_type,post"
)
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
    # The post the file is attached to; None when unattached. Absent from
    # items rebuilt from manifest rows, which only need the file.
    post: NotRequired[int | None]


class SitrepPost(NamedTuple):
    """A post in INSP's SitRep category, with the sitrep number in its title."""

    post_id: int
    sitrep_number: int
    title: str
    published: datetime


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
        ValueError: If the header has no date, if its two dates disagree and
            the nearest anchors expect different dates (anchors follow more
            than one number-to-date offset), or if neither date fits.
    """
    printed, reference = header.report_date, header.reference_date
    if printed is None or reference is None or printed == reference:
        if resolved := printed or reference:
            return resolved
        raise ValueError(f"No date found in header: {header}")

    sitrep_number = header.sitrep_number
    if sitrep_number is None or not anchors:
        raise ValueError(f"Header dates disagree and no anchor to settle: {header}")
    distance = min(abs(number - sitrep_number) for number, _ in anchors)
    nearest = [a for a in anchors if abs(a[0] - sitrep_number) == distance]
    candidates = {
        anchor_date + timedelta(days=sitrep_number - number)
        for number, anchor_date in nearest
    }
    if len(candidates) > 1:
        raise ValueError(
            f"Nearest anchors {sorted(nearest)} expect different dates "
            f"{sorted(candidates)}: {header}"
        )
    expected = candidates.pop()
    if expected not in {printed, reference}:
        raise ValueError(f"Neither header date fits the expected {expected}: {header}")
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


def list_wordpress(url: str) -> list[dict[str, object]]:
    """Fetch every page of a WordPress REST listing.

    Args:
        url: Listing URL with its query string, without `page`.

    Returns:
        The items of all pages, in the API's order.
    """
    items: list[dict[str, object]] = []
    page, total_pages = 1, 1
    while page <= total_pages:
        response = get(f"{url}&page={page}")
        total_pages = int(response.headers["X-WP-TotalPages"])
        items.extend(response.json())
        page += 1
    return items


def list_api_pdfs(api: str) -> list[MediaItem]:
    """List every sitrep PDF in one WordPress media library.

    Returns:
        Media items with id, modified, source_url, mime_type and post.
    """
    items = cast(list[MediaItem], list_wordpress(f"{api}{MEDIA_QUERY}"))
    return [item for item in items if item["mime_type"] == PDF_MIME]


def parse_post(raw: dict[str, object]) -> SitrepPost | None:
    """Read a SitRep post's id, sitrep number and publication time.

    Returns:
        The post, or None (logged) when its title has no sitrep number.
    """
    title_field = raw["title"]
    if not isinstance(title_field, dict):
        raise ValueError(f"Post {raw['id']} has no rendered title: {raw!r}")
    title = html.unescape(str(title_field["rendered"])).strip()
    match = SITREP_REF.search(title)
    if match is None:
        logger.warning("INSP post %s has no sitrep number: %r", raw["id"], title)
        return None
    published = datetime.fromisoformat(str(raw["date_gmt"])).replace(tzinfo=UTC)
    return SitrepPost(int(str(raw["id"])), int(match.group(1)), title, published)


def list_sitrep_posts() -> list[SitrepPost]:
    """List the posts in INSP's SitRep category that name a sitrep number."""
    raw = list_wordpress(f"{INSP_POSTS_API}{POSTS_QUERY}")
    posts = [post for item in raw if (post := parse_post(item)) is not None]
    logger.info("INSP lists %d SitRep posts", len(posts))
    return posts


def is_this_outbreak(item: MediaItem, *, sitrep_post_ids: set[int]) -> bool:
    """Whether a media item is one of this outbreak's sitreps.

    It is when its name carries "MVEBDB" or, from INSP_RENAMED_FIRST_MEDIA_ID
    on, when its name carries "MVE_RDC" or it is attached to a SitRep post.
    """
    url = item["source_url"].upper()
    if INSP_FILE_MARKER in url:
        return True
    if item["id"] < INSP_RENAMED_FIRST_MEDIA_ID:
        return False
    return INSP_RENAMED_MARKER in url or item.get("post") in sitrep_post_ids


def list_media(posts: list[SitrepPost]) -> list[MediaItem]:
    """List this outbreak's sitrep PDFs in INSP's media library.

    Args:
        posts: INSP's SitRep posts; a PDF attached to one is kept by any name.

    Returns:
        Media items with id, modified, source_url, mime_type and post.
    """
    listed = list_api_pdfs(INSP_MEDIA_API)
    post_ids = {post.post_id for post in posts}
    items = [
        item for item in listed if is_this_outbreak(item, sitrep_post_ids=post_ids)
    ]
    logger.info("INSP media API lists %d sitrep PDFs", len(items))
    warn_newer_rejected(listed, kept=items)
    return items


def warn_newer_rejected(listed: list[MediaItem], *, kept: list[MediaItem]) -> None:
    """Warn about sitrep PDFs uploaded after the newest kept one but filtered out.

    INSP has renamed its files before. A renamed sitrep attached to its post is
    kept, but an unattached one would be dropped, so newer rejected uploads are
    logged for a person to check. `check` fails the run in that case.
    """
    newest_kept = max((item["id"] for item in kept), default=0)
    for item in listed:
        if item["id"] > newest_kept and item not in kept:
            logger.warning(
                "INSP media %s is newer than every kept sitrep but its name "
                "does not match this outbreak; check whether INSP renamed its "
                "files: %s",
                item["id"],
                item["source_url"],
            )


def is_ministry(source_url: str) -> bool:
    """Whether a file came from the ministry's media API."""
    return source_url.startswith(MINISTRY_SITE)


def local_pdf_name(item: MediaItem) -> str:
    """Name a media item's local PDF from its id and `modified` stamp.

    Keying on `modified` as well as the id gives an upload replaced in place
    a new file name, so each version of an upload has its own local copy.

    Raises:
        ValueError: If `modified` has no letters or digits to name the file by.
    """
    stamp = re.sub(r"[^0-9A-Za-z]+", "", item["modified"])
    if not stamp:
        raise ValueError(f"Media {item['id']} has no usable modified stamp")
    return f"{item['id']}_{stamp}.pdf"


def download(item: MediaItem, *, refresh: bool = False) -> Path:
    """Download a media item to the sitrep folder unless already present.

    Files are named by media id and `modified` stamp because source filenames
    contain "N°" and duplicate each other across uploads.

    Args:
        item: Media item to fetch.
        refresh: Download even if a file exists.

    Returns:
        Path to the local PDF.
    """
    path = SITREP_DIR / local_pdf_name(item)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
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
    """Fill `report_date` where it is null, using every file as an anchor.

    Only new and replaced uploads have a null date. A stored date is kept, so
    a new anchor never moves the date of a report already processed. The cost
    is that a later run leaves a wrong stored date as it is. To fix one, edit
    `report_date` for that upload in both the manifest and
    data/processed/drc_2026_sitrep_extraction_status.csv, since release refuses
    a mismatch between them.
    """
    headers = [
        SitrepHeader(*row)
        for row in manifest.select(
            "sitrep_number", "printed_report_date", "reference_date", "publish_date"
        ).iter_rows()
    ]
    anchors = find_anchors(headers)
    resolved = [
        stored if stored is not None else resolve_report_date(h, anchors=anchors)
        for stored, h in zip(manifest["report_date"], headers, strict=True)
    ]
    return manifest.with_columns(report_date=pl.Series(resolved, dtype=pl.Date))


def flag_latest(manifest: pl.DataFrame) -> pl.DataFrame:
    """Mark the file to use for each report date.

    A ministry file kept from before INSP became the only source beats an INSP
    one, so a processed date keeps its file; among files from the same site,
    the most recently modified wins.
    """
    no_url = manifest.filter(pl.col("source_url").is_null())
    if no_url.height:
        logger.warning(
            "%d manifest rows have no source_url and rank as non-ministry: %s",
            no_url.height,
            no_url["media_id"].to_list(),
        )
    from_ministry = pl.col("source_url").str.starts_with(MINISTRY_SITE).fill_null(False)
    ranked = manifest.sort(
        from_ministry, "modified", "media_id", descending=True
    ).with_columns(is_latest=pl.int_range(pl.len()).over("report_date") == 0)
    flagged = ranked.sort("report_date", "media_id")

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


def build_manifest(
    previous: pl.DataFrame | None, *, posts: list[SitrepPost]
) -> pl.DataFrame:
    """Build the manifest, downloading only media items that are new or changed.

    Rows for media the API no longer lists, and the ministry's files, are
    kept with `in_api` False, so a removed upload never deletes its date from
    the outputs.

    Args:
        previous: The existing manifest, or None on a first run.
        posts: INSP's SitRep posts, whose attached PDFs are kept by any name.

    Returns:
        One row per sitrep PDF with `is_latest` flagged per report date.

    Raises:
        ValueError: If an INSP media id is also a ministry file's, since local
            files and the extraction ledger are keyed by media id alone.
    """
    previous_rows = previous.to_dicts() if previous is not None else []
    known = {(row["media_id"], row["modified"]): row for row in previous_rows}
    known_ids = {row["media_id"] for row in previous_rows}
    ministry_ids = {
        r["media_id"] for r in previous_rows if is_ministry(r["source_url"])
    }

    items = list_media(posts)
    clashes = {item["id"] for item in items} & ministry_ids
    if clashes:
        raise ValueError(f"INSP media ids clash with ministry ids: {sorted(clashes)}")

    rows: list[dict[str, object]] = []
    for item in items:
        if cached := known.get((item["id"], item["modified"])):
            rows.append(cached | {"in_api": True})
            continue
        # A known id with a new `modified` time is an upload replaced in place.
        replaced = item["id"] in known_ids
        try:
            path = download(item, refresh=replaced)
        except httpx.HTTPStatusError as e:
            # INSP lists some uploads whose file is gone.
            if e.response.status_code != HTTP_NOT_FOUND:
                raise
            logger.warning("Skipping INSP media %s: file not found", item["id"])
            continue
        rows.append(manifest_row(item, parse_header(read_first_page(path))))
        logger.info("%s %s", "Replaced" if replaced else "Added", rows[-1]["file_name"])

    listed_ids = {row["media_id"] for row in rows}
    for row in previous_rows:
        if row["media_id"] not in listed_ids:
            if not is_ministry(row["source_url"]):
                logger.warning(
                    "%s (%s) is no longer listed by the API; keeping it",
                    row["file_name"],
                    row["report_date"],
                )
            rows.append(row | {"in_api": False})

    manifest = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
    return flag_latest(resolve_report_dates(manifest))


def find_unscraped_latest(
    manifest: pl.DataFrame, posts: list[SitrepPost], *, now: datetime
) -> SitrepPost | None:
    """Return INSP's newest posted sitrep if the manifest lacks it.

    Posts younger than POST_GRACE are ignored, since INSP may attach the PDF
    after posting. Printed sitrep numbers are occasionally wrong, so a failure
    can also mean a misnumbered PDF; the message names both to check.

    Args:
        manifest: The sitrep manifest.
        posts: INSP's SitRep posts.
        now: Current time, timezone-aware.

    Returns:
        The newest settled post when its number is above every manifest
        sitrep number, else None.

    Raises:
        ValueError: If there is no settled post or the manifest has no numbers.
    """
    settled = [post for post in posts if now - post.published >= POST_GRACE]
    if not settled:
        raise ValueError("INSP lists no SitRep post older than the grace period")
    latest_post = max(settled, key=lambda post: post.sitrep_number)
    numbers = [n for n in manifest["sitrep_number"].to_list() if n is not None]
    if not numbers:
        raise ValueError("The manifest has no sitrep numbers")
    latest_scraped: int = max(numbers)
    logger.info(
        "Latest posted sitrep %d (%s); latest scraped %d",
        latest_post.sitrep_number,
        latest_post.title,
        latest_scraped,
    )
    return latest_post if latest_post.sitrep_number > latest_scraped else None


def check_latest() -> int:
    """Compare INSP's newest posted sitrep with the manifest.

    Returns:
        0 when the manifest has it, 1 when it does not.
    """
    manifest = pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA)
    missing = find_unscraped_latest(
        manifest, list_sitrep_posts(), now=datetime.now(UTC)
    )
    if missing is None:
        return 0
    latest_scraped = manifest.filter(
        pl.col("sitrep_number") == pl.col("sitrep_number").max()
    ).row(0, named=True)
    # A GitHub Actions error annotation, so the run summary shows it.
    print(
        f"::error::INSP posted sitrep {missing.sitrep_number} ({missing.title}, "
        f"post {missing.post_id}) but the newest in the manifest is "
        f"{latest_scraped['sitrep_number']} ({latest_scraped['file_name']}). "
        "Its PDF is not attached to the post or was filtered out; see the "
        "download step's warnings.",
        flush=True,
    )
    return 1


def download_all() -> None:
    """Refresh the sitrep manifest and download any new PDFs."""
    previous = (
        pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA)
        if MANIFEST_PATH.exists()
        else None
    )
    manifest = build_manifest(previous, posts=list_sitrep_posts())
    log_gaps(manifest.filter("is_latest"))
    manifest.write_csv(MANIFEST_PATH)
    logger.info(
        "Wrote %d files (%d report dates) to %s",
        manifest.height,
        manifest["is_latest"].sum(),
        MANIFEST_PATH.name,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return 1 when the check finds a missed sitrep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        nargs="?",
        choices=("download", "check"),
        default="download",
        help="download (default) refreshes the manifest; check fails when INSP "
        "has posted a newer sitrep than the manifest holds",
    )
    args = parser.parse_args(argv)
    if args.command == "check":
        return check_latest()
    download_all()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
