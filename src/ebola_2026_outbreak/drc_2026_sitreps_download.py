"""Find and download the DRC sitreps of the 2026 outbreak from INSP.

INSP (https://insp.cd/category/sitrep/) publishes each sitrep first. Its
site is a WordPress front end whose media API lists every upload, other
outbreaks' sitreps included. Every PDF is listed and a file is kept when its
name carries "MVEBDB", "MVE_RDC" or "MVE-Ituri", or when it is attached to a
post in INSP's SitRep category. INSP renames its files often (sitrep 179
dropped "MVE_RDC"), so attachment to a SitRep post keeps a file whatever its
name.

Those posts are also the check that nothing was missed. `check` fails when a
settled SitRep post has no attached PDF in the manifest (POSTS_WITHOUT_PDF
lists the posts INSP left without one), and when a day between the first and
last report date has no sitrep (KNOWN_GAPS lists the days INSP published
none). The second part catches sitreps posted outside the category or left
unattached.

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
# SitRep posts INSP left without an attached PDF, by post id, with the reason
# `check` lets each pass.
POSTS_WITHOUT_PDF: dict[int, str] = {
    24781: "Sitrep 1 (14 May): no attached file. It is unattached media 24659, "
    "kept by its MVE-Ituri name.",
    24783: "Sitrep 2 (17 May): no attached file. It is unattached media 24706, "
    "kept by its MVE-Ituri name.",
    24785: "Sitrep 4 (19 May): no attached file. It is unattached media 24710, "
    "kept by its MVE-Ituri name.",
    24789: "Sitrep 5 (19 May): no attached file. It is unattached media 24714, "
    "kept by its MVE_RDC name.",
    24791: "Sitrep 6 (20 May): no attached file. It is unattached media 24728, "
    "kept by its MVE_RDC name.",
    24759: "Sitrep 6 (20 May), posted twice: no attached file. It is unattached "
    "media 24728, kept by its MVE_RDC name.",
}
# Days between the first and last report date with no sitrep, with the reason.
# For each, INSP has no post numbered for that day in any category and no file
# for it in its media library, and the ministry files do not cover it.
KNOWN_GAPS: dict[date, str] = {
    date(2026, 5, 15): "INSP published no sitrep: sitrep 1 is 14 May, sitrep 2 17 May",
    date(2026, 5, 16): "INSP published no sitrep: sitrep 1 is 14 May, sitrep 2 17 May",
    date(2026, 5, 18): "INSP published no sitrep 3",
    date(2026, 6, 12): "INSP published no sitrep 29",
    date(2026, 6, 26): "INSP published no sitrep 43",
    date(2026, 6, 28): "INSP published no sitrep 45",
    date(2026, 7, 28): "INSP published no sitrep 75",
    date(2026, 7, 29): "INSP published no sitrep 76",
    date(2026, 11, 2): "INSP published no sitrep 172",
}
# Host of the ministry files kept from before INSP became the only source.
MINISTRY_SITE = "https://administration.sante.gouv.cd/"
HTTP_NOT_FOUND = 404
# INSP's library also holds other outbreaks' sitreps. This outbreak's carry
# "MVEBDB" (August to October) or "MVE_RDC" (May to August, and from sitrep
# 169) in the file name. Every MVE_RDC file on INSP is this outbreak's. INSP's
# first sitreps (1, 2 and 4) are named "MVE-Ituri". The 2025 outbreak's files
# are named "MVE16" and match no marker.
INSP_FILE_MARKERS = ("MVEBDB", "MVE_RDC", "MVE-ITURI")
PDF_MIME = "application/pdf"
# Listed by MIME type, since a filename search misses renamed uploads.
MEDIA_QUERY = (
    f"?mime_type={PDF_MIME}&per_page=100&_fields=id,modified,source_url,mime_type,post"
)
SITREP_DIR = ROOT / "data" / "raw" / OUTBREAK / "sitreps"
MANIFEST_PATH = ROOT / "data" / "processed" / f"{OUTBREAK}_sitrep_manifest.csv"

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

    It is when its name carries one of INSP_FILE_MARKERS or it is attached to
    a SitRep post.
    """
    url = item["source_url"].upper()
    if any(marker in url for marker in INSP_FILE_MARKERS):
        return True
    return item.get("post") in sitrep_post_ids


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
    """Warn about SitRep-named PDFs newer than the newest kept one but rejected.

    The listing covers every PDF on INSP, so only rejected files with "SITREP"
    in the URL (any case) are logged. A renamed sitrep attached to its post is
    kept, but an unattached one is dropped, so a person should check the logged
    files. The warning does not fail the run. `check` fails it when the
    sitrep's post has no PDF in the manifest or its day has no sitrep.
    """
    newest_kept = max((item["id"] for item in kept), default=0)
    for item in listed:
        if (
            item["id"] > newest_kept
            and item not in kept
            and "SITREP" in item["source_url"].upper()
        ):
            logger.warning(
                "INSP media %s is newer than every kept sitrep but is neither "
                "named nor attached as this outbreak's; check whether it is a "
                "renamed sitrep: %s",
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


def find_posts_without_pdf(
    posts: list[SitrepPost],
    media: list[MediaItem],
    *,
    manifest_ids: set[int],
    now: datetime,
) -> list[SitrepPost]:
    """Return settled SitRep posts with no attached PDF in the manifest.

    Posts younger than POST_GRACE are skipped, since INSP may attach the PDF
    after posting. So are posts in POSTS_WITHOUT_PDF, and one with a PDF in the
    manifest is logged so its entry can be removed.

    Args:
        posts: INSP's SitRep posts.
        media: Every PDF in INSP's media library, with the post it is attached to.
        manifest_ids: Media ids in the manifest.
        now: Current time, timezone-aware.

    Returns:
        The failing posts, ordered by sitrep number.

    Raises:
        ValueError: If no post is older than the grace period.
    """
    settled = [post for post in posts if now - post.published >= POST_GRACE]
    if not settled:
        raise ValueError("INSP lists no SitRep post older than the grace period")
    attached: dict[int, set[int]] = {}
    for item in media:
        if (post_id := item.get("post")) is not None:
            attached.setdefault(post_id, set()).add(item["id"])
    missing = []
    for post in settled:
        has_pdf = bool(attached.get(post.post_id, set()) & manifest_ids)
        if post.post_id not in POSTS_WITHOUT_PDF:
            if not has_pdf:
                missing.append(post)
        elif has_pdf:
            logger.warning(
                "POSTS_WITHOUT_PDF lists post %d but it now has an attached PDF",
                post.post_id,
            )
    return sorted(missing, key=lambda post: (post.sitrep_number, post.post_id))


def find_gaps(report_dates: set[date]) -> list[date]:
    """Return days between the first and last report date with no sitrep.

    Days in KNOWN_GAPS are skipped. One with a sitrep is logged so its entry
    can be removed.

    Raises:
        ValueError: If there are no report dates.
    """
    if not report_dates:
        raise ValueError("The manifest has no report dates")
    for day in sorted(KNOWN_GAPS.keys() & report_dates):
        logger.warning("KNOWN_GAPS lists %s but it now has a sitrep", day)
    first, last = min(report_dates), max(report_dates)
    days = (first + timedelta(days=i) for i in range((last - first).days + 1))
    return [d for d in days if d not in report_dates and d not in KNOWN_GAPS]


def check_complete() -> int:
    """Check that the manifest holds every sitrep INSP has published.

    Prints a GitHub Actions error annotation, so the run summary shows it, for
    each SitRep post with no PDF in the manifest and each day with no sitrep.

    Returns:
        0 when nothing is missing, 1 otherwise.
    """
    manifest = pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA)
    posts = list_sitrep_posts()
    missing_posts = find_posts_without_pdf(
        posts,
        list_api_pdfs(INSP_MEDIA_API),
        manifest_ids=set(manifest["media_id"].to_list()),
        now=datetime.now(UTC),
    )
    gaps = find_gaps(set(manifest.filter("is_latest")["report_date"].to_list()))
    for post in missing_posts:
        print(
            f"::error::INSP post {post.post_id} (sitrep {post.sitrep_number}, "
            f"{post.title}) has no attached PDF in the manifest. Its PDF is not "
            "attached, was filtered out or failed to download; see the download "
            "step's warnings.",
            flush=True,
        )
    for day in gaps:
        print(
            f"::error::No sitrep in the manifest for {day}. Look for it on INSP "
            "(posted outside the SitRep category or left unattached), or add the "
            "day to KNOWN_GAPS with the reason.",
            flush=True,
        )
    logger.info(
        "%d SitRep posts, %d without a PDF in the manifest; %d days without a sitrep",
        len(posts),
        len(missing_posts),
        len(gaps),
    )
    return 1 if missing_posts or gaps else 0


def download_all() -> None:
    """Refresh the sitrep manifest and download any new PDFs."""
    previous = (
        pl.read_csv(MANIFEST_PATH, schema=MANIFEST_SCHEMA)
        if MANIFEST_PATH.exists()
        else None
    )
    manifest = build_manifest(previous, posts=list_sitrep_posts())
    manifest.write_csv(MANIFEST_PATH)
    logger.info(
        "Wrote %d files (%d report dates) to %s",
        manifest.height,
        manifest["is_latest"].sum(),
        MANIFEST_PATH.name,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return 1 when the check finds a missing sitrep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        nargs="?",
        choices=("download", "check"),
        default="download",
        help="download (default) refreshes the manifest; check fails when a "
        "SitRep post has no PDF in the manifest or a day has no sitrep",
    )
    args = parser.parse_args(argv)
    if args.command == "check":
        return check_complete()
    download_all()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
