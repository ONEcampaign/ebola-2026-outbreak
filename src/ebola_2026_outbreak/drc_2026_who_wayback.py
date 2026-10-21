"""Backfill the WHO Bundibugyo table history from Wayback Machine snapshots.

The live WHO page shows only the latest figures. This takes one capture per
day from the Wayback Machine, parses each with the live-page parser and
upserts into the same tracked CSV. Captures from before the Bundibugyo section
existed, or with a layout the parser rejects, are skipped and listed.
"""

import logging
import time
from datetime import UTC, datetime

import polars as pl

from ebola_2026_outbreak.drc_2026_who import (
    OUTPUT_PATH,
    ROOT,
    SNAPSHOT_DIR,
    SOURCE_URL,
    parse_who_table,
    upsert,
)
from ebola_2026_outbreak.fetch import get

logger = logging.getLogger(__name__)

CDX_URL = "https://web.archive.org/cdx/search/cdx"
# `id_` returns the archived page without the Wayback toolbar.
SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{url}"
WAYBACK_DIR = SNAPSHOT_DIR / "wayback"
# The outbreak was declared in mid-May 2026; earlier captures predate the table.
BACKFILL_FROM = "20260501"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
# archive.org refuses connections above ~15 requests/minute.
REQUEST_PAUSE_S = 5.0


def list_snapshots(url: str, *, since: str = BACKFILL_FROM) -> list[str]:
    """List one successful Wayback capture timestamp per day for `url`.

    Args:
        url: Archived page URL.
        since: Earliest capture to include, as YYYYMMDD.

    Returns:
        Capture timestamps (YYYYMMDDhhmmss), oldest first.
    """
    params = (
        f"?url={url}&from={since}&output=json&fl=timestamp"
        "&filter=statuscode:200&collapse=timestamp:8"
    )
    rows = get(CDX_URL + params).json()
    if not rows or rows[0] != ["timestamp"]:
        raise ValueError(f"Unexpected CDX response header: {rows[:1]}")
    return [row[0] for row in rows[1:]]


def load_snapshot(timestamp: str) -> str:
    """Return a capture's HTML, downloading it only if not already cached."""
    path = WAYBACK_DIR / f"{timestamp}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")

    html = get(SNAPSHOT_URL.format(timestamp=timestamp, url=SOURCE_URL)).text
    path.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_PAUSE_S)
    return html


def parse_snapshots(timestamps: list[str]) -> tuple[pl.DataFrame, dict[str, str]]:
    """Parse each capture, collecting rows and the reason for each skip.

    Returns:
        Parsed rows from every usable capture, and {timestamp: reason} for
        captures the parser rejected.
    """
    frames: list[pl.DataFrame] = []
    skipped: dict[str, str] = {}
    for timestamp in timestamps:
        retrieved_at = datetime.strptime(timestamp, TIMESTAMP_FORMAT).replace(
            tzinfo=UTC
        )
        source_url = SNAPSHOT_URL.format(timestamp=timestamp, url=SOURCE_URL)
        try:
            frames.append(
                parse_who_table(
                    load_snapshot(timestamp),
                    retrieved_at=retrieved_at,
                    source_url=source_url,
                )
            )
        except ValueError as e:
            skipped[timestamp] = str(e)

    if not frames:
        raise ValueError(f"No Wayback capture could be parsed: {skipped}")
    return pl.concat(frames), skipped


def main() -> None:
    """Backfill the WHO table from every daily Wayback capture."""
    WAYBACK_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = list_snapshots(SOURCE_URL)
    logger.info("Found %d daily captures since %s", len(timestamps), BACKFILL_FROM)

    df, skipped = parse_snapshots(timestamps)
    logger.info(
        "Parsed %d captures, skipped %d", len(timestamps) - len(skipped), len(skipped)
    )
    for timestamp, reason in skipped.items():
        logger.warning("Skipped capture %s: %s", timestamp, reason)
    logger.info("Captures cached in %s", WAYBACK_DIR.relative_to(ROOT))

    upsert(df, OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
