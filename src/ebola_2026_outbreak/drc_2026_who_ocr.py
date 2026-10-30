"""Read the WHO Bundibugyo table from the images WHO published until September.

From late May to early September 2026 the WHO page showed the country table as
a PNG. This finds those images in the cached Wayback captures, reads each with
RapidOCR, rebuilds the table's cell rows from the text positions, and runs them
through the same parser as the HTML table.

RapidOCR misses the "-" cells, so values are placed by column position and
an empty cell reads as "-". A digit OCR misses would look the same, so every
count indicator, not only confirmed cases and deaths, must sum to the Total
row, and low-confidence text rejects the image.

Run `drc_2026_who_wayback.py` first. This reads its cached captures.
Requires the `ocr` dependency group: `uv sync --group ocr`.
"""

import logging
import re
import statistics
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import unquote, urljoin, urlparse

import httpx
import polars as pl

from ebola_2026_outbreak.drc_2026_who import (
    FIRST_HEADER,
    INDICATORS,
    NON_ADDITIVE,
    OUTPUT_PATH,
    ROOT,
    SNAPSHOT_DIR,
    SOURCE_URL,
    TOTAL_LABEL,
    find_table_image,
    parse_rows,
    upsert,
)
from ebola_2026_outbreak.drc_2026_who_wayback import (
    SNAPSHOT_URL,
    TIMESTAMP_FORMAT,
    WAYBACK_DIR,
)
from ebola_2026_outbreak.fetch import get

if TYPE_CHECKING:
    from rapidocr import RapidOCR

logger = logging.getLogger(__name__)

IMAGE_DIR = SNAPSHOT_DIR / "images"
# `im_` returns the archived image itself.
ARCHIVED_IMAGE_URL = SNAPSHOT_URL.replace("id_", "im_")

DATE_ROW_TEXT = "Data as of"
EMPTY_CELL = "-"
# OCR sometimes reads the "-" cells, often with low confidence.
DASH = re.compile("[-\u2013\u2014]")  # hyphen, en dash, em dash
# Tokens on one table line sit within half a text height of each other.
LINE_TOLERANCE = 0.5
# Values are left-aligned under their header; a token starting more than this
# many text heights left of the first indicator column is part of the name.
COLUMN_TOLERANCE = 1.5
# Lowest OCR confidence accepted for any token inside the table.
MIN_SCORE = 0.8
# Every count indicator must sum to the Total row (see module docstring).
OCR_STRICT_INDICATORS = frozenset(INDICATORS.values()) - NON_ADDITIVE


class OcrToken(NamedTuple):
    """One piece of text found by OCR, with its bounding box in pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float

    @property
    def y_center(self) -> float:
        """Vertical centre of the box."""
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        """Box height."""
        return self.y1 - self.y0


def group_lines(tokens: list[OcrToken]) -> list[list[OcrToken]]:
    """Group tokens into text lines by vertical position, each sorted left to right.

    Raises:
        ValueError: If there are no tokens.
    """
    if not tokens:
        raise ValueError("OCR found no text")
    tolerance = LINE_TOLERANCE * statistics.median(t.height for t in tokens)
    lines: list[list[OcrToken]] = []
    for token in sorted(tokens, key=lambda t: t.y_center):
        line = lines[-1] if lines else None
        if line and abs(token.y_center - line[-1].y_center) <= tolerance:
            line.append(token)
        else:
            lines.append([token])
    return [sorted(line, key=lambda t: t.x0) for line in lines]


def _line_text(line: list[OcrToken]) -> str:
    """Join a line's tokens with spaces."""
    return " ".join(t.text for t in line)


def _check_scores(tokens: list[OcrToken]) -> None:
    """Raise if any token that carries data was read with low confidence."""
    low = [(t.text, round(t.score, 2)) for t in tokens if t.score < MIN_SCORE]
    if low:
        raise ValueError(f"Low OCR confidence: {low}")


def tokens_to_rows(tokens: list[OcrToken]) -> list[list[str]]:
    """Rebuild the table's cell rows from OCR tokens.

    The header's first line fixes one column per indicator at its left edge;
    later lines with nothing in the name column append to the nearest header
    cell ("Suspected" + "cases"). Each body token goes to the column whose left
    edge is nearest, and tokens left of the first indicator column form the
    country name. Empty cells read "-". Reading stops after the Total row.

    Args:
        tokens: OCR output for the whole image.

    Returns:
        Cell texts per row, header first, in the shape `parse_rows` expects.

    Raises:
        ValueError: If the header is missing or text inside the table has low
            OCR confidence.
    """
    lines = group_lines(tokens)
    start = next(
        (i for i, line in enumerate(lines) if FIRST_HEADER in _line_text(line)),
        None,
    )
    if start is None:
        raise ValueError(f"No header line containing {FIRST_HEADER!r}")

    header_line = lines[start]
    columns = [t for t in header_line if FIRST_HEADER not in t.text]
    lefts = [t.x0 for t in columns]
    header_cells = [[t.text] for t in columns]
    unit = statistics.median(t.height for t in tokens)

    def column_of(token: OcrToken) -> int | None:
        """Index of the nearest column by left edge, or None if left of all."""
        if token.x0 < lefts[0] - COLUMN_TOLERANCE * unit:
            return None
        return min(range(len(lefts)), key=lambda i: abs(token.x0 - lefts[i]))

    # The May 2026 images carry one "Data as of" line above the header.
    rows: list[list[str]] = [
        [_line_text(line)]
        for line in lines[:start]
        if DATE_ROW_TEXT in _line_text(line)
    ][-1:]
    body_started = False
    for line in lines[start + 1 :]:
        text = _line_text(line)
        if DATE_ROW_TEXT in text:
            _check_scores(line)
            body_started = True
            rows.append([text])
            continue

        name_tokens = [t for t in line if column_of(t) is None]
        if not body_started and not name_tokens:
            for token in line:
                if (i := column_of(token)) is not None:
                    header_cells[i].append(token.text)
            continue
        body_started = True

        # Footnote markers ("***", sometimes misread as "*木") carry no data.
        name = [t for t in name_tokens if re.search(r"[A-Za-z]", t.text)]
        cells: list[list[OcrToken]] = [[] for _ in lefts]
        for token in line:
            i = column_of(token)
            if i is not None and not DASH.fullmatch(token.text.strip()):
                cells[i].append(token)
        _check_scores([*name, *(t for c in cells for t in c)])
        row = [_line_text(name), *(_line_text(c) if c else EMPTY_CELL for c in cells)]
        rows.append(row)
        if row[0] == TOTAL_LABEL:
            break  # Footnotes below the table are not rows.

    header = [FIRST_HEADER, *(" ".join(parts) for parts in header_cells)]
    return [header, *rows]


@cache
def _engine() -> "RapidOCR":
    """Load the OCR engine once; the import is deferred to the `ocr` group."""
    from rapidocr import RapidOCR

    return RapidOCR()


def run_ocr(image: bytes) -> list[OcrToken]:
    """Read text and boxes from an image.

    Raises:
        TypeError: If RapidOCR returns something other than full-pipeline output.
    """
    from rapidocr.utils.output import RapidOCROutput

    result = _engine()(image)
    if not isinstance(result, RapidOCROutput):
        raise TypeError(f"Unexpected RapidOCR output: {type(result).__name__}")
    if result.boxes is None or result.txts is None or result.scores is None:
        return []
    return [
        OcrToken(
            x0=float(min(p[0] for p in box)),
            y0=float(min(p[1] for p in box)),
            x1=float(max(p[0] for p in box)),
            y1=float(max(p[1] for p in box)),
            text=text,
            score=float(score),
        )
        for box, text, score in zip(
            result.boxes, result.txts, result.scores, strict=True
        )
    ]


def load_image(url: str, *, timestamp: str) -> bytes:
    """Return a table image, from cache, WHO's CDN, or the Wayback Machine."""
    path = IMAGE_DIR / unquote(Path(urlparse(url).path).name)
    if path.exists():
        return path.read_bytes()
    try:
        content = get(url).content
    except httpx.HTTPError as e:
        logger.warning("CDN fetch failed for %s (%s); trying Wayback", url, e)
        content = get(ARCHIVED_IMAGE_URL.format(timestamp=timestamp, url=url)).content
    path.write_bytes(content)
    return content


def find_images() -> dict[str, str]:
    """Map each table image URL to the first cached capture that shows it.

    Raises:
        ValueError: If there are no cached captures.
    """
    captures = sorted(WAYBACK_DIR.glob("*.html"))
    if not captures:
        raise ValueError(f"No cached captures in {WAYBACK_DIR}; run the backfill")
    images: dict[str, str] = {}
    for capture in captures:
        try:
            src = find_table_image(capture.read_text(encoding="utf-8"))
        except ValueError:
            continue  # Captured before the Bundibugyo section existed.
        if src:
            images.setdefault(urljoin(SOURCE_URL, src), capture.stem)
    return images


def parse_image(url: str, *, timestamp: str) -> pl.DataFrame:
    """OCR one table image and parse it with every count indicator strict."""
    rows = tokens_to_rows(run_ocr(load_image(url, timestamp=timestamp)))
    retrieved_at = datetime.strptime(timestamp, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    return parse_rows(
        rows,
        retrieved_at=retrieved_at,
        source_url=url,
        strict_indicators=OCR_STRICT_INDICATORS,
    )


def main() -> None:
    """OCR every table image in the cached captures and upsert the results."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    images = find_images()
    logger.info("Found %d table images in cached captures", len(images))

    frames: list[pl.DataFrame] = []
    failures: dict[str, str] = {}
    for url, timestamp in images.items():
        try:
            frames.append(parse_image(url, timestamp=timestamp))
        except ValueError as e:
            failures[url] = str(e)

    logger.info("Parsed %d images, rejected %d", len(frames), len(failures))
    for url, reason in failures.items():
        logger.warning(
            "Rejected %s: %s", unquote(Path(urlparse(url).path).name), reason
        )
    if not frames:
        raise ValueError("No table image could be parsed")
    upsert(pl.concat(frames), OUTPUT_PATH)
    logger.info("Images cached in %s", IMAGE_DIR.relative_to(ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
