"""Read sitrep table rows from the PDF text layer and check values against them.

The text layer is the third, model-free reading of each sitrep. `read_rows`
finds every province summary row, zone row, zone-table subtotal row and the
zone table's Total row, and records the row's printed tokens. `row_matches`
then says whether a model's values for that row are exactly what the page
prints.

Method, on the pdfplumber words of the PDF:

- A row label is a run of words whose name keys join to a known name: a
  canonical province, a gazetteer zone or alias, or "Total". A label may wrap
  onto the next line ("Makiso-" / "Kisangani"). It is a row label only when
  no other alphabetic word shares its line, which rules out narrative text.
- The row's band runs from the label's top down to just above the next
  alphabetic word, since some rows print their values a point or two above or
  below the label. Each word in the band is cut to digits, comma, percent and
  slash ("sur" reads as "/", "NA" and "ND" stay, "(" splits a word) and the
  words are joined left to right by single spaces: "6 960 3 217 46,2 %".
- A table starts at a capitalised "Létalité" header word and ends at the
  first Total row before the next such word. The table is the province
  summary when a province row prints "a/b" (ZS touchées), else the zone table.
- The column order comes from the header words. The province summary prints
  "Nouveaux cas" last in SitRep 96 and first from SitRep 105; a new-last row
  is stored new-first, so every province row reads new, cases, deaths, CFR,
  ZS a/b, then the optional "ZS actuellement" a/b. The zone table prints the
  cumulative columns, then the day's (header "intra-CTE"). A table whose
  header matches no known layout is skipped, so its rows are not located.
- In the zone table a province label opens that province's subtotal row and
  the zone rows that follow belong to it. A zone named like its province
  (Tshopo) follows its own subtotal row.

A key found twice is not located, so an ambiguous page is flagged and the row
is not guessed. `row_matches` compares exactly. A blank prints nothing and a
printed 0 prints "0", so blank-for-0 and 0-for-blank both fail. A space may
split a number only at a thousands boundary or before "%". A CFR matches a
printing with 0 to 2 decimals within 0.05 of the value. The check cannot see
a value moved into an adjacent blank cell of the same row ("1, blank" and
"blank, 1" print alike). The sum checks catch that.

`dump_words` writes the words of a PDF's table pages as a JSON fixture, which
`read_rows` reads as it reads the PDF.
"""

import json
import logging
import math
import re
from collections.abc import Iterator, Mapping, Set
from pathlib import Path
from typing import NamedTuple

import pdfplumber

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    FIELDS,
    PERCENT_FIELDS,
    Gazetteer,
    Number,
    RowKey,
    Section,
    TextRow,
    name_key,
    sha256_file,
)
from ebola_2026_outbreak.drc_2026_sitreps_parse import PROVINCES

logger = logging.getLogger(__name__)

TOL = 3  # points: words this close vertically share a line
WRAP = 12  # points: a wrapped label's next word starts within this distance
CFR_TOLERANCE = 0.05 + 1e-9

PROVINCE_KEYS = {name_key(p): p for p in PROVINCES.values()}
TOTAL_KEYS = frozenset({"total", "totalgeneral", "totalnational", "totalrdc"})

NULL = r"(?:N[AD] )?"
ZS = r"\d+ ?/ ?\d+(?: \d+,\d+ ?%)?"  # "28/36 77,8 %", from "28/36 (77,8 %)"


class Word(NamedTuple):
    """A pdfplumber word: 1-based page, top and left edge in points, text."""

    page: int
    top: float
    x0: float
    text: str

    @property
    def key(self) -> str:
        return name_key(self.text)

    @property
    def is_token(self) -> bool:
        """NA/ND cells and the "sur" of "20 sur 36" are values, not labels."""
        return bool(re.fullmatch(r"N[AD]\W*", self.text)) or self.text == "sur"

    @property
    def is_alpha(self) -> bool:
        return bool(re.search(r"[a-z]", self.key)) and not self.is_token

    @property
    def value(self) -> str:
        """The word as a printed value: "(77,8" -> "77,8", "NA!" -> "NA"."""
        if self.text == "sur":
            return "/"
        if self.is_token:
            return self.text[:2]
        return re.sub(r"[^\d,%/(]", "", self.text).replace("(", " ").strip()


class Row(NamedTuple):
    """A row label and its band of printed values."""

    key: str  # the label's name key
    page: int
    top: float
    band: str

    @property
    def pos(self) -> tuple[int, float]:
        return (self.page, self.top)


class Table(NamedTuple):
    """A table's header anchor and the rows from its header to its Total."""

    anchor: Word
    rows: list[Row]


# Words


def _drop_page_number(words: list[Word]) -> list[Word]:
    """Drop a lone number on a page's bottom line, the page number."""
    if not words:
        return words
    bottom = max(w.top for w in words)
    last = [w for w in words if bottom - w.top <= TOL]
    if len(last) == 1 and last[0].text.isdigit():
        words.remove(last[0])
    return words


def _pdf_words(pdf_path: Path) -> list[Word]:
    """Every word of the PDF, page by page, in reading order."""
    words: list[Word] = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            page_words = [
                Word(number, round(w["top"], 2), round(w["x0"], 2), w["text"])
                for w in page.extract_words()
            ]
            words.extend(sorted(_drop_page_number(page_words)))
    return words


def _dump_words(path: Path) -> list[Word]:
    """Words of a `dump_words` fixture."""
    words: list[Word] = []
    for page, lines in json.loads(path.read_text())["pages"].items():
        for line in lines:
            top, x0, text = line.split(" ", 2)
            words.append(Word(int(page), float(top), float(x0), text))
    return sorted(words)


def _load_words(source: Path) -> list[Word]:
    return _dump_words(source) if source.suffix == ".json" else _pdf_words(source)


# Rows and tables


def _labels(alpha: list[Word], keys: Set[str]) -> Iterator[list[Word]]:
    """Runs of alphabetic words whose keys join to the longest name in `keys`.

    `alpha` holds one page's alphabetic words in reading order; a run may
    continue onto the next line, within `WRAP` points.
    """
    prefixes = {k[:n] for k in keys for n in range(1, len(k) + 1)}
    i = 0
    while i < len(alpha):
        acc, end = "", None
        for j in range(i, len(alpha)):
            if j > i and alpha[j].top - alpha[j - 1].top > WRAP:
                break
            acc += alpha[j].key
            if acc not in prefixes:
                break
            if acc in keys:
                end = j + 1
        if end is None:
            i += 1
            continue
        yield alpha[i:end]
        i = end


def _find_rows(words: list[Word], keys: Set[str]) -> list[Row]:
    """Every table row labelled with a name in `keys`, in reading order.

    A label sharing its line with another alphabetic word is running text
    ("Au total, 26 décès"), and a label with no value in its band is not a
    row.
    """
    rows: list[Row] = []
    for page in sorted({w.page for w in words}):
        page_words = [w for w in words if w.page == page]
        alpha = [w for w in page_words if w.is_alpha]
        for label in _labels(alpha, keys):
            own, first = set(label), label[0]
            if any(
                o not in own and abs(o.top - w.top) <= TOL for w in label for o in alpha
            ):
                continue
            below = [o.top for o in alpha if o not in own and o.top > first.top + TOL]
            end = min(below, default=math.inf) - TOL
            band_words = sorted(
                (
                    o
                    for o in page_words
                    if not o.is_alpha and first.top - TOL <= o.top < end
                ),
                key=lambda o: o.x0,
            )
            band = " ".join(v for o in band_words if (v := o.value))
            if band:
                key = "".join(w.key for w in label)
                rows.append(Row(key, first.page, first.top, band))
    return rows


def _tables(words: list[Word], rows: list[Row]) -> list[Table]:
    """Each table: a capitalised "Létalité" header word, then the rows up to
    the first Total row before the next such word.

    Running text writes "la létalité", and the page 1 tile "LETALITE" has no
    Total row before the province table's header, so neither opens a table.
    """
    anchors = [
        w for w in words if w.key == "letalite" and w.text.lstrip("(")[:1].isupper()
    ]
    tables: list[Table] = []
    for i, anchor in enumerate(anchors):
        start = (anchor.page, anchor.top)
        limit = (
            (anchors[i + 1].page, anchors[i + 1].top)
            if i + 1 < len(anchors)
            else (math.inf, math.inf)
        )
        total = next(
            (r for r in rows if r.key in TOTAL_KEYS and start < r.pos < limit), None
        )
        if total is not None:
            tables.append(
                Table(anchor, [r for r in rows if start < r.pos <= total.pos])
            )
    return tables


def _header(words: list[Word], anchor: Word) -> list[Word]:
    """Header words: the anchor's page, from 50 pt above it to 25 pt below."""
    return [
        w
        for w in words
        if w.page == anchor.page and anchor.top - 50 <= w.top <= anchor.top + 25
    ]


def _new_first(band: str) -> str:
    """Move a trailing "Nouveaux cas" cell to the front of a province band.

    "6 960 3 217 46,2 % 28/36 77,8 % 42" becomes "42 6 960 3 217 46,2 % 28/36
    77,8 %". A band without a ZS cell is left as printed and will not match.
    """
    m = re.fullmatch(rf"(.*?{ZS})((?: (?:\d+|N[AD]))*)", band)
    return f"{m[2].strip()} {m[1]}".strip() if m else band


def _province_rows(words: list[Word], table: Table) -> Iterator[tuple[RowKey, Row]]:
    header = _header(words, table.anchor)
    new = [w.x0 for w in header if w.key.startswith("nouv")]
    if not new:
        logger.warning(
            "Province table on page %d: no new-cases header", table.anchor.page
        )
        return
    new_last = min(new) > table.anchor.x0
    for row in table.rows:
        if row.key in PROVINCE_KEYS:
            band = _new_first(row.band) if new_last else row.band
            yield (
                RowKey("provinces", PROVINCE_KEYS[row.key], ""),
                row._replace(band=band),
            )


def _zone_rows(
    words: list[Word], table: Table, gazetteer: Gazetteer
) -> Iterator[tuple[RowKey, Row]]:
    if not any("intra" in w.key for w in _header(words, table.anchor)):
        logger.warning("Zone table on page %d: unknown layout", table.anchor.page)
        return
    province = None
    for row in table.rows:
        zones = gazetteer.get(row.key, ())
        if row.key in TOTAL_KEYS:
            yield RowKey("zone_total", "", ""), row
        elif row.key in PROVINCE_KEYS and not (
            province == PROVINCE_KEYS[row.key]
            and any(z.province == province for z in zones)
        ):
            province = PROVINCE_KEYS[row.key]
            yield RowKey("zone_subtotals", province, ""), row
        else:
            match = next((z for z in zones if z.province == province), None)
            if match is None and len(zones) == 1:
                match = zones[0]
            if match is None:
                logger.warning(
                    "Zone row %r on page %d has no province", row.key, row.page
                )
                continue
            yield RowKey("zones", *match), row


def read_rows(source: Path, gazetteer: Gazetteer) -> dict[RowKey, TextRow]:
    """Read every province, zone, zone subtotal and zone Total row of a sitrep.

    Args:
        source: The sitrep PDF, or a `dump_words` JSON fixture of it.
        gazetteer: From `drc_2026_sitrep_schema.load_gazetteer`; its zone
            names and aliases, the canonical provinces and "Total" are the
            only row labels read.

    Returns:
        One `TextRow` per canonical province (in `provinces` and
        `zone_subtotals`), per gazetteer zone, and for `zone_total`. A row
        the text layer does not print, or prints twice, has
        `located=False`.
    """
    words = _load_words(source)
    keys = set(gazetteer) | set(PROVINCE_KEYS) | TOTAL_KEYS
    rows = _find_rows(words, keys)

    found: dict[RowKey, list[Row]] = {}
    for table in _tables(words, rows):
        province_table = any(
            r.key in PROVINCE_KEYS and re.search(r"\d ?/ ?\d", r.band)
            for r in table.rows
        )
        located = (
            _province_rows(words, table)
            if province_table
            else _zone_rows(words, table, gazetteer)
        )
        for key, row in located:
            found.setdefault(key, []).append(row)

    expected = [
        RowKey(section, province, "")
        for section in ("provinces", "zone_subtotals")
        for province in PROVINCE_KEYS.values()
    ]
    expected.append(RowKey("zone_total", "", ""))
    expected.extend(RowKey("zones", *z) for zones in gazetteer.values() for z in zones)

    out: dict[RowKey, TextRow] = {}
    for key in dict.fromkeys(expected):
        hits = found.get(key, [])
        if len(hits) > 1:
            logger.warning("Row %s printed %d times; not located", key, len(hits))
        if len(hits) == 1:
            out[key] = TextRow(key, hits[0].band, True, hits[0].page)
        else:
            out[key] = TextRow(key, None, False, None)
    return out


# Matching


def _count(value: Number) -> str | None:
    """Pattern for a count as printed, with its trailing space."""
    if isinstance(value, bool) or value < 0 or value != int(value):
        return None
    return f"{int(value):,}".replace(",", " ?") + " "


def _percent(value: Number) -> str | None:
    """Pattern for every printing of a percentage within the tolerance.

    Tables print one decimal; some rows print none or two ("100,00%").
    """
    if isinstance(value, bool) or value < 0:
        return None
    printed = []
    for decimals in (0, 1, 2):
        scale = 10**decimals
        lo = math.ceil((value - CFR_TOLERANCE) * scale)
        hi = math.floor((value + CFR_TOLERANCE) * scale)
        for k in range(lo, hi + 1):
            whole = f"{k // scale:,}".replace(",", " ?")
            printed.append(whole + (f",{k % scale:0{decimals}d}" if decimals else ""))
    if not printed:
        return None
    return "(?:" + "|".join(printed) + ") ?% "


def _cell(values: Mapping[str, Number | None], field: str) -> str | None:
    value = values.get(field)
    if value is None:
        return NULL
    return _percent(value) if field in PERCENT_FIELDS else _count(value)


def _pattern(values: Mapping[str, Number | None], section: Section) -> str | None:
    """Pattern for the printed row, or None when a value cannot be printed."""
    if section == "provinces":
        affected, total = values.get("zs_affected"), values.get("zs_total")
        if affected is None and total is None:
            zs = NULL
        elif affected is None or total is None:
            return None
        else:
            a, b = _count(affected), _count(total)
            if a is None or b is None:
                return None
            zs = rf"{a.rstrip()} ?/ ?{b.rstrip()}(?: \d+,\d+ ?%)? "
        cells = [
            _cell(values, f) for f in ("new_confirmed", "cum_confirmed", "cum_deaths")
        ]
        cells += [_cell(values, "cfr_pct"), zs, rf"(?:{ZS} )?"]
    else:
        cells = [_cell(values, f) for f in FIELDS[section]]
    if any(c is None for c in cells):
        return None
    return "".join(c for c in cells if c is not None)


def row_matches(
    text_row: TextRow, values: Mapping[str, Number | None], section: Section
) -> bool:
    """Whether a row's values are exactly what the text layer prints.

    Args:
        text_row: From `read_rows`.
        values: The row's fields, as in `Extraction.rows`; a missing field
            counts as null.
        section: The row's section, which fixes the column order.

    Returns:
        False when the row was not located, a value cannot be printed (a
        negative or fractional count, half a ZS pair), or the printed tokens
        differ: a null prints nothing or NA, a count prints its digits with an
        optional space at each thousands boundary, and a percentage prints
        within 0.05 of the value with 0 to 2 decimals.
    """
    if not text_row.located or text_row.printed is None:
        return False
    pattern = _pattern(values, section)
    return (
        pattern is not None
        and re.fullmatch(pattern, text_row.printed + " ") is not None
    )


# Fixtures


def dump_words(pdf_path: Path, out_path: Path) -> Path:
    """Write the words of a sitrep's table pages as a JSON fixture.

    The fixture holds the PDF's name and sha256, and for each page from a
    table's header to its Total row one "top x0 text" string per word, in
    reading order. Whole pages are kept, since a row's band and its
    running-text test read the words around it.

    Args:
        pdf_path: The sitrep PDF.
        out_path: Where to write the fixture.

    Returns:
        `out_path`.
    """
    words = _pdf_words(pdf_path)
    pages: set[int] = set()
    for table in _tables(words, _find_rows(words, TOTAL_KEYS)):
        pages.update(range(table.anchor.page, table.rows[-1].page + 1))
    dump = {
        "source": pdf_path.name,
        "sha256": sha256_file(pdf_path),
        "pages": {
            str(page): [f"{w.top} {w.x0} {w.text}" for w in words if w.page == page]
            for page in sorted(pages)
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2) + "\n")
    return out_path
