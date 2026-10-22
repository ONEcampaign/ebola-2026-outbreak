"""Read sitrep table rows from the PDF text layer and check values against them.

The text layer is the third, model-free reading of each sitrep. `read_rows`
finds every province summary row, zone row, zone-table subtotal row and the
zone table's Total row, and records what each of the row's cells prints.
`row_matches` then says whether a model's values for that row are exactly
what the page prints.

Method, on the pdfplumber words of the PDF and their x extents:

- A row label is a run of words whose name keys join to a known name: a
  canonical province, a gazetteer zone or alias, or "Total". A label may wrap
  onto the next line ("Makiso-" / "Kisangani"). It is a row label only when
  no other alphabetic word shares its line, which rules out narrative text.
- A value word belongs to the row whose label is the nearest alphabetic word
  above or below it, since rows print values a few points off the label's
  line and some cells on two lines around it. A value word about as near to
  another line leaves the rows it might belong to ambiguous. Each word in
  the row's band is cut to digits, comma, percent and slash ("sur" reads as
  "/", "NA" and "ND" stay, "(" splits a word).
- A table starts at a capitalised "Létalité" header word and ends at the
  first Total row before the next such word. The table is the province
  summary when a province row prints "a/b" (ZS touchées), else the zone table.
  A zone table without an "intra-CTE" header is skipped, so its rows are not
  located.
- The header runs from as far above "Létalité" as the first row is below
  it, with words the extractor split into glyph runs joined again. One
  header word marks each column ("(n)" twice, "Nouv.", "(Swab+)",
  "intra-CTE", "Total" in the zone table; "Nouveaux", "Cas", "Décès",
  "historiquement" or else "touchées", "actuellement" in the province
  summary), and the phrase holding it on its line gives the column's x
  extent. The columns must come out in a known order: the zone table's, or
  the province summary's with "Nouveaux cas" first (SitRep 105 onwards) or
  last (SitRep 96).
- Band words a few points apart form one cell ("1 984", "28/36 (77,8 %)"),
  and a "%" ends a cell on its line. A cell on two lines reads line by line.
  Each column reaches halfway to its neighbours, and a cell belongs to the
  column whose reach holds its centre with a margin. A column with no cell
  is blank. The placed cells, read left to right, must give back the band.
- In the zone table a province label opens that province's subtotal row and
  the zone rows that follow belong to it. A zone named like its province
  (Tshopo) follows its own subtotal row.

A key found twice is not located, so an ambiguous page is flagged and the row
is not guessed. A located row has no cells when a value word near it is
ambiguous, its table's header words are missing, repeated or out of order,
a cell sits near the boundary of two columns, two cells fall in one column,
or a cell does not read as one value of its column. `row_matches` then
fails, so the row is held.

`row_matches` compares field by field. A blank cell prints nothing and a
printed 0 prints "0", so blank-for-0, 0-for-blank and a value moved into an
adjacent blank cell all fail. A space may split a number only at a thousands
boundary or before "%". A CFR matches a printing with 0 to 2 decimals within
0.05 of the value.

`dump_words` writes the words of a PDF's table pages as a JSON fixture, which
`read_rows` reads as it reads the PDF.
"""

import itertools
import json
import logging
import math
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Set
from pathlib import Path
from typing import NamedTuple

import pdfplumber

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    FIELDS,
    PERCENT_FIELDS,
    ZONE_TABLE_FIELDS,
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
AMBIGUITY = 2  # points: a value word this much nearer one line than another is clear
WRAP = 12  # points: a wrapped label's next word starts within this distance
SPLIT_GAP = 0.5  # points: header words within this gap or overlap are one word
CELL_GAP = 6  # points: words of one cell are at most this far apart
GUARD = 3  # points: a cell's centre is at least this far inside its column
CFR_TOLERANCE = 0.05 + 1e-9

PROVINCE_KEYS = {name_key(p): p for p in PROVINCES.values()}
TOTAL_KEYS = frozenset({"total", "totalgeneral", "totalnational", "totalrdc"})
NULL_TOKENS = frozenset({"NA", "ND"})

# What one cell may print, by the kind of its column.
CELL_GRAMMAR = {
    "count": re.compile(r"\d+|\d{1,3}(?: \d{3})+|NA|ND"),
    "percent": re.compile(r"\d+(?:,\d{1,2})? ?%|NA|ND"),
    # "28/36 77,8 %", from "28/36 (77,8 %)"; the share is dropped.
    "zs": re.compile(r"(\d+) ?/ ?(\d+)(?: \d+,\d+ ?%)?|NA|ND"),
}
COLUMN_KIND = {"cfr_pct": "percent", "zs": "zs", "zs_current": "zs"}


class Word(NamedTuple):
    """A pdfplumber word: 1-based page, top, left and right edges in points,
    text."""

    page: int
    top: float
    x0: float
    x1: float
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
    """A row label and the value words of its band, left to right."""

    key: str  # the label's name key
    page: int
    top: float
    words: tuple[Word, ...]
    ambiguous: bool  # a value word near the row may belong to another line

    @property
    def pos(self) -> tuple[int, float]:
        return (self.page, self.top)

    @property
    def band(self) -> str:
        return " ".join(w.value for w in self.words)


class Table(NamedTuple):
    """A table's header anchor and the rows from its header to its Total."""

    anchor: Word
    rows: list[Row]


# A table's columns, left to right: (column, x0, x1 of its header phrase).
type Columns = list[tuple[str, float, float]]

PROVINCE_LAYOUTS = (
    ("new_confirmed", "cum_confirmed", "cum_deaths", "cfr_pct", "zs"),
    ("new_confirmed", "cum_confirmed", "cum_deaths", "cfr_pct", "zs", "zs_current"),
    ("cum_confirmed", "cum_deaths", "cfr_pct", "zs", "new_confirmed"),  # SitRep 96
)


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
                Word(
                    number,
                    round(w["top"], 2),
                    round(w["x0"], 2),
                    round(w["x1"], 2),
                    w["text"],
                )
                for w in page.extract_words()
            ]
            words.extend(sorted(_drop_page_number(page_words)))
    return words


def _dump_words(path: Path) -> list[Word]:
    """Words of a `dump_words` fixture."""
    words: list[Word] = []
    for page, lines in json.loads(path.read_text())["pages"].items():
        for line in lines:
            top, x0, x1, text = line.split(" ", 3)
            words.append(Word(int(page), float(top), float(x0), float(x1), text))
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
    ("Au total, 26 décès"). Each value word belongs to the row whose label
    is the nearest alphabetic word above or below it, since a row may print
    a value a few points off its label's line, or a cell on two lines
    straddling it. A value word whose distance to another line is within
    `AMBIGUITY` of its distance to the nearest makes every row it might
    belong to ambiguous. A label with no value and no ambiguity is not a
    row.
    """
    rows: list[Row] = []
    for page in sorted({w.page for w in words}):
        page_words = [w for w in words if w.page == page]
        alpha = [w for w in page_words if w.is_alpha]
        labels = [
            label
            for label in _labels(alpha, keys)
            if not any(
                o not in label and abs(o.top - w.top) <= TOL
                for w in label
                for o in alpha
            )
        ]
        owner = {w: i for i, label in enumerate(labels) for w in label}
        bands: list[list[Word]] = [[] for _ in labels]
        ambiguous = [False] * len(labels)
        for o in page_words:
            if o.is_alpha or not o.value or not alpha:
                continue
            nearest = min(alpha, key=lambda a: abs(a.top - o.top))
            row = owner.get(nearest)
            if row is not None:
                bands[row].append(o)
            reach = abs(nearest.top - o.top) + AMBIGUITY
            for a in alpha:
                rival = owner.get(a)
                if rival != row and abs(a.top - o.top) < reach:
                    for i in (row, rival):
                        if i is not None:
                            ambiguous[i] = True
        for label, band, unclear in zip(labels, bands, ambiguous, strict=True):
            if band or unclear:
                key = "".join(w.key for w in label)
                band.sort(key=lambda o: o.x0)
                rows.append(Row(key, page, label[0].top, tuple(band), unclear))
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


def _header(words: list[Word], table: Table) -> list[Word]:
    """Alphabetic words of a table's header, left to right.

    The header is centred on the "Létalité" cell, so it runs from as far
    above the anchor as the first row is below it. Words the extractor split
    with no gap ("No" "uv.") are joined.
    """
    anchor, first = table.anchor, table.rows[0]
    if first.page != anchor.page:
        return []
    lo = 2 * anchor.top - first.top
    header = [
        w
        for w in words
        if w.page == anchor.page and w.is_alpha and lo <= w.top < first.top
    ]
    joined: list[Word] = []
    for w in sorted(header, key=lambda w: w.x0):
        i = next(
            (
                i
                for i, j in enumerate(joined)
                if abs(j.top - w.top) <= TOL and abs(w.x0 - j.x1) < SPLIT_GAP
            ),
            None,
        )
        if i is None:
            joined.append(w)
        else:
            joined[i] = joined[i]._replace(x1=w.x1, text=joined[i].text + w.text)
    return joined


def _one(words: Iterable[Word]) -> Word | None:
    found = list(words)
    return found[0] if len(found) == 1 else None


def _phrase(word: Word, header: list[Word]) -> tuple[float, float]:
    """x extent of the phrase holding `word` on its header line: the words
    joined to it by gaps of at most `CELL_GAP` ("Total décès confirmés")."""
    line = [w for w in header if abs(w.top - word.top) <= TOL]
    lo = hi = line.index(word)
    while lo > 0 and line[lo].x0 - line[lo - 1].x1 <= CELL_GAP:
        lo -= 1
    while hi + 1 < len(line) and line[hi + 1].x0 - line[hi].x1 <= CELL_GAP:
        hi += 1
    return line[lo].x0, line[hi].x1


def _columns(
    markers: Mapping[str, Word | None],
    header: list[Word],
    layouts: Collection[tuple[str, ...]],
) -> Columns | None:
    """Columns from each column's header word, or None when a header word is
    missing, two header phrases overlap, or the columns, left to right, are
    not one of `layouts`."""
    columns = []
    for column, word in markers.items():
        if word is None:
            return None
        columns.append((column, *_phrase(word, header)))
    columns.sort(key=lambda c: c[1])
    if tuple(c for c, _, _ in columns) not in layouts:
        return None
    if any(a[2] >= b[1] for a, b in itertools.pairwise(columns)):
        return None
    return columns


def _province_columns(header: list[Word]) -> Columns | None:
    """Columns of the province summary, "zs" being the "a/b" cell of affected
    zones and "zs_current" the optional "ZS actuellement" cell.

    "Cas" is the last "cas" left of "Décès", since "Nouveaux cas" prints one.
    The ZS column is marked by "historiquement" where the header prints "ZS
    historiquement touchées", since a left-aligned ZS cell starts under it,
    and by "touchées" otherwise.
    A table title ("Répartition des cas et décès confirmés") reaching into
    the header brings a second "décès", which leaves the columns unresolved.
    """
    deaths = _one(w for w in header if w.key == "deces")
    cases = [w for w in header if w.key == "cas" and deaths and w.x1 < deaths.x0]
    markers = {
        "new_confirmed": _one(w for w in header if w.key.startswith("nouv")),
        "cum_confirmed": cases[-1] if cases else None,
        "cum_deaths": deaths,
        "cfr_pct": _one(w for w in header if w.key == "letalite"),
        "zs": _one(w for w in header if w.key == "historiquement")
        or _one(w for w in header if w.key == "touchees"),
    }
    if any(w.key == "actuellement" for w in header):
        markers["zs_current"] = _one(w for w in header if w.key == "actuellement")
    return _columns(markers, header, PROVINCE_LAYOUTS)


def _zone_columns(header: list[Word]) -> Columns | None:
    """Columns of the zone table, by "Cas (n)" and "Décès (n)", "Létalité",
    then the day's "Nouv. Cas", "(Swab+)", "intra-CTE" and "Total"."""
    counts = [w for w in header if w.key == "n"]
    cfr = _one(w for w in header if w.key == "letalite")
    markers = {
        "cum_confirmed": counts[0] if len(counts) == 2 else None,
        "cum_deaths": counts[1] if len(counts) == 2 else None,
        "cfr_pct": cfr,
        "new_confirmed": _one(w for w in header if w.key.startswith("nouv")),
        "community_deaths": _one(w for w in header if "swab" in w.key),
        "cte_deaths": _one(w for w in header if w.key == "intracte"),
        "total_deaths_day": _one(
            w for w in header if w.key == "total" and cfr and w.x0 > cfr.x1
        ),
    }
    return _columns(markers, header, [ZONE_TABLE_FIELDS])


def _reading_order(cell: list[Word]) -> list[Word]:
    """A cell's words line by line, then left to right: a cell printed on two
    lines ("25/36 (69,4" over "%)") reads as one value."""
    lines: list[list[Word]] = []
    for word in sorted(cell, key=lambda w: w.top):
        if lines and word.top - lines[-1][0].top <= TOL:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [w for line in lines for w in sorted(line, key=lambda w: w.x0)]


def _column_text(row: Row, columns: Columns) -> dict[str, str | None] | None:
    """Each column's printed text, None for a blank, or None for the row when
    a cell is not clearly under one column or two cells share one.

    Words at most `CELL_GAP` points apart form one cell ("1 984", "28/36
    (77,8 %)"), and a word ending in "%" closes its cell on its line, since
    a ZS cell can start closer than that after a CFR. Each column reaches
    halfway to its neighbours' header phrases, and an outer column as far
    outward as inward. A cell belongs to the column whose reach holds its centre, at
    least `GUARD` points inside, which holds for left-aligned, centred and
    right-aligned cells. The placed cells, read left to right, must give back
    the row's band.
    """
    cells: list[list[Word]] = []
    for word in row.words:
        if (
            cells
            and word.x0 - max(w.x1 for w in cells[-1]) <= CELL_GAP
            and not (
                cells[-1][-1].value.endswith("%")
                and abs(cells[-1][-1].top - word.top) <= TOL
            )
        ):
            cells[-1].append(word)
        else:
            cells.append([word])

    inner = [(a[2] + b[1]) / 2 for a, b in itertools.pairwise(columns)]
    bounds = [2 * columns[0][1] - inner[0], *inner, 2 * columns[-1][2] - inner[-1]]
    placed: dict[str, list[Word]] = {}
    for cell in cells:
        centre = (cell[0].x0 + max(w.x1 for w in cell)) / 2
        under = [
            column
            for (column, _, _), (lo, hi) in zip(
                columns, itertools.pairwise(bounds), strict=True
            )
            if lo + GUARD <= centre <= hi - GUARD
        ]
        if len(under) != 1 or under[0] in placed:
            return None
        placed[under[0]] = cell

    in_order = [w for column, _, _ in columns for w in placed.get(column, [])]
    if " ".join(w.value for w in in_order) != row.band:
        return None
    return {
        column: " ".join(w.value for w in _reading_order(placed[column]))
        if column in placed
        else None
        for column, _, _ in columns
    }


def _cells(row: Row, columns: Columns) -> dict[str, str | None] | None:
    """The row's printed text by field, or None when a cell is unclear or
    does not read as one value of its column."""
    text = None if row.ambiguous else _column_text(row, columns)
    if text is None:
        return None
    cells: dict[str, str | None] = {}
    for column, printed in text.items():
        m = None
        if printed is not None:
            m = CELL_GRAMMAR[COLUMN_KIND.get(column, "count")].fullmatch(printed)
            if m is None:
                return None
        if column == "zs":
            pair = (m[1], m[2]) if m and m[1] else (printed, printed)
            cells["zs_affected"], cells["zs_total"] = pair
        elif column != "zs_current":
            cells[column] = printed
    return cells


# Rows by key


def _province_rows(table: Table) -> Iterator[tuple[RowKey, Row]]:
    for row in table.rows:
        if row.key in PROVINCE_KEYS:
            yield RowKey("provinces", PROVINCE_KEYS[row.key], ""), row


def _zone_rows(
    header: list[Word], table: Table, gazetteer: Gazetteer
) -> Iterator[tuple[RowKey, Row]]:
    if not any("intra" in w.key for w in header):
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
        `located=False`. A located row whose values cannot be placed in
        columns has `cells=None`.
    """
    words = _load_words(source)
    keys = set(gazetteer) | set(PROVINCE_KEYS) | TOTAL_KEYS
    rows = _find_rows(words, keys)

    found: dict[RowKey, list[tuple[Row, dict[str, str | None] | None]]] = {}
    for table in _tables(words, rows):
        header = _header(words, table)
        if any(
            r.key in PROVINCE_KEYS and re.search(r"\d ?/ ?\d", r.band)
            for r in table.rows
        ):
            located = _province_rows(table)
            columns = _province_columns(header)
        else:
            located = _zone_rows(header, table, gazetteer)
            columns = _zone_columns(header)
        if columns is None:
            logger.warning("Table on page %d: columns unresolved", table.anchor.page)
        for key, row in located:
            cells = None if columns is None else _cells(row, columns)
            if columns is not None and cells is None:
                logger.warning("Row %s: values not placed in columns", key)
            found.setdefault(key, []).append((row, cells))

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
            row, cells = hits[0]
            out[key] = TextRow(key, row.band, True, row.page, cells)
        else:
            out[key] = TextRow(key, None, False, None, None)
    return out


# Matching


def _cell_matches(cell: str | None, value: Number | None, field: str) -> bool:
    if value is None:
        return cell is None or cell in NULL_TOKENS
    if cell is None or cell in NULL_TOKENS or isinstance(value, bool) or value < 0:
        return False
    if field in PERCENT_FIELDS:
        printed = float(cell.rstrip(" %").replace(",", "."))
        return abs(printed - value) <= CFR_TOLERANCE
    return value == int(value) and int(cell.replace(" ", "")) == value


def row_matches(
    text_row: TextRow, values: Mapping[str, Number | None], section: Section
) -> bool:
    """Whether a row's values are exactly what the text layer prints.

    Args:
        text_row: From `read_rows`.
        values: The row's fields, as in `Extraction.rows`; a missing field
            counts as null.
        section: The row's section, which fixes its fields.

    Returns:
        False when the row's cells are unknown, or any field differs from
        its cell: a null needs a blank or NA cell, a count needs its exact
        whole number, and a percentage needs a printing within 0.05 of it.
    """
    cells = text_row.cells
    if not text_row.located or cells is None:
        return False
    return all(_cell_matches(cells[f], values.get(f), f) for f in FIELDS[section])


# Fixtures


def dump_words(pdf_path: Path, out_path: Path) -> Path:
    """Write the words of a sitrep's table pages as a JSON fixture.

    The fixture holds the PDF's name and sha256, and for each page from a
    table's header to its Total row one "top x0 x1 text" string per word, in
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
            str(page): [
                f"{w.top} {w.x0} {w.x1} {w.text}" for w in words if w.page == page
            ]
            for page in sorted(pages)
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2) + "\n")
    return out_path
