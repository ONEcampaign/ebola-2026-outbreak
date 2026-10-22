"""Tests for the sitrep text-layer row reader.

`fixtures/sitrep_words/` holds `dump_words` fixtures of the 13 golden
reports (SitReps 96 to 157 in the current layout) and `truth.csv`, their
hand-checked province, zone, zone subtotal and zone Total rows under
canonical keys. A blank cell in `truth.csv` is a null value; a column the
section does not have is ignored.
"""

import csv
import itertools
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    FIELDS,
    PERCENT_FIELDS,
    RowKey,
    Section,
    TextRow,
    Values,
    load_gazetteer,
)
from ebola_2026_outbreak.drc_2026_sitrep_textlayer import read_rows, row_matches

FIXTURES = Path(__file__).parent / "fixtures" / "sitrep_words"
GAZETTEER = load_gazetteer()


def read_truth() -> dict[str, dict[RowKey, Values]]:
    truth: defaultdict[str, dict[RowKey, Values]] = defaultdict(dict)
    with (FIXTURES / "truth.csv").open() as f:
        for r in csv.DictReader(f):
            key = RowKey(r["section"], r["province"], r["zone"])
            truth[r["sitrep"]][key] = {
                field: None
                if r[field] == ""
                else float(r[field])
                if field in PERCENT_FIELDS
                else int(r[field])
                for field in FIELDS[key.section]
            }
    return dict(truth)


TRUTH = read_truth()


@pytest.fixture(scope="module", params=sorted(TRUTH))
def report(request: pytest.FixtureRequest) -> tuple[str, dict[RowKey, TextRow]]:
    sitrep = request.param
    return sitrep, read_rows(FIXTURES / f"SitRep_{sitrep}.json", GAZETTEER)


def test_truth_rows_located_and_matched(
    report: tuple[str, dict[RowKey, TextRow]],
) -> None:
    sitrep, rows = report
    truth = TRUTH[sitrep]
    assert {k.section for k in truth} == {
        "provinces",
        "zone_subtotals",
        "zone_total",
        "zones",
    }
    failed = [k for k, v in truth.items() if not row_matches(rows[k], v, k.section)]
    assert failed == []
    # Every located row is a truth row: the reader invents none.
    assert {k for k, r in rows.items() if r.located} == set(truth)


def mutations(values: Values, section: Section) -> list[Values]:
    """Single-cell changes: null to 0, a value to null, +1 on a count, +0.1
    on a percentage, and a value swapped with an adjacent blank cell."""
    out = []
    for field, value in values.items():
        if value is None:
            changed = [0.0 if field in PERCENT_FIELDS else 0]
        else:
            changed = [None, value + (0.1 if field in PERCENT_FIELDS else 1)]
        out.extend(values | {field: c} for c in changed)
    for a, b in itertools.pairwise(FIELDS[section]):
        if (values[a] is None) != (values[b] is None):
            out.append(values | {a: values[b], b: values[a]})
    return out


def test_single_cell_mutations_fail(
    report: tuple[str, dict[RowKey, TextRow]],
) -> None:
    sitrep, rows = report
    passed = [
        (key, mutated)
        for key, values in TRUTH[sitrep].items()
        for mutated in mutations(values, key.section)
        if row_matches(rows[key], mutated, key.section)
    ]
    assert passed == []


ISIRO = RowKey("zones", "Haut-Uélé", "Isiro")


def test_zero_and_blank_swap_fails() -> None:
    # SitRep 96 prints Isiro as "58 24 41,4% 3 0 _ 0": 0 community deaths and
    # a blank CTE cell. Swapped, the row prints the same tokens.
    rows = read_rows(FIXTURES / "SitRep_096.json", GAZETTEER)
    truth = TRUTH["096"][ISIRO]
    assert (truth["community_deaths"], truth["cte_deaths"]) == (0, None)
    swapped = truth | {"community_deaths": None, "cte_deaths": 0}

    assert row_matches(rows[ISIRO], truth, "zones")
    assert not row_matches(rows[ISIRO], swapped, "zones")


ZONE = RowKey("zones", "Ituri", "Bunia")


def _edit_157(tmp_path: Path, edit: Callable[[str], str | None]) -> Path:
    """SitRep 157's dump with each word line passed through `edit`, which
    returns the new line or None to drop it."""
    dump = json.loads((FIXTURES / "SitRep_157.json").read_text())
    dump["pages"] = {
        page: [e for w in words if (e := edit(w)) is not None]
        for page, words in dump["pages"].items()
    }
    path = tmp_path / "SitRep_157.json"
    path.write_text(json.dumps(dump))
    return path


def test_unprinted_row_not_located(tmp_path: Path) -> None:
    path = _edit_157(tmp_path, lambda w: None if w.endswith(" Bunia") else w)

    rows = read_rows(path, GAZETTEER)

    assert rows[ZONE] == TextRow(ZONE, None, False, None, None)
    assert not row_matches(rows[ZONE], TRUTH["157"][ZONE], "zones")
    assert rows[RowKey("zones", "Ituri", "Damas")].located


def _shift(text: str, dx: float) -> Callable[[str], str]:
    """An edit moving each dump word printing `text` by `dx` points."""

    def edit(word: str) -> str:
        top, x0, x1, t = word.split(" ", 3)
        if t != text:
            return word
        return f"{top} {float(x0) + dx} {float(x1) + dx} {t}"

    return edit


@pytest.mark.parametrize(
    ("edit", "located"),
    [
        # Without its "intra-CTE" header word the zone table is not read.
        (lambda w: None if w.endswith(" intra-CTE") else w, False),
        # A third "(n)" header word leaves the zone columns unresolved.
        (lambda w: w.replace(" Nouv.", " (n)"), True),
        # "Létalité" moved right of "Nouv. Cas" puts the columns out of order.
        (_shift("Létalité", 200), True),
        # Bunia's 1895 cases moved to the boundary of the deaths column, then
        # into it beside the 633 deaths.
        (_shift("1895", 21), True),
        (_shift("1895", 42), True),
    ],
    ids=["no-intra-cte", "repeated-header", "out-of-order", "boundary", "shared"],
)
def test_unplaced_values_hold_the_row(
    tmp_path: Path, edit: Callable[[str], str | None], located: bool
) -> None:
    rows = read_rows(_edit_157(tmp_path, edit), GAZETTEER)

    assert rows[ZONE].located is located
    assert rows[ZONE].cells is None
    assert not row_matches(rows[ZONE], TRUTH["157"][ZONE], "zones")


@pytest.mark.parametrize(
    ("cells", "values", "matches"),
    [
        (
            ("1 895", "633", "33,4 %", "14", "4", "1", "5"),
            (1895, 633, 33.4, 14, 4, 1, 5),
            True,
        ),
        (
            ("1895", "633", "33,4%", "14", "4", "1", "5"),
            (1895, 633, 33.4, 14, 4, 1, 5),
            True,
        ),
        # A blank and a 0 differ, in both directions.
        (("12", "6", "50,0%", "1", "0", None, "0"), (12, 6, 50.0, 1, 0, None, 0), True),
        (
            ("12", "6", "50,0%", "1", "0", None, "0"),
            (12, 6, 50.0, 1, None, 0, 0),
            False,
        ),
        (("12", "6", "50,0%", "1", "0", None, "0"), (12, 6, 50.0, 1, 0, 0, 0), False),
        # A CFR prints with 0 to 2 decimals, within 0.05 of the value.
        (
            ("16", "5", "31,3%", None, None, None, "0"),
            (16, 5, 31.25, None, None, None, 0),
            True,
        ),
        (
            ("16", "5", "31%", None, None, None, "0"),
            (16, 5, 31.25, None, None, None, 0),
            False,
        ),
        (
            ("2", "2", "100,00%", None, None, None, "0"),
            (2, 2, 100.0, None, None, None, 0),
            True,
        ),
        (
            ("2", "2", "100%", None, None, None, "0"),
            (2, 2, 100.0, None, None, None, 0),
            True,
        ),
        # NA is a token; a null matches it or a blank, and a value matches neither.
        (
            ("NA", "45", "NA", None, None, None, "0"),
            (None, 45, None, None, None, None, 0),
            True,
        ),
        (
            ("NA", "45", "NA", None, None, None, "0"),
            (0, 45, None, None, None, None, 0),
            False,
        ),
        # A count must be a whole, non-negative number.
        (
            ("2", "2", "100,0%", None, None, None, "0"),
            (2, 2, 100.0, None, None, None, -0.5),
            False,
        ),
    ],
)
def test_row_matches(
    cells: tuple[str | None, ...], values: tuple[float | None, ...], matches: bool
) -> None:
    fields = FIELDS["zones"]
    row = TextRow(ZONE, "", True, 3, dict(zip(fields, cells, strict=True)))
    assert row_matches(row, dict(zip(fields, values, strict=True)), "zones") is matches


def test_unplaced_row_never_matches() -> None:
    row = TextRow(ZONE, "1 895 633 33,4 % 14 4 1 5", True, 3, None)
    values = dict(zip(FIELDS["zones"], (1895, 633, 33.4, 14, 4, 1, 5), strict=True))
    assert not row_matches(row, values, "zones")
