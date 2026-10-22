"""Tests for the sitrep text-layer row reader.

`fixtures/sitrep_words/` holds `dump_words` fixtures of the 13 golden
reports (SitReps 96 to 157 in the current layout) and `truth.csv`, their
hand-checked province, zone, zone subtotal and zone Total rows under
canonical keys. A blank cell in `truth.csv` is a null value; a column the
section does not have is ignored.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    FIELDS,
    PERCENT_FIELDS,
    RowKey,
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


def mutations(values: Values) -> list[Values]:
    """Single-cell changes: null to 0, a value to null, +1 on a count, +0.1
    on a percentage."""
    out = []
    for field, value in values.items():
        if value is None:
            changed = [0.0 if field in PERCENT_FIELDS else 0]
        else:
            changed = [None, value + (0.1 if field in PERCENT_FIELDS else 1)]
        out.extend(values | {field: c} for c in changed)
    return out


def test_single_cell_mutations_fail(
    report: tuple[str, dict[RowKey, TextRow]],
) -> None:
    sitrep, rows = report
    passed = [
        (key, mutated)
        for key, values in TRUTH[sitrep].items()
        for mutated in mutations(values)
        if row_matches(rows[key], mutated, key.section)
    ]
    assert passed == []


ZONE = RowKey("zones", "Ituri", "Bunia")


def test_unprinted_row_not_located(tmp_path: Path) -> None:
    dump = json.loads((FIXTURES / "SitRep_157.json").read_text())
    dump["pages"] = {
        page: [w for w in words if not w.endswith(" Bunia")]
        for page, words in dump["pages"].items()
    }
    path = tmp_path / "SitRep_157.json"
    path.write_text(json.dumps(dump))

    rows = read_rows(path, GAZETTEER)

    assert rows[ZONE] == TextRow(ZONE, None, False, None)
    assert not row_matches(rows[ZONE], TRUTH["157"][ZONE], "zones")
    assert rows[RowKey("zones", "Ituri", "Damas")].located


@pytest.mark.parametrize(
    ("printed", "values", "matches"),
    [
        ("1 895 633 33,4 % 14 4 1 5", (1895, 633, 33.4, 14, 4, 1, 5), True),
        ("1895 633 33,4% 14 4 1 5", (1895, 633, 33.4, 14, 4, 1, 5), True),
        # A space splits a number only at a thousands boundary.
        ("18 95 633 33,4% 14 4 1 5", (1895, 633, 33.4, 14, 4, 1, 5), False),
        ("12 6 50,0% 1 0 0 0", (12, 6, 50.0, 1, 0, 0, 0), True),
        ("1 26 50,0% 1 0 0 0", (12, 6, 50.0, 1, 0, 0, 0), False),
        # A CFR prints with 0 to 2 decimals, within 0.05 of the value.
        ("16 5 31,3% 0", (16, 5, 31.25, None, None, None, 0), True),
        ("16 5 31% 0", (16, 5, 31.25, None, None, None, 0), False),
        ("2 2 100,00% 0", (2, 2, 100.0, None, None, None, 0), True),
        ("2 2 100% 0", (2, 2, 100.0, None, None, None, 0), True),
        # NA is a token; a null matches it or nothing.
        ("NA 45 NA 0", (None, 45, None, None, None, None, 0), True),
        ("45 0", (None, 45, None, None, None, None, 0), True),
        ("NA 45 NA 0", (0, 45, None, None, None, None, 0), False),
        # A count must be a whole, non-negative number.
        ("2 2 100,0% 0", (2, 2, 100.0, None, None, None, -0.5), False),
    ],
)
def test_row_matches(
    printed: str, values: tuple[float | None, ...], matches: bool
) -> None:
    row = TextRow(ZONE, printed, True, 3)
    assert (
        row_matches(row, dict(zip(FIELDS["zones"], values, strict=True)), "zones")
        is matches
    )
