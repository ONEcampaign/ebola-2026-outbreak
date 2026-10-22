"""Tests for reconciling two extractions and checking a sitrep's sums."""

from collections.abc import Iterable
from typing import Any

import pytest

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    COUNT_FIELDS,
    FIELDS,
    Discrepancy,
    Extraction,
    Flag,
    Normalised,
    RowKey,
    Section,
    TextRow,
    Values,
    normalise,
)
from ebola_2026_outbreak.drc_2026_sitrep_validate import check_sums, reconcile

KPIS = RowKey("kpis", "", "")
TOTAL = RowKey("zone_total", "", "")
ITURI = RowKey("zone_subtotals", "Ituri", "")
ITURI_PROVINCE = RowKey("provinces", "Ituri", "")
A = RowKey("zones", "Ituri", "A")
B = RowKey("zones", "Ituri", "B")
C = RowKey("zones", "Ituri", "C")

# (cum_confirmed, cum_deaths, new_confirmed, community, cte, total deaths).
ZONES = {
    "Ituri": {
        "A": (10, 2, 1, None, 1, 1),
        "B": (20, 5, 0, 1, 0, 1),
        "C": (4, 0, None, None, None, None),
    },
    "Nord-Kivu": {
        "D": (8, 4, 2, 0, 0, 0),
        "E": (30, 6, 1, 1, 1, 2),
        "F": (6, 3, 0, None, None, None),
    },
}


def cfr(cases: int, deaths: int) -> float:
    return round(100 * deaths / cases, 1)


def table_row(counts: Iterable[int | None]) -> Values:
    cases, deaths, *daily = counts
    assert cases is not None and deaths is not None
    row: Values = dict(zip(COUNT_FIELDS, (cases, deaths, *daily), strict=True))
    return {
        f: cfr(cases, deaths) if f == "cfr_pct" else row[f] for f in FIELDS["zones"]
    }


def report() -> dict[RowKey, Values]:
    """A consistent report: two provinces of three zones, blanks included."""
    rows: dict[RowKey, Values] = {}
    for province, zones in ZONES.items():
        for zone, counts in zones.items():
            rows[RowKey("zones", province, zone)] = table_row(counts)
        sums = [sum(c[i] or 0 for c in zones.values()) for i in range(6)]
        subtotal = rows[RowKey("zone_subtotals", province, "")] = table_row(sums)
        rows[RowKey("provinces", province, "")] = {
            "new_confirmed": subtotal["new_confirmed"],
            "cum_confirmed": subtotal["cum_confirmed"],
            "cum_deaths": subtotal["cum_deaths"],
            "cfr_pct": subtotal["cfr_pct"],
            "zs_affected": 3,
            "zs_total": 3,
        }
    subtotals = [v for k, v in rows.items() if k.section == "zone_subtotals"]
    total = rows[TOTAL] = table_row(
        [sum(s[f] or 0 for s in subtotals) for f in COUNT_FIELDS]
    )
    rows[KPIS] = {
        "cum_confirmed": total["cum_confirmed"],
        "cum_deaths": total["cum_deaths"],
        "cfr_pct": total["cfr_pct"],
        "in_isolation": 5,
        "cum_recovered": 7,
        "contact_followup_pct": 91.5,
    }
    return rows


def changed(
    rows: dict[RowKey, Values], key: RowKey, **fields: Any
) -> dict[RowKey, Values]:
    return {**rows, key: {**rows[key], **fields}}


def printed(values: Values, section: Section) -> str:
    """The row as the text layer prints it: blank cells leave no token."""
    tokens = []
    for field in FIELDS[section]:
        value = values[field]
        if isinstance(value, float):
            tokens.append(f"{value:.1f}".replace(".", ","))
        elif value is not None:
            tokens.append(str(value))
    return " ".join(tokens)


def match(text: TextRow, values: Values, section: Section) -> bool:
    return text.printed == printed(values, section)


def text_layer(rows: dict[RowKey, Values]) -> dict[RowKey, TextRow]:
    return {
        key: TextRow(key, printed(values, key.section), True, 2, None)
        for key, values in rows.items()
        if key.section != "kpis"
    }


def model(rows: dict[RowKey, Values], flags: Iterable[Flag] = ()) -> Normalised:
    return Normalised(Extraction(150, "2026-10-11", rows), list(flags))


def run(
    gemini: dict[RowKey, Values],
    mistral: dict[RowKey, Values],
    text_rows: dict[RowKey, TextRow] | None = None,
    **kwargs: Any,
):
    text_rows = text_layer(report()) if text_rows is None else text_rows
    return reconcile(model(gemini), model(mistral), text_rows, match=match, **kwargs)


def reasons(flags: Iterable[Flag]) -> dict[tuple[RowKey, str | None], str]:
    return {(f.key, f.field): f.reason for f in flags}


def test_agreeing_models_that_match_the_text_are_accepted() -> None:
    truth = report()
    assert run(truth, truth) == (truth, [])


WRONG = {"cum_confirmed": 11}
OTHER_WRONG = {"cum_confirmed": 12}
ALL = FIELDS["zones"]


@pytest.mark.parametrize(
    ("gemini", "mistral", "arbitration", "accepted", "flagged"),
    [
        pytest.param(
            WRONG,
            WRONG,
            True,
            {},
            dict.fromkeys(ALL, "text_mismatch"),
            id="agree-text-mismatch",
        ),
        pytest.param({}, WRONG, True, None, {}, id="disagree-gemini-matches"),
        pytest.param(WRONG, {}, True, None, {}, id="disagree-mistral-matches"),
        pytest.param(
            {},
            WRONG,
            False,
            "agreed",
            {"cum_confirmed": "models_disagree"},
            id="disagree-one-matches-no-arbitration",
        ),
        pytest.param(
            WRONG,
            OTHER_WRONG,
            True,
            {},
            {
                f: "models_disagree" if f == "cum_confirmed" else "text_mismatch"
                for f in ALL
            },
            id="disagree-neither-matches",
        ),
    ],
)
def test_row_acceptance(
    gemini: dict[str, int],
    mistral: dict[str, int],
    arbitration: bool,
    accepted: Values | str | None,
    flagged: dict[str, str],
) -> None:
    truth = report()
    result = run(
        changed(truth, A, **gemini),
        changed(truth, A, **mistral),
        arbitration=arbitration,
    )

    if accepted is None:
        accepted = truth[A]
    elif accepted == "agreed":
        accepted = {f: v for f, v in truth[A].items() if f != "cum_confirmed"}
    assert result.accepted[A] == accepted
    assert reasons(result.flags) == {(A, f): r for f, r in flagged.items()}
    assert {f.page for f in result.flags} <= {2}
    assert {k: v for k, v in result.accepted.items() if k != A} == {
        k: v for k, v in truth.items() if k != A
    }


def test_row_not_located_holds_the_row() -> None:
    truth = report()
    text_rows = text_layer(truth)
    del text_rows[A]
    text_rows[B] = TextRow(B, None, False, None, None)

    result = run(truth, truth, text_rows)

    assert result.accepted[A] == result.accepted[B] == {}
    assert reasons(result.flags) == {
        (k, f): "row_not_located" for k in (A, B) for f in ALL
    }


def test_row_from_one_model_is_arbitrated_by_the_text() -> None:
    truth = report()
    without_a = {k: v for k, v in truth.items() if k != A}

    assert run(truth, without_a) == (truth, [])
    held = run(truth, without_a, arbitration=False)
    assert held.accepted[A] == {}
    assert reasons(held.flags) == {(A, f): "models_disagree" for f in ALL}


def test_name_flags_hold_the_row_once_per_cell() -> None:
    truth = report()
    gemini = model(truth, [Flag(A, f, "wrong_province") for f in ALL])
    mistral = model(truth, [Flag(A, f, "unknown_name") for f in ALL])

    result = reconcile(gemini, mistral, text_layer(truth), match=match)

    assert result.accepted[A] == {}
    assert result.flags == [Flag(A, f, "wrong_province", 2) for f in ALL]


def test_held_row_skips_the_sums_that_need_it() -> None:
    truth = report()
    gemini = model(truth, [Flag(A, f, "unknown_name") for f in ALL])

    result = reconcile(gemini, model(truth), text_layer(truth), match=match)

    assert check_sums(result.accepted, set()) == ([], [])


@pytest.mark.parametrize(
    ("mistral", "flagged"),
    [
        pytest.param({}, {}, id="agree"),
        pytest.param(
            {"cfr_pct": 25.7, "in_isolation": None},
            {"cfr_pct": "models_disagree", "in_isolation": "models_disagree"},
            id="disagree",
        ),
    ],
)
def test_kpis_need_both_models(mistral: Values, flagged: dict[str, str]) -> None:
    truth = report()

    result = run(truth, changed(truth, KPIS, **mistral))

    assert result.accepted[KPIS] == {
        f: v for f, v in truth[KPIS].items() if f not in flagged
    }
    assert reasons(result.flags) == {(KPIS, f): r for f, r in flagged.items()}
    assert all(f.page is None for f in result.flags)


def test_kpis_from_one_model_are_held() -> None:
    truth = report()
    without_kpis = {k: v for k, v in truth.items() if k != KPIS}

    result = run(without_kpis, truth)

    assert result.accepted[KPIS] == {}
    assert reasons(result.flags) == {
        (KPIS, f): "models_disagree" for f in FIELDS["kpis"]
    }


@pytest.mark.parametrize(
    ("gemini_cfr", "mistral_cfr", "agree"),
    [(48.3, 48.30000001, False), (48.3, 48.3, True), (48, 48.0, True)],
)
def test_percentages_agree_only_when_exact(
    gemini_cfr: float, mistral_cfr: float, agree: bool
) -> None:
    def normalised(cfr_pct: float) -> Normalised:
        kpis = dict.fromkeys(FIELDS["kpis"], 1) | {"cfr_pct": cfr_pct}
        raw = {
            "sitrep_no": 150,
            "report_date": "2026-10-11",
            "kpis": kpis,
            "provinces": [],
            "zone_subtotals": [],
            "zone_total": None,
            "zones": [],
        }
        return normalise(raw, {})

    result = reconcile(normalised(gemini_cfr), normalised(mistral_cfr), {}, match=match)

    assert ("cfr_pct" in result.accepted[KPIS]) is agree
    assert reasons(result.flags) == (
        {} if agree else {(KPIS, "cfr_pct"): "models_disagree"}
    )


def test_consistent_report_passes_the_sum_checks() -> None:
    # Zone C and the daily columns hold blanks, which count as 0.
    assert check_sums(report(), set()) == ([], [])


ITURI_ZONES = [A, B, C]
NORD_KIVU = RowKey("zone_subtotals", "Nord-Kivu", "")


@pytest.mark.parametrize(
    ("key", "fields", "flagged"),
    [
        pytest.param(
            A,
            {"new_confirmed": 2},
            [(k, "new_confirmed") for k in [ITURI, *ITURI_ZONES]],
            id="zones-to-subtotal",
        ),
        pytest.param(
            TOTAL,
            {"cte_deaths": 3},
            [(k, "cte_deaths") for k in [TOTAL, ITURI, NORD_KIVU]],
            id="subtotals-to-total",
        ),
        pytest.param(
            ITURI_PROVINCE,
            {"cum_deaths": 8, "cfr_pct": cfr(34, 8)},
            [(ITURI_PROVINCE, "cum_deaths"), (ITURI, "cum_deaths")],
            id="provinces-to-subtotals",
        ),
        pytest.param(
            KPIS,
            {"cum_confirmed": 79, "cfr_pct": cfr(79, 20)},
            [(KPIS, "cum_confirmed"), (TOTAL, "cum_confirmed")],
            id="total-to-kpis",
        ),
        pytest.param(
            B,
            {"cfr_pct": 25.2},
            [(B, "cfr_pct"), (B, "cum_deaths"), (B, "cum_confirmed")],
            id="cfr",
        ),
    ],
)
def test_failed_sum_flags_machine_accepted_operands(
    key: RowKey, fields: Values, flagged: list[tuple[RowKey, str]]
) -> None:
    values = changed(report(), key, **fields)

    flags, discrepancies = check_sums(values, set())

    assert sorted(flags) == sorted(Flag(k, f, "sum_mismatch") for k, f in flagged)
    assert discrepancies == []


def test_failed_sum_on_verified_operands_is_a_discrepancy() -> None:
    values = changed(report(), A, new_confirmed=2)
    operands = {(k, "new_confirmed") for k in [ITURI, *ITURI_ZONES]}

    assert check_sums(values, operands) == (
        [],
        [Discrepancy(ITURI, "new_confirmed", 1, 2)],
    )
    flags, discrepancies = check_sums(values, operands - {(A, "new_confirmed")})
    assert flags == [Flag(A, "new_confirmed", "sum_mismatch")]
    assert discrepancies == []


def test_cfr_discrepancy_records_the_recomputed_rate() -> None:
    values = changed(report(), B, cfr_pct=25.2)
    verified = {(B, f) for f in ("cfr_pct", "cum_deaths", "cum_confirmed")}

    assert check_sums(values, verified) == ([], [Discrepancy(B, "cfr_pct", 25.2, 25.0)])


@pytest.mark.parametrize(
    ("cases", "deaths", "cfr_pct", "passes"),
    [
        (16, 3, 18.7, True),
        (16, 3, 18.8, True),
        (16, 3, 18.9, False),
        (0, 0, 0.0, True),
        (0, 0, 5.0, False),
        (3, 1, None, True),
    ],
)
def test_cfr_tolerance(
    cases: int, deaths: int, cfr_pct: float | None, passes: bool
) -> None:
    values = changed(
        report(), C, cum_confirmed=cases, cum_deaths=deaths, cfr_pct=cfr_pct
    )

    flags, _ = check_sums(values, set())

    assert ((C, "cfr_pct") not in reasons(flags)) is passes


def test_operand_under_review_skips_the_check() -> None:
    values = changed(report(), A, new_confirmed=2)
    values[B] = {f: v for f, v in values[B].items() if f != "new_confirmed"}

    assert check_sums(values, set()) == ([], [])


@pytest.mark.parametrize(
    ("dropped", "missing_rows"),
    [
        pytest.param({ITURI}, [ITURI], id="subtotal-of-province-with-zones"),
        pytest.param(
            {ITURI, *ITURI_ZONES}, [ITURI], id="subtotal-of-province-in-summary"
        ),
        pytest.param({ITURI, NORD_KIVU}, [ITURI, NORD_KIVU], id="every-subtotal"),
        pytest.param({TOTAL}, [TOTAL], id="zone-total"),
        pytest.param({KPIS}, [KPIS], id="kpis"),
    ],
)
def test_missing_row_is_flagged_for_entry(
    dropped: set[RowKey], missing_rows: list[RowKey]
) -> None:
    values = {k: v for k, v in report().items() if k not in dropped}

    assert check_sums(values, set()) == (
        [
            Flag(k, f, "section_missing")
            for k in sorted(missing_rows)
            for f in FIELDS[k.section]
        ],
        [],
    )


def test_subtotal_without_zones_flags_the_missing_zones() -> None:
    values = {k: v for k, v in report().items() if k not in ITURI_ZONES}

    assert check_sums(values, set()) == (
        [Flag(RowKey("zones", "Ituri", ""), None, "section_missing")],
        [],
    )


def test_empty_report_flags_every_section() -> None:
    flags, _ = check_sums({}, set())

    assert reasons(flags) == {
        (RowKey("provinces", "", ""), None): "section_missing",
        (RowKey("zone_subtotals", "", ""), None): "section_missing",
        (RowKey("zones", "", ""), None): "section_missing",
        **{(KPIS, f): "section_missing" for f in FIELDS["kpis"]},
        **{(TOTAL, f): "section_missing" for f in FIELDS["zone_total"]},
    }


def test_shift_into_blank_cell_fails_the_daily_sum() -> None:
    # Both models move zone A's CTE death into the blank community column.
    # The stub text layer compares tokens, and "10 2 20,0 1 1 1" reads the
    # same either way, so it accepts the shift; the zones-to-subtotal sum
    # still sees it.
    truth = report()
    shifted = changed(truth, A, community_deaths=1, cte_deaths=None)

    result = run(shifted, shifted)
    flags, discrepancies = check_sums(result.accepted, set())

    assert result.flags == []
    assert result.accepted[A] == shifted[A]
    assert reasons(flags) == {
        (k, f): "sum_mismatch"
        for f in ("community_deaths", "cte_deaths")
        for k in [ITURI, *ITURI_ZONES]
    }
    assert discrepancies == []
