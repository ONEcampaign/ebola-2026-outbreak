"""Reconcile two models' extractions of a sitrep and check its sums.

`reconcile` decides, row by row, which values two models agree on and the
PDF text layer confirms. A table row is accepted when both models give the
same row and it matches the row's printed cells. When the models disagree
and exactly one model's row matches the printed cells, that row is accepted
(text-layer arbitration, switched off with `arbitration=False`). Every other
row is held: a row the text layer did not locate, a row whose name is unknown
or filed under the wrong province, or a row both models read the same way but
the text contradicts has every field flagged. A row the models disagree on
and the text cannot settle has the fields the models dispute flagged
`models_disagree`, and, when neither model matches the text, the rest flagged
`text_mismatch`. The headline indicators have no text-layer row, so each is
accepted only when both models give the same value.

A zone row that one model alone names, under a name the gazetteer lacks and
the text layer does not print (`printed_labels`), is a misread of a zone the
other model read correctly ("Mili-Murhesa" for "Miti-Murhesa"). It is
dropped with a log line instead of holding the report, provided its province
is canonical and every zone row the text layer locates there came back from
a model under its canonical name. A name both models give, or one the page
prints, is still flagged `unknown_name`, so a zone new to the gazetteer
reaches a reviewer. A located zone neither model returned may have the
misread as its only copy, so the misread is held, which covers a zone
printing nothing but zeros and blanks that no sum check would catch. A
dropped row standing for a zone the text layer did not locate leaves its
province's zones short of the subtotal row, and the first sum check below
flags the subtotal unless that zone too prints only zeros and blanks.

Model agreement is exact, percentages included: 48.3 and 48.30000001
disagree. Only the matcher compares a printed CFR within 0.05. The matcher
compares each field with the cell printed in its own column, so a value
moved into an adjacent blank cell, or a 0 swapped with a blank, does not
match.

`check_sums` then checks the report's arithmetic on the accepted and reviewed
values:

1. zones add up to their province's subtotal row in the zone table;
2. the subtotal rows add up to the zone table's Total row;
3. cumulative cases and deaths of the province summary table equal the
   zone table's subtotal rows;
4. cumulative cases and deaths of the zone table's Total row equal the
   headline indicators;
5. every printed CFR is within 0.05 of 100 * deaths / cases;
6. a report missing a row it must print fails: the headline row, the zone
   table's Total row, and a subtotal row for each province with zone or
   summary rows. Each field of the missing row is flagged
   `section_missing`, so a reviewer can enter it. A subtotal row with no
   zone rows under it flags its province's zones as a missing section, and
   a section with no row at all is flagged as a whole.

A blank counts as 0 in a sum. A check with an operand still under review is
skipped, and a missing row is flagged by check 6, so no check goes unchecked
silently. A failed check flags `sum_mismatch` on every operand a machine
accepted, since two models can make the same error the text layer cannot
see. A failed check whose operands a reviewer has all verified is a
publisher discrepancy, released as printed and recorded.
"""

import logging
from collections.abc import Callable, Iterable, Set

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    COUNT_FIELDS,
    FIELDS,
    SECTIONS,
    Discrepancy,
    Flag,
    FlagReason,
    Normalised,
    Number,
    Reconciled,
    RowKey,
    Section,
    TextRow,
    Values,
    name_key,
)
from ebola_2026_outbreak.drc_2026_sitreps_parse import PROVINCES

logger = logging.getLogger(__name__)

# Largest gap between a printed CFR and the CFR its counts give.
CFR_TOLERANCE = 0.05

# Whether a model's values reproduce a row's printed cells.
type Matcher = Callable[[TextRow, Values, Section], bool]
# The name keys, of those given, the text layer prints as a row label.
type Labels = Callable[[Set[str]], Set[str]]
# One value of the report: (row, field).
type Cell = tuple[RowKey, str]

_CUMULATIVE = ("cum_confirmed", "cum_deaths")
_PROVINCES = frozenset(PROVINCES.values())
_KPIS = RowKey("kpis", "", "")
_TOTAL = RowKey("zone_total", "", "")


# Reconciliation


def reconcile(
    gemini: Normalised,
    mistral: Normalised,
    text_rows: dict[RowKey, TextRow],
    *,
    match: Matcher,
    printed_labels: Labels,
    arbitration: bool = True,
) -> Reconciled:
    """Accept the values two models agree on and the text layer confirms.

    Every row either model returned is a key of `accepted`, holding only its
    accepted fields; a row held whole maps to an empty dict. `check_sums`
    relies on this to skip a sum whose operands are still under review.

    Args:
        gemini: Gemini's extraction, from `normalise`.
        mistral: Mistral's extraction, from `normalise`.
        text_rows: Printed text-layer rows by key; a missing key is a row
            the text layer did not locate.
        match: Whether a model's values reproduce a printed row.
        printed_labels: Which unknown zone names the text layer prints, to
            tell a new zone from one model's misread.
        arbitration: Accept the one model whose row matches the text when
            the models disagree.

    Returns:
        The accepted values and one flag per held value, in row key order.
        Name flags from either model's `normalise` carry through, with the
        row's page added. A misread zone row (see the module docstring) is
        left out of both.
    """
    misread = _misreads(gemini, mistral, text_rows, printed_labels)
    for key in sorted(misread):
        logger.info("Dropping %s: one model's misread zone name", key)
    gemini_rows = {k: v for k, v in gemini.extraction.rows.items() if k not in misread}
    mistral_rows = {
        k: v for k, v in mistral.extraction.rows.items() if k not in misread
    }
    name_reasons: dict[Cell, FlagReason] = {}
    for flag in [*gemini.name_flags, *mistral.name_flags]:
        if flag.field is not None and flag.key not in misread:
            name_reasons.setdefault((flag.key, flag.field), flag.reason)
    in_doubt = {key for key, _ in name_reasons}

    accepted: dict[RowKey, Values] = {}
    flags: list[Flag] = []
    for key in sorted(gemini_rows.keys() | mistral_rows.keys() | in_doubt):
        text = text_rows.get(key)
        page = text.page if text is not None else None
        if key in in_doubt:
            values: Values = {}
            reasons = {
                f: name_reasons[key, f]
                for f in FIELDS[key.section]
                if (key, f) in name_reasons
            }
        else:
            values, reasons = _reconcile_row(
                key,
                gemini_rows.get(key),
                mistral_rows.get(key),
                text,
                match=match,
                arbitration=arbitration,
            )
        accepted[key] = values
        flags.extend(
            Flag(key, field, reason, page) for field, reason in reasons.items()
        )
    return Reconciled(accepted, flags)


def _misreads(
    gemini: Normalised,
    mistral: Normalised,
    text_rows: dict[RowKey, TextRow],
    printed_labels: Labels,
) -> set[RowKey]:
    """Zone rows flagged `unknown_name` whose name only one model gives and
    the text layer does not print, in a province whose located zone rows the
    models all returned."""
    gemini_names, mistral_names = (
        {name_key(k.zone) for k in m.extraction.rows if k.section == "zones"}
        for m in (gemini, mistral)
    )
    returned = gemini.extraction.rows.keys() | mistral.extraction.rows.keys()
    unreturned = {
        key.province
        for key, text in text_rows.items()
        if key.section == "zones" and text.located and key not in returned
    }
    unknown = {
        f.key: name_key(f.key.zone)
        for f in [*gemini.name_flags, *mistral.name_flags]
        if f.reason == "unknown_name"
        and f.key.section == "zones"
        and f.key.zone
        and f.key.province in _PROVINCES - unreturned
    }
    candidates = {n for n in unknown.values() if n in gemini_names ^ mistral_names}
    printed = printed_labels(candidates) if candidates else set()
    return {k for k, n in unknown.items() if n in candidates - printed}


def _reconcile_row(
    key: RowKey,
    gemini: Values | None,
    mistral: Values | None,
    text: TextRow | None,
    *,
    match: Matcher,
    arbitration: bool,
) -> tuple[Values, dict[str, FlagReason]]:
    """Accepted values and flag reasons by field for one row.

    A row only one model returned disagrees on every field.
    """
    fields = FIELDS[key.section]
    if gemini is None or mistral is None:
        disputed = list(fields)
        agreed: Values = {}
    else:
        disputed = [f for f in fields if gemini[f] != mistral[f]]
        agreed = {f: gemini[f] for f in fields if f not in disputed}

    if key.section == "kpis":
        return agreed, dict.fromkeys(disputed, "models_disagree")
    if text is None or not text.located or text.printed is None:
        return {}, dict.fromkeys(fields, "row_not_located")

    matching = [
        values
        for values in (gemini, mistral)
        if values is not None and match(text, values, key.section)
    ]
    if not disputed:
        if matching:
            return agreed, {}
        return {}, dict.fromkeys(fields, "text_mismatch")
    if len(matching) == 1 and arbitration:
        return dict(matching[0]), {}
    if matching:
        # The text confirms the fields the models share; only the disputed
        # ones stay open.
        return agreed, dict.fromkeys(disputed, "models_disagree")
    return {}, {
        f: "models_disagree" if f in disputed else "text_mismatch" for f in fields
    }


# Sum checks


def check_sums(
    values: dict[RowKey, Values], verified: set[Cell]
) -> tuple[list[Flag], list[Discrepancy]]:
    """Check the report's arithmetic on accepted and reviewed values.

    Args:
        values: Accepted values with reviewed values applied, keyed as by
            `reconcile`. A field absent from its row is under review, and
            every check that needs it is skipped.
        verified: Cells whose value a reviewer entered.

    Returns:
        `section_missing` flags on every field of each missing row, on the
        zones of each subtotal row with none, and on each section with no
        row at all, then `sum_mismatch` flags on the
        machine-accepted operands of every failed check, one per cell, and a
        discrepancy for each failed check whose operands are all verified.
    """
    provinces = sorted({k.province for k in values if k.province in _PROVINCES})
    expected = {_KPIS, _TOTAL, *(RowKey("zone_subtotals", p, "") for p in provinces)}
    known = values.keys() | expected
    flags = [
        Flag(RowKey(section, "", ""), None, "section_missing")
        for section in SECTIONS
        if not any(key.section == section for key in known)
    ]
    flags.extend(
        Flag(key, field, "section_missing")
        for key in sorted(expected - values.keys())
        for field in FIELDS[key.section]
    )
    zone_provinces = {k.province for k in values if k.section == "zones"}
    flags.extend(
        Flag(RowKey("zones", key.province, ""), None, "section_missing")
        for key in sorted(values)
        if key.section == "zone_subtotals" and key.province not in zone_provinces
    )
    mismatched: dict[Cell, None] = {}
    discrepancies: list[Discrepancy] = []

    for target, operands, printed, computed in _failures(values, provinces):
        unverified = [c for c in (target, *operands) if c not in verified]
        if unverified:
            mismatched.update(dict.fromkeys(unverified))
        else:
            discrepancies.append(Discrepancy(*target, printed, computed))

    flags.extend(Flag(key, field, "sum_mismatch") for key, field in mismatched)
    return flags, discrepancies


def _failures(
    values: dict[RowKey, Values], provinces: list[str]
) -> Iterable[tuple[Cell, list[Cell], Number | None, Number]]:
    """Failed checks as (target, operands, printed target, computed value)."""

    def present(cells: Iterable[Cell]) -> bool:
        return all(key in values and field in values[key] for key, field in cells)

    def value(cell: Cell) -> Number:
        key, field = cell
        return values[key][field] or 0

    for target, operands in _equations(values, provinces):
        if not present([target, *operands]):
            continue
        computed = sum(value(c) for c in operands)
        if value(target) != computed:
            yield target, operands, values[target[0]][target[1]], computed

    for key in values:
        cfr, deaths, cases = (
            (key, "cfr_pct"),
            (key, "cum_deaths"),
            (key, "cum_confirmed"),
        )
        if not present([cfr, deaths, cases]) or values[key]["cfr_pct"] is None:
            continue
        computed = 100 * value(deaths) / value(cases) if value(cases) else 0.0
        # The epsilon keeps a rounding tie in (3 / 16 is 18.75, printed 18.7).
        if abs(value(cfr) - computed) > CFR_TOLERANCE + 1e-9:
            yield cfr, [deaths, cases], values[key]["cfr_pct"], round(computed, 2)


def _equations(
    values: dict[RowKey, Values], provinces: list[str]
) -> Iterable[tuple[Cell, list[Cell]]]:
    """Every (target, operands) pair where the target should equal the sum.

    Pairs name rows that may be missing; `_failures` skips those.
    """
    subtotals = sorted(
        {RowKey("zone_subtotals", p, "") for p in provinces}
        | {k for k in values if k.section == "zone_subtotals"}
    )
    for subtotal in subtotals:
        zones = [
            k
            for k in values
            if k.section == "zones" and k.province == subtotal.province
        ]
        if not zones:
            # Flagged as missing zones by `check_sums`.
            continue
        for field in COUNT_FIELDS:
            yield (subtotal, field), [(z, field) for z in zones]

    for field in COUNT_FIELDS:
        yield (_TOTAL, field), [(s, field) for s in subtotals]

    for province in provinces:
        for field in _CUMULATIVE:
            yield (
                (RowKey("provinces", province, ""), field),
                [(RowKey("zone_subtotals", province, ""), field)],
            )

    for field in _CUMULATIVE:
        yield (_KPIS, field), [(_TOTAL, field)]
