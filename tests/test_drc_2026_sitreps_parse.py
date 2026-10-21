"""Tests for the sitrep province table parser."""

from datetime import date
from pathlib import Path

import pytest

from ebola_2026_outbreak.drc_2026_sitreps_parse import (
    TOTAL,
    ProvinceCounts,
    find_province_table,
    parse_counts,
    parse_sitrep,
    readings,
    validate_table,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sitrep_tables"

# Total row per fixture, one per layout or wrap variant seen in the sitreps.
EXPECTED_TOTALS = {
    "2026-06-15": ProvinceCounts(29, 837, 196),  # new cases last, "20 sur 36"
    "2026-08-10": ProvinceCounts(68, 4449, 2061),  # new cases last, one "ND"
    "2026-08-28": ProvinceCounts(82, 5945, 2862),  # new cases first
    "2026-09-08": ProvinceCounts(86, 6843, 3310),  # Total numbers on line above
    "2026-09-13": ProvinceCounts(58, 7258, 3510),  # "Sud-" / numbers / "Ubangi**"
    "2026-09-15": ProvinceCounts(59, 7404, 3577),  # "Sud- 100,0" / ... / "Ubangi** %"
    "2026-09-16": ProvinceCounts(71, 7475, 3605),  # "Sud-Ubangi 100,0" / ... / "** %"
    "2026-10-18": ProvinceCounts(97, 9552, 4602),  # "TOTAL", 8 provinces
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED_TOTALS.items())
def test_find_province_table(name: str, expected: ProvinceCounts) -> None:
    table = find_province_table((FIXTURES / f"{name}.txt").read_text())
    validate_table(table)
    assert table[TOTAL] == expected


def test_wrapped_province_is_kept() -> None:
    table = find_province_table((FIXTURES / "2026-09-15.txt").read_text())
    assert table["Sud-Ubangi"] == ProvinceCounts(0, 1, 1)


def test_missing_new_cases_is_none() -> None:
    table = find_province_table((FIXTURES / "2026-08-10.txt").read_text())
    assert table["Haut-Uélé"] == ProvinceCounts(None, 113, 48)


def test_parse_sitrep_long_rows() -> None:
    df = parse_sitrep(
        (FIXTURES / "2026-10-18.txt").read_text(),
        report_date=date(2026, 10, 18),
        sitrep_number=157,
        media_id=710,
    )
    assert df.height == 8 * 3
    assert TOTAL not in df["province"].to_list()
    cases = df.filter(df["indicator"] == "Cumulative confirmed cases")
    assert cases["value"].sum() == 9552


def test_readings_groups_thousands() -> None:
    assert [5, 659] in readings(["5", "659"])
    assert [5659] in readings(["5", "659"])
    assert [4911] in readings(["4911"])


def test_readings_rejects_leading_zero_group() -> None:
    # "0 316" is new cases 0 then 316 cases, never 316 read as one number.
    assert [316] not in readings(["0", "316"])


@pytest.mark.parametrize(
    ("rest", "expected"),
    [
        ("44 5 659 2 583 45,6 % 28/36 25/36", ProvinceCounts(44, 5659, 2583)),
        ("0 316 130 41,1% 7/13 (53,8 %)", ProvinceCounts(0, 316, 130)),
        ("767 157 20,5% 20 sur 36 (55,5 %) 29", ProvinceCounts(29, 767, 157)),
        ("113 48 42,5% 6/13 (46,1 %) ND", ProvinceCounts(None, 113, 48)),
    ],
)
def test_parse_counts(rest: str, expected: ProvinceCounts) -> None:
    assert parse_counts(rest) == expected


def test_parse_counts_rejects_cfr_mismatch() -> None:
    with pytest.raises(ValueError, match="readings match CFR"):
        parse_counts("44 5 659 2 583 12,0 %")


def test_validate_table_rejects_bad_sum() -> None:
    table = {
        "Ituri": ProvinceCounts(1, 10, 5),
        "Tshopo": ProvinceCounts(0, 3, 1),
        TOTAL: ProvinceCounts(1, 14, 6),
    }
    with pytest.raises(ValueError, match="sum to 13 cases"):
        validate_table(table)


def test_no_table_raises() -> None:
    with pytest.raises(ValueError, match="No province table"):
        find_province_table("Ituri (85), au Nord-Kivu (19).\nAucun tableau.")
