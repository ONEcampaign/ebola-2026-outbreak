"""Tests for the sitrep province table parser."""

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ebola_2026_outbreak import drc_2026_sitreps_parse as parse_module
from ebola_2026_outbreak.drc_2026_sitreps_download import MANIFEST_SCHEMA
from ebola_2026_outbreak.drc_2026_sitreps_parse import (
    TOTAL,
    ProvinceCounts,
    SkipCategory,
    TableException,
    canonical_province,
    check_monotonic,
    find_province_table,
    join_split_rows,
    parse_counts,
    parse_sitrep,
    readings,
    validate_table,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sitrep_tables"
FIXTURE_TEXT_1018 = (FIXTURES / "2026-10-18.txt").read_text()

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
        modified="2026-10-20T14:41:10",
    )
    assert df.height == 8 * 3
    assert TOTAL not in df["province"].to_list()
    cases = df.filter(df["indicator"] == "Cumulative confirmed cases")
    assert cases["value"].sum() == 9552


# Canonical name -> spellings it must accept: COD-AB's (data/reference/
# cod_ab_health_zones.csv), the sitreps' and a spaced or unaccented variant.
PROVINCE_SPELLINGS = {
    "Bas-Uélé": ("Bas-Uele", "Bas-Uélé", "Bas Uele"),
    "Équateur": ("Equateur", "Équateur", "EQUATEUR"),
    "Haut-Katanga": ("Haut-Katanga", "Haut Katanga"),
    "Haut-Lomami": ("Haut-Lomami", "Haut Lomami"),
    "Haut-Uélé": ("Haut-Uele", "Haut-Uélé", "Haut Uélé"),
    "Ituri": ("Ituri", "ITURI"),
    "Kasaï": ("Kasaï", "Kasai"),
    "Kasaï-Central": ("Kasaï-Central", "Kasai Central"),
    "Kasaï-Oriental": ("Kasaï-Oriental", "Kasai-Oriental"),
    "Kinshasa": ("Kinshasa",),
    "Kongo-Central": ("Kongo-Central", "Kongo Central"),
    "Kwango": ("Kwango",),
    "Kwilu": ("Kwilu",),
    "Lomami": ("Lomami",),
    "Lualaba": ("Lualaba",),
    "Maï-Ndombe": ("Maï-Ndombe", "Mai-Ndombe", "Mai Ndombe"),
    "Maniema": ("Maniema",),
    "Mongala": ("Mongala",),
    "Nord-Kivu": ("Nord-Kivu", "Nord Kivu"),
    "Nord-Ubangi": ("Nord-Ubangi", "Nord Ubangi"),
    "Sankuru": ("Sankuru",),
    "Sud-Kivu": ("Sud-Kivu", "Sud Kivu"),
    "Sud-Ubangi": ("Sud-Ubangi", "Sud Ubangi"),
    "Tanganyika": ("Tanganyika",),
    "Tshopo": ("Tshopo",),
    "Tshuapa": ("Tshuapa",),
}


def test_every_province_has_spellings() -> None:
    cod_ab = pl.read_csv(
        Path(__file__).parents[1] / "data" / "reference" / "cod_ab_health_zones.csv"
    )
    assert len(PROVINCE_SPELLINGS) == 26
    assert set(cod_ab["province"]) <= {
        s for spellings in PROVINCE_SPELLINGS.values() for s in spellings
    }


@pytest.mark.parametrize(
    ("label", "name"),
    [(s, name) for name, spellings in PROVINCE_SPELLINGS.items() for s in spellings],
)
def test_canonical_province(label: str, name: str) -> None:
    assert canonical_province(label) == name


@pytest.mark.parametrize(
    ("lines", "name"),
    [
        (["Kasaï-", "0 1 1 100,0 % 1/16 —", "Central**"], "Kasaï-Central"),
        (["Kongo- 100,0", "0 1 1 1/16 —", "Central %"], "Kongo-Central"),
        (["Maï-Ndombe 100,0", "0 1 1 1/16 —", "** %"], "Maï-Ndombe"),
        (["Kasaï", "0 1 1 100,0 % 1/16 —", "Oriental**"], "Kasaï-Oriental"),
        (["Kasai 100,0", "0 1 1 1/16 —", "Central %"], "Kasaï-Central"),
        (["Kasaï 0 1 1 100,0 % 1/16 —", "Central**"], "Kasaï-Central"),
        (["Kasaï 0 1 1 100,0", "Oriental %"], "Kasaï-Oriental"),
    ],
)
def test_new_wrapped_province_is_rejoined(lines: list[str], name: str) -> None:
    (joined,) = join_split_rows(lines)
    assert canonical_province(joined.split()[0]) == name


def test_province_row_of_a_new_province_parses() -> None:
    text = "\n".join(
        [
            "Ituri 0 10 5 50,0 % 1/36",
            "Kasaï-Central 1 2 1 50,0 % 1/26",
            "Kasaï 0 4 1 25,0 % 1/18",
            "Total 1 16 7 43,8 %",
        ]
    )
    table = find_province_table(text)
    validate_table(table)
    assert table["Kasaï-Central"] == ProvinceCounts(1, 2, 1)
    assert table["Kasaï"] == ProvinceCounts(0, 4, 1)


def test_province_wrapped_at_its_space_is_not_kasai() -> None:
    # Read line by line, the Kasaï-Central row is a Kasaï row that still
    # balances against Total.
    text = "\n".join(
        [
            "Ituri 0 10 5 50,0 % 1/36",
            "Kasaï 1 2 1 50,0 % 1/26",
            "Central**",
            "Total 1 12 6 50,0 %",
        ]
    )
    table = find_province_table(text)
    validate_table(table)
    assert table["Kasaï-Central"] == ProvinceCounts(1, 2, 1)
    assert "Kasaï" not in table


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
        ("34 7380 3 769 * 51,1% 28/36 (77,8 %)", ProvinceCounts(34, 7380, 3769)),
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


# Sitreps with a misprinted Total CFR and a contacts table by province after the
# case table. Contacts captions differ: "Tableau 4.", "Tableau 2 :", "TABLEAU 4 —".
MISPRINTED_TOTALS = {
    "2026-06-17": (ProvinceCounts(21, 896, 232), "26,3 %", "25.89 %"),
    "2026-07-01": (ProvinceCounts(54, 1460, 452), "30,9 %", "30.96 %"),
    "2026-08-03": (ProvinceCounts(72, 3874, 1751), "45,1 %", "45.20 %"),
}

# A contacts table whose rates are exact, so its rows pass the CFR check.
CONTACTS_TABLE = """Tableau 4. Suivi des contacts des cas confirmés
Province Contacts sous suivi (n) Contacts vus (n) Taux de suivi (%)
Ituri 4 659 3 297 70,8%
Nord-Kivu 1 628 1 148 70,5%
Sud-Kivu 80 80 100,0%
Total 6 367 4 525 71,1%
"""

CASE_ROWS = """Ituri 10 5 50,0% 1
Tshopo 4 1 25,0% 0
"""


@pytest.mark.parametrize(
    ("name", "expected", "printed", "computed"),
    [(name, *values) for name, values in MISPRINTED_TOTALS.items()],
)
def test_misprinted_total_cfr_read_from_province_sums(
    name: str,
    expected: ProvinceCounts,
    printed: str,
    computed: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = (FIXTURES / f"{name}.txt").read_text()
    with caplog.at_level(logging.WARNING):
        table = find_province_table(text, source=f"{name} (media 1)")
    validate_table(table)
    assert table[TOTAL] == expected
    assert table["Ituri"].cases < expected.cases  # the case table
    assert f"{name} (media 1): Total row CFR printed {printed}" in caplog.text
    assert f"is {computed}" in caplog.text


def test_misprinted_total_warning_names_sitrep(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        parse_sitrep(
            (FIXTURES / "2026-06-17.txt").read_text(),
            report_date=date(2026, 6, 17),
            sitrep_number=None,
            media_id=901,
            modified="2026-06-18T10:00:00",
        )
    assert "2026-06-17 (media 901): Total row CFR printed 26,3 %" in caplog.text


def test_unreadable_total_stops_search() -> None:
    # Total says 15 cases and provinces sum to 14. The search stops there and
    # does not return the contacts table below.
    text = CASE_ROWS + "Total 15 6 41,0% 1\n" + CONTACTS_TABLE
    with pytest.raises(ValueError, match="Total row unreadable"):
        find_province_table(text)


def test_total_from_sums_checks_new_cases() -> None:
    # Cases and deaths equal the sums, but new cases (2) do not (1).
    with pytest.raises(ValueError, match="Total row unreadable"):
        find_province_table(CASE_ROWS + "Total 14 6 40,0% 2\n")


def test_contacts_table_before_case_table_is_skipped() -> None:
    text = CONTACTS_TABLE + "\n" + FIXTURE_TEXT_1018
    table = find_province_table(text)
    assert table[TOTAL] == EXPECTED_TOTALS["2026-10-18"]


def test_contacts_table_with_misprinted_total_is_skipped() -> None:
    # A contacts Total that fails its rate check must not stop the search.
    contacts = CONTACTS_TABLE.replace("71,1%", "71,9%")
    table = find_province_table(contacts + "\n" + FIXTURE_TEXT_1018)
    assert table[TOTAL] == EXPECTED_TOTALS["2026-10-18"]


def test_contacts_table_with_failed_first_row_is_skipped() -> None:
    # The failed first row must not move the table start below the header.
    contacts = CONTACTS_TABLE.replace("70,8%", "70,1%")
    with pytest.raises(ValueError, match="readings match CFR"):
        parse_counts("4 659 3 297 70,1%")
    table = find_province_table(contacts + "\n" + FIXTURE_TEXT_1018)
    assert table[TOTAL] == EXPECTED_TOTALS["2026-10-18"]


def test_headingless_case_table_parses() -> None:
    table = find_province_table(CASE_ROWS + "Total 14 6 42,9% 1\n")
    assert table[TOTAL] == ProvinceCounts(1, 14, 6)


def test_contacts_header_above_an_earlier_table_is_ignored() -> None:
    # The contacts header belongs to the table above it.
    text = CONTACTS_TABLE.replace("71,1%", "71,9%") + CASE_ROWS + "Total 14 6 42,9% 1\n"
    assert find_province_table(text)[TOTAL] == ProvinceCounts(1, 14, 6)


def test_no_table_raises() -> None:
    with pytest.raises(ValueError, match="No province table"):
        find_province_table("Ituri (85), au Nord-Kivu (19).\nAucun tableau.")


REPORT_DATE = date(2026, 10, 18)
MEDIA_ID = 710
OLD_MODIFIED, NEW_MODIFIED = "2026-10-19T09:00:00", "2026-10-20T14:41:10"
FIXTURE_TEXT = FIXTURE_TEXT_1018


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    modified: str,
    text: str,
    exceptions: dict[tuple[int, str], TableException],
) -> Path:
    """Run `main` on a one-row manifest with stubbed PDF text and exceptions.

    Returns:
        The output path, which exists only if `main` wrote it.
    """
    manifest = pl.DataFrame(
        [
            {
                "report_date": REPORT_DATE,
                "is_latest": True,
                "media_id": MEDIA_ID,
                "modified": modified,
                "source_url": f"https://example.org/{MEDIA_ID}.pdf",
            }
        ]
    )
    manifest = manifest.select(
        pl.col(c) if c in manifest.columns else pl.lit(None).alias(c)
        for c in MANIFEST_SCHEMA
    ).cast(MANIFEST_SCHEMA)
    manifest_path, output_path = tmp_path / "manifest.csv", tmp_path / "out.csv"
    manifest.write_csv(manifest_path)
    monkeypatch.setattr(parse_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(parse_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(parse_module, "PROVINCE_TABLE_EXCEPTIONS", exceptions)
    monkeypatch.setattr(parse_module, "ensure_pdf", lambda **_: Path("710.pdf"))
    monkeypatch.setattr(parse_module, "read_text", lambda _: text)
    parse_module.main()
    return output_path


def excepted(modified: str) -> dict[tuple[int, str], TableException]:
    """One exception for the test upload under `modified`."""
    return {
        (MEDIA_ID, modified): TableException(
            REPORT_DATE, SkipCategory.NO_TABLE, "alerts summary only"
        )
    }


def test_main_writes_nothing_when_a_sitrep_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="1 sitreps could not be parsed"):
        run_main(
            monkeypatch, tmp_path, modified=NEW_MODIFIED, text="no table", exceptions={}
        )
    assert not (tmp_path / "out.csv").exists()


def test_main_skips_excepted_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = run_main(
        monkeypatch,
        tmp_path,
        modified=OLD_MODIFIED,
        text="no table",
        exceptions=excepted(OLD_MODIFIED),
    )
    assert pl.read_csv(output_path).height == 0


def test_main_failure_names_exception_with_other_modified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="1 sitreps could not be parsed"):
        run_main(
            monkeypatch,
            tmp_path,
            modified=NEW_MODIFIED,
            text="no table",
            exceptions=excepted(OLD_MODIFIED),
        )
    assert (
        f"media {MEDIA_ID} has an exception for modified {OLD_MODIFIED}" in caplog.text
    )
    assert f"the manifest says {NEW_MODIFIED}" in caplog.text


def test_main_fails_on_excepted_upload_that_would_parse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The text parses cleanly, so only the exception check can stop the run.
    with pytest.raises(ValueError, match="1 sitreps could not be parsed"):
        run_main(
            monkeypatch,
            tmp_path,
            modified=NEW_MODIFIED,
            text=FIXTURE_TEXT,
            exceptions=excepted(OLD_MODIFIED),
        )
    assert not (tmp_path / "out.csv").exists()
    assert f"the manifest says {NEW_MODIFIED}" in caplog.text


def test_main_warns_on_exception_matching_no_latest_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exceptions = {
        **excepted(OLD_MODIFIED),
        (999, OLD_MODIFIED): TableException(
            date(2026, 5, 1), SkipCategory.NO_TABLE, "gone"
        ),
    }
    with caplog.at_level(logging.WARNING):
        run_main(
            monkeypatch,
            tmp_path,
            modified=OLD_MODIFIED,
            text="no table",
            exceptions=exceptions,
        )
    assert (
        f"Exception for media 999 modified {OLD_MODIFIED} (2026-05-01) matches no"
        in caplog.text
    )
    assert f"media {MEDIA_ID} modified" not in caplog.text


def test_check_monotonic_logs_province_fall(caplog: pytest.LogCaptureFixture) -> None:
    df = pl.DataFrame(
        {
            "date": [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 2)],
            "province": ["Ituri", "Ituri", "Ituri"],
            "indicator": [
                "Cumulative confirmed cases",
                "Cumulative confirmed cases",
                "New confirmed cases (24h)",
            ],
            "value": [10, 9, None],
        }
    )
    with caplog.at_level(logging.WARNING):
        check_monotonic(df)
    assert (
        "Ituri cumulative confirmed cases falls from 10 (2026-09-01) to 9 (2026-09-02)"
        in caplog.text
    )
