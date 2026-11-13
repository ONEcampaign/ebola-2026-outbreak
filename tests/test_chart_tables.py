"""Tests for the chart-ready tables."""

import logging
from datetime import date, timedelta

import polars as pl
import pytest

from ebola_2026_outbreak.chart_tables import (
    CASES,
    CFR,
    DRC,
    DRC_FIRST_CONFIRMED,
    RECOVERED,
    WEEK,
    WEST_AFRICA,
    WEST_AFRICA_FIRST_CONFIRMED,
    check_province_agreement,
    country_summary,
    drc_outbreak_series,
    headline_figures,
    latest_date,
    prose_date,
    province_map,
    summary_date,
    weekly,
    weekly_comparison,
    west_africa_series,
    with_newer_sitrep,
    zone_map,
    zone_map_date,
)
from ebola_2026_outbreak.drc_2026_sitrep_schema import GAZETTEER_PATH
from ebola_2026_outbreak.drc_health_zone_geometry import COD_AB_PATH, zone_codes

D1, D2 = date(2026, 10, 17), date(2026, 10, 18)


def who_rows(rows: list[tuple[date, str, str, float | None]]) -> pl.DataFrame:
    """Long WHO rows from (date, country, indicator, value) tuples."""
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "country": pl.String,
            "indicator": pl.String,
            "value": pl.Float64,
        },
        orient="row",
    )


@pytest.fixture
def who() -> pl.DataFrame:
    """DRC on two dates and Kenya with no recoveries reported."""
    return who_rows(
        [
            (D1, "DRC", "Confirmed cases", 900.0),
            (D1, "DRC", "Confirmed deaths", 400.0),
            (D1, "DRC", "Recovered", 100.0),
            (D2, "DRC", "Confirmed cases", 1000.0),
            (D2, "DRC", "Confirmed deaths", 482.0),
            (D2, "DRC", "Recovered", 200.0),
            (D1, "Kenya", "Confirmed cases", 9.0),
            (D1, "Kenya", "Confirmed deaths", 1.0),
            (D1, "Kenya", "Recovered", None),
        ]
    )


def test_country_summary_uses_each_countrys_latest_date(who: pl.DataFrame) -> None:
    table = country_summary(who)
    rows = {r["Country"]: r for r in table.iter_rows(named=True)}
    assert rows["DRC"][CASES] == 1000
    assert rows["DRC"][CFR] == 48.2
    assert rows["Kenya"][CFR] == 11.1
    assert rows["Kenya"][RECOVERED] is None


def test_country_summary_total_row(who: pl.DataFrame) -> None:
    total = country_summary(who).row(-1, named=True)
    assert total["Country"] == "Total"
    assert total[CASES] == 1009
    assert total[RECOVERED] == 200  # Kenya's missing value is skipped.
    assert total[CFR] == round(483 / 1009 * 100, 1)


def provinces(rows: list[tuple[str, int]]) -> pl.DataFrame:
    """Cumulative case rows for the latest sitrep plus an older one."""
    latest = [(D2, p, "Cumulative confirmed cases", v) for p, v in rows]
    older = [(D1, "Ituri", "Cumulative confirmed cases", 1)]
    return pl.DataFrame(
        latest + older,
        schema={
            "date": pl.Date,
            "province": pl.String,
            "indicator": pl.String,
            "value": pl.Int64,
        },
        orient="row",
    )


LOOKUP = pl.DataFrame(
    {"Name": ["Ituri", "Haut-Uele", "Kasaï"], "OBJECTID": [3, 68, 128]}
)


def test_province_map_matches_accents() -> None:
    out = province_map(provinces([("Ituri", 10), ("Haut-Uélé", 5)]), LOOKUP)
    assert out.rows() == [("Ituri", 3, 10), ("Haut-Uele", 68, 5), ("Kasaï", 128, None)]


def test_province_map_keeps_every_lookup_province() -> None:
    out = province_map(provinces([("Ituri", 10), ("Kasai", 0)]), LOOKUP)
    assert out.rows() == [
        ("Ituri", 3, 10),
        ("Haut-Uele", 68, None),
        ("Kasaï", 128, None),
    ]


def test_province_map_raises_on_unknown_province() -> None:
    with pytest.raises(ValueError, match="No OBJECTID"):
        province_map(provinces([("Kinshasa", 1)]), LOOKUP)


def test_weekly_takes_last_value_in_week() -> None:
    start = date(2026, 5, 15)
    series = pl.DataFrame(
        {
            "date": [start, start + timedelta(days=6), start + timedelta(days=14)],
            "value": [1, 5, 9],
        }
    )
    out = weekly(series, start=start, name="x").sort(WEEK)
    assert out.rows() == [(0, 5), (2, 9)]


def test_weekly_rejects_data_before_week_zero() -> None:
    series = pl.DataFrame({"date": [date(2026, 5, 1)], "value": [1]})
    with pytest.raises(ValueError, match="before its week 0"):
        weekly(series, start=date(2026, 5, 15), name="x")


def test_weekly_comparison_keeps_empty_weeks_null() -> None:
    wa = pl.DataFrame(
        {"date": [WEST_AFRICA_FIRST_CONFIRMED + timedelta(weeks=3)], "value": [100]}
    )
    drc = pl.DataFrame({"date": [DRC_FIRST_CONFIRMED], "value": [5]})
    out = weekly_comparison(wa, drc)
    assert out[WEEK].to_list() == [0, 1, 2, 3]
    assert out[WEST_AFRICA].to_list() == [None, None, None, 100]
    assert out[DRC].to_list() == [5, None, None, None]


def test_drc_outbreak_series_carries_countries_forward() -> None:
    who = who_rows(
        [
            (D1, "Uganda", "Confirmed cases", 20.0),
            (D1, "DRC", "Confirmed cases", 900.0),
            (D2, "DRC", "Confirmed cases", 1000.0),
        ]
    )
    assert drc_outbreak_series(who).rows() == [(D1, 920.0), (D2, 1020.0)]


def test_west_africa_series_joins_without_overlap() -> None:
    early = pl.DataFrame({"date": [date(2014, 8, 28)], "value": [1899.0]})
    who = pl.DataFrame({"date": [date(2014, 8, 29)], "value": [1754.0]})
    assert west_africa_series(early, who)["value"].to_list() == [1899.0, 1754.0]


def test_west_africa_series_rejects_overlap() -> None:
    early = pl.DataFrame({"date": [date(2014, 8, 29)], "value": [1899.0]})
    who = pl.DataFrame({"date": [date(2014, 8, 29)], "value": [1754.0]})
    with pytest.raises(ValueError, match="overlaps"):
        west_africa_series(early, who)


ZONES = pl.DataFrame(
    {"Code": ["CD1", "CD2", "CD3"], "Name": ["Bunia", "Aru", "Mambasa"]}
).with_columns(pl.Series(CASES, [9, 0, None], dtype=pl.Int64))


def test_headline_figures(who: pl.DataFrame) -> None:
    mapped = province_map(provinces([("Ituri", 10), ("Haut-Uélé", 5)]), LOOKUP)
    figures = headline_figures(
        summary=country_summary(who),
        summary_as_of=D2,
        mapped=mapped,
        sitrep_as_of=D1,
        zones=ZONES,
        zone_as_of=date(2026, 10, 16),
    )
    assert figures == {
        "total_confirmed_cases": {"value": 1009, "update_date": "18 October"},
        "total_deaths": {"value": 483, "update_date": "18 October"},
        "death_rate": {"value": 47.9, "update_date": "18 October"},
        "provinces_affected": {"value": 2, "update_date": "17 October"},
        "zones_affected": {"value": 1, "update_date": "16 October"},
    }


def test_headline_figures_rejects_missing_total(who: pl.DataFrame) -> None:
    summary = country_summary(who).filter(pl.col("Country") != "Total")
    with pytest.raises(ValueError, match="one 'Total' row"):
        headline_figures(
            summary=summary,
            summary_as_of=D2,
            mapped=LOOKUP,
            sitrep_as_of=D2,
            zones=ZONES,
            zone_as_of=D2,
        )


def test_summary_date_uses_drc_date_over_newer_country(
    caplog: pytest.LogCaptureFixture,
) -> None:
    drc = "Democratic Republic of the Congo"
    latest = who_rows(
        [
            (D1, drc, "Confirmed cases", 1000.0),
            (D2, "Kenya", "Confirmed cases", 9.0),
        ]
    )
    with caplog.at_level(logging.INFO):
        assert summary_date(latest) == D1
    assert "Kenya is as of 2026-10-18" in caplog.text


def test_summary_date_rejects_missing_drc() -> None:
    latest = who_rows([(D2, "Kenya", "Confirmed cases", 9.0)])
    with pytest.raises(ValueError, match="No 'Democratic Republic of the Congo'"):
        summary_date(latest)


def test_latest_date_rejects_empty() -> None:
    with pytest.raises(ValueError, match="No dates"):
        latest_date(pl.Series("date", [], dtype=pl.Date))


def test_prose_date_drops_leading_zero_and_year() -> None:
    assert prose_date(date(2026, 6, 5)) == "5 June"


DRC_NAME = "Democratic Republic of the Congo"
D3 = date(2026, 10, 19)


def sitrep(day: date, rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    """Province cases and deaths on one sitrep date."""
    records = [
        (day, p, label, v)
        for p, cases, deaths in rows
        for label, v in (
            ("Cumulative confirmed cases", cases),
            ("Cumulative confirmed deaths", deaths),
        )
    ]
    return pl.DataFrame(
        records,
        schema={
            "date": pl.Date,
            "province": pl.String,
            "indicator": pl.String,
            "value": pl.Int64,
        },
        orient="row",
    )


@pytest.fixture
def who_drc() -> pl.DataFrame:
    """WHO DRC figures on D2 and Kenya on D1."""
    return who_rows(
        [
            (D2, DRC_NAME, "Confirmed cases", 1000.0),
            (D2, DRC_NAME, "Confirmed deaths", 482.0),
            (D2, DRC_NAME, "Recovered", 200.0),
            (D1, "Kenya", "Confirmed cases", 9.0),
            (D1, "Kenya", "Confirmed deaths", 1.0),
        ]
    )


def test_with_newer_sitrep_replaces_drc_row(who_drc: pl.DataFrame) -> None:
    latest = with_newer_sitrep(
        who_drc, sitrep(D3, [("Ituri", 900, 400), ("Tshopo", 150, 90)])
    )
    rows = {r["Country"]: r for r in country_summary(latest).iter_rows(named=True)}
    assert rows[DRC_NAME][CASES] == 1050
    assert rows[DRC_NAME]["Confirmed deaths"] == 490
    assert rows[DRC_NAME][RECOVERED] == 200  # WHO's, as sitreps have none.
    assert rows["Kenya"][CASES] == 9
    assert rows["Total"][CASES] == 1059
    assert latest_date(latest["date"]) == D3


@pytest.mark.parametrize("day", [D1, D2])
def test_with_newer_sitrep_keeps_who_when_not_older(
    who_drc: pl.DataFrame, day: date
) -> None:
    assert with_newer_sitrep(who_drc, sitrep(day, [("Ituri", 5, 1)])).equals(who_drc)


def test_with_newer_sitrep_rejects_missing_deaths(who_drc: pl.DataFrame) -> None:
    cases_only = sitrep(D3, [("Ituri", 900, 400)]).filter(
        pl.col("indicator") == "Cumulative confirmed cases"
    )
    with pytest.raises(ValueError, match="no national"):
        with_newer_sitrep(who_drc, cases_only)


def test_drc_outbreak_series_ends_on_newer_sitrep(who_drc: pl.DataFrame) -> None:
    latest = with_newer_sitrep(who_drc, sitrep(D3, [("Ituri", 1050, 490)]))
    assert drc_outbreak_series(latest).rows() == [
        (D1, 9.0),
        (D2, 1009.0),
        (D3, 1059.0),
    ]


def zone_rows(rows: list[tuple[date, str, str, int]]) -> pl.DataFrame:
    """Sitrep zone cumulative case rows from (date, province, zone, value)."""
    return pl.DataFrame(
        [(d, "zone", p, z, "Cumulative confirmed cases", v) for d, p, z, v in rows],
        schema={
            "date": pl.Date,
            "level": pl.String,
            "province": pl.String,
            "zone": pl.String,
            "indicator": pl.String,
            "value": pl.Int64,
        },
        orient="row",
    )


ZONE_CODES = pl.DataFrame(
    {
        "province": ["Ituri", "Ituri", "Bas-Uélé"],
        "zone": ["Bunia", "A ventiler", "Dakwa"],
        "code": ["CD1", None, None],
    }
)
ZONE_SHAPES = pl.DataFrame({"Code": ["CD1", "CD2"], "Name": ["Bunia", "Aru"]})


def test_zone_map_keeps_every_map_zone() -> None:
    by_zone = zone_rows(
        [
            (D1, "Ituri", "Bunia", 5),
            (D2, "Ituri", "Bunia", 9),
            (D2, "Ituri", "A ventiler", 4),
            (D2, "Bas-Uélé", "Dakwa", 0),
        ]
    )
    out = zone_map(by_zone, ZONE_CODES, ZONE_SHAPES, as_of=D2)
    assert out.rows() == [("CD1", "Bunia", 9), ("CD2", "Aru", None)]


# Dakwa has a null code; Nizi is not in the codes at all.
@pytest.mark.parametrize(
    ("province", "zone"), [("Bas-Uélé", "Dakwa"), ("Ituri", "Nizi")]
)
def test_zone_map_rejects_cases_without_code(province: str, zone: str) -> None:
    with pytest.raises(ValueError, match="gazetteer alias"):
        zone_map(
            zone_rows([(D2, province, zone, 3)]), ZONE_CODES, ZONE_SHAPES, as_of=D2
        )


def test_zone_map_rejects_new_gazetteer_zone_cod_ab_spells_differently() -> None:
    cod_ab = pl.read_csv(COD_AB_PATH, infer_schema=False)
    gazetteer = pl.concat(
        [
            pl.read_csv(GAZETTEER_PATH, infer_schema=False),
            pl.DataFrame(
                {"province": ["Ituri"], "zone": ["Bunya"], "aliases": [None]},
                schema={"province": pl.String, "zone": pl.String, "aliases": pl.String},
            ),
        ]
    )
    codes = zone_codes(cod_ab, gazetteer)
    shapes = cod_ab.select(pl.col("code").alias("Code"), pl.col("zone").alias("Name"))
    with pytest.raises(ValueError, match="Ituri / Bunya"):
        zone_map(zone_rows([(D2, "Ituri", "Bunya", 3)]), codes, shapes, as_of=D2)


def test_zone_map_rejects_code_without_shape() -> None:
    codes = ZONE_CODES.with_columns(code=pl.lit("CD9"))
    with pytest.raises(ValueError, match="no map shape"):
        zone_map(zone_rows([(D2, "Ituri", "Bunia", 3)]), codes, ZONE_SHAPES, as_of=D2)


def test_zone_map_sums_rows_sharing_a_code() -> None:
    codes = pl.DataFrame(
        {
            "province": ["Bas-Uélé", "Bas-Uélé"],
            "zone": ["Ango", "Dakwa"],
            "code": ["CD2", "CD2"],
        }
    )
    by_zone = zone_rows([(D2, "Bas-Uélé", "Ango", 2), (D2, "Bas-Uélé", "Dakwa", 1)])
    out = zone_map(by_zone, codes, ZONE_SHAPES, as_of=D2)
    assert out.rows() == [("CD2", "Aru", 3), ("CD1", "Bunia", None)]


def test_zone_map_rejects_missing_case_indicator() -> None:
    renamed = zone_rows([(D2, "Ituri", "Bunia", 3)]).with_columns(
        indicator=pl.lit("Cumulative cases")
    )
    with pytest.raises(ValueError, match="No zone rows"):
        zone_map(renamed, ZONE_CODES, ZONE_SHAPES, as_of=D2)


def test_zone_map_date_ignores_province_rows() -> None:
    by_zone = pl.concat(
        [
            zone_rows([(D1, "Ituri", "Bunia", 3)]),
            zone_rows([(D2, "Ituri", "", 3)]).with_columns(level=pl.lit("province")),
        ]
    )
    assert zone_map_date(by_zone) == D1


CASES_LABEL, DEATHS_LABEL = "Cumulative confirmed cases", "Cumulative confirmed deaths"
MEDIA, MODIFIED = 26137, "2026-11-12T05:33:36"


def parsed(
    rows: list[tuple[str, int, int]], *, day: date = D2, media_id: int = MEDIA
) -> pl.DataFrame:
    """Parsed province cases and deaths from (province, cases, deaths)."""
    return sitrep(day, rows).with_columns(
        media_id=pl.lit(media_id, dtype=pl.Int64), modified=pl.lit(MODIFIED)
    )


def zone_table(
    subtotals: list[tuple[str, int, int]], *, media_id: int = MEDIA
) -> pl.DataFrame:
    """Zone table on D2: one zone row plus province subtotal rows."""
    records = [(D2, "zone", "Ituri", "Bunia", CASES_LABEL, 1)] + [
        (D2, "province", p, None, label, v)
        for p, cases, deaths in subtotals
        for label, v in ((CASES_LABEL, cases), (DEATHS_LABEL, deaths))
    ]
    return pl.DataFrame(
        records,
        schema={
            "date": pl.Date,
            "level": pl.String,
            "province": pl.String,
            "zone": pl.String,
            "indicator": pl.String,
            "value": pl.Int64,
        },
        orient="row",
    ).with_columns(media_id=pl.lit(media_id, dtype=pl.Int64), modified=pl.lit(MODIFIED))


def ledger(discrepancies: int = 0) -> pl.DataFrame:
    """Extraction ledger row for the zone table's upload."""
    return pl.DataFrame(
        {
            "media_id": [MEDIA],
            "modified": [MODIFIED],
            "status": ["released"],
            "discrepancies": [discrepancies],
        }
    )


def test_check_province_agreement_matches_by_name_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    older = parsed([("Ituri", 1, 1)], day=D1)  # Other dates are not compared.
    provinces = pl.concat([older, parsed([("Haut-Uélé", 10, 4), ("Ituri", 5, 2)])])
    by_zone = zone_table([("HAUT UELE", 10, 4), ("Ituri", 5, 2)])
    with caplog.at_level(logging.INFO):
        check_province_agreement(provinces, by_zone, ledger(), as_of=D2)
    assert "subtotals match" in caplog.text


def test_check_province_agreement_rejects_different_value() -> None:
    by_zone = zone_table([("Ituri", 5, 3)])
    with pytest.raises(
        ValueError,
        match=r"2026-10-18.*Ituri Cumulative confirmed deaths: parsed 2, zone table 3",
    ):
        check_province_agreement(parsed([("Ituri", 5, 2)]), by_zone, ledger(), as_of=D2)


@pytest.mark.parametrize(
    ("parsed_rows", "subtotals", "missing"),
    [
        ([("Ituri", 5, 2), ("Tshopo", 1, 0)], [("Ituri", 5, 2)], "zone table None"),
        ([("Ituri", 5, 2)], [("Ituri", 5, 2), ("Tshopo", 1, 0)], "parsed None"),
    ],
)
def test_check_province_agreement_rejects_province_in_one_table(
    parsed_rows: list[tuple[str, int, int]],
    subtotals: list[tuple[str, int, int]],
    missing: str,
) -> None:
    with pytest.raises(
        ValueError, match=f"Tshopo Cumulative confirmed cases.*{missing}"
    ):
        check_province_agreement(
            parsed(parsed_rows), zone_table(subtotals), ledger(), as_of=D2
        )


def test_check_province_agreement_skips_no_parsed_rows_on_zone_date(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        check_province_agreement(
            parsed([("Ituri", 5, 2)], day=D1),
            zone_table([("Ituri", 9, 9)]),
            ledger(),
            as_of=D2,
        )
    assert "no rows for the zone map's date" in caplog.text


def test_check_province_agreement_skips_other_upload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provinces = parsed([("Ituri", 5, 2)], media_id=MEDIA - 1)
    with caplog.at_level(logging.WARNING):
        check_province_agreement(
            provinces, zone_table([("Ituri", 9, 9)]), ledger(), as_of=D2
        )
    assert f"[({MEDIA - 1}, '{MODIFIED}')]" in caplog.text
    assert f"[({MEDIA}, '{MODIFIED}')]" in caplog.text


def test_check_province_agreement_skips_replaced_upload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """INSP can replace an upload in place, changing only `modified`."""
    replaced = "2026-11-13T09:00:00"
    provinces = parsed([("Ituri", 5, 2)]).with_columns(modified=pl.lit(replaced))
    with caplog.at_level(logging.WARNING):
        check_province_agreement(
            provinces, zone_table([("Ituri", 9, 9)]), ledger(), as_of=D2
        )
    assert f"'{replaced}'" in caplog.text
    assert f"'{MODIFIED}'" in caplog.text
    assert "skipping" in caplog.text


def test_check_province_agreement_skips_upload_with_discrepancies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    by_zone = zone_table([("Ituri", 9, 9)])
    with caplog.at_level(logging.WARNING):
        check_province_agreement(
            parsed([("Ituri", 5, 2)]), by_zone, ledger(2), as_of=D2
        )
    assert "2 publisher discrepancies" in caplog.text


def test_check_province_agreement_rejects_upload_missing_from_ledger() -> None:
    by_zone = zone_table([("Ituri", 5, 2)])
    with pytest.raises(ValueError, match="not in ledger"):
        check_province_agreement(
            parsed([("Ituri", 5, 2)]),
            by_zone,
            ledger().with_columns(media_id=1),
            as_of=D2,
        )
