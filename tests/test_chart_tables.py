"""Tests for the chart-ready tables."""

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
    country_summary,
    drc_outbreak_series,
    headline_figures,
    latest_date,
    prose_date,
    province_map,
    weekly,
    weekly_comparison,
    west_africa_series,
)

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


def test_province_map_matches_accents_and_drops_zero() -> None:
    out = province_map(
        provinces([("Ituri", 10), ("Haut-Uélé", 5), ("Kasai", 0)]), LOOKUP
    )
    assert out.rows() == [("Ituri", 3, 10), ("Haut-Uele", 68, 5)]


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


def test_headline_figures(who: pl.DataFrame) -> None:
    mapped = province_map(provinces([("Ituri", 10), ("Haut-Uélé", 5)]), LOOKUP)
    figures = headline_figures(
        summary=country_summary(who), who_as_of=D2, mapped=mapped, sitrep_as_of=D1
    )
    assert figures == {
        "total_confirmed_cases": {"value": 1009, "update_date": "18 October"},
        "total_deaths": {"value": 483, "update_date": "18 October"},
        "death_rate": {"value": 47.9, "update_date": "18 October"},
        "provinces_affected": {"value": 2, "update_date": "17 October"},
    }


def test_headline_figures_rejects_missing_total(who: pl.DataFrame) -> None:
    summary = country_summary(who).filter(pl.col("Country") != "Total")
    with pytest.raises(ValueError, match="one 'Total' row"):
        headline_figures(summary=summary, who_as_of=D2, mapped=LOOKUP, sitrep_as_of=D2)


def test_latest_date_rejects_empty() -> None:
    with pytest.raises(ValueError, match="No dates"):
        latest_date(pl.Series("date", [], dtype=pl.Date))


def test_prose_date_drops_leading_zero_and_year() -> None:
    assert prose_date(date(2026, 6, 5)) == "5 June"
