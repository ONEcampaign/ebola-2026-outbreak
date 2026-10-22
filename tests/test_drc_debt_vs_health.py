"""Tests for DR Congo debt service against government health spending."""

import polars as pl
import pytest

from ebola_2026_outbreak.drc_debt_vs_health import (
    DEBT,
    ENTITY_NAME,
    HEALTH,
    combine,
    debt_service,
)


def net_flows(rows: list[tuple[int, str, str, float | None]]) -> pl.DataFrame:
    """Net flows rows from (year, entity_name, prices, value_outflows) tuples."""
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "entity_name": pl.String,
            "prices": pl.String,
            "value_outflows": pl.Float64,
        },
        orient="row",
    )


def test_debt_service_sums_one_entity_in_millions() -> None:
    df = net_flows(
        [
            (2022, ENTITY_NAME, "current", -600e6),
            (2022, ENTITY_NAME, "current", -254.5e6),
            (2023, ENTITY_NAME, "current", 0.0),
            (2022, "Congo", "current", -1e9),
        ]
    )
    out = debt_service(df)
    assert out.rows() == [(2022, 854.5), (2023, 0.0)]
    # A zero year stays a plain zero rather than -0.0.
    assert str(out["value"][1]) == "0.0"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(2022, "Congo", "current", -1.0)], "No net flows rows"),
        ([(2022, ENTITY_NAME, "constant", -1.0)], "current prices"),
        ([(2022, ENTITY_NAME, "current", None)], "null outflows"),
        ([(2022, ENTITY_NAME, "current", 1.0)], "positive outflows"),
    ],
)
def test_debt_service_rejects(
    rows: list[tuple[int, str, str, float | None]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        debt_service(net_flows(rows))


def test_debt_service_rejects_missing_column() -> None:
    df = net_flows([(2022, ENTITY_NAME, "current", -1.0)]).drop("prices")
    with pytest.raises(ValueError, match="prices"):
        debt_service(df)


def test_combine_keeps_years_from_either_source() -> None:
    debt = pl.DataFrame({"year": [2023, 2024], "value": [399.174, 386.16]})
    health = pl.DataFrame({"year": [2022, 2023], "value": [458.84, 464.089]})
    out = combine(debt=debt, health=health, first_year=2022)
    assert out.columns == ["Country", "Year", DEBT, HEALTH]
    assert out.select("Year", DEBT, HEALTH).rows() == [
        (2022, None, 458.84),
        (2023, 399.17, 464.09),
        (2024, 386.16, None),
    ]
    assert out["Country"].unique().to_list() == ["DR Congo"]


def test_combine_rejects_duplicate_years() -> None:
    debt = pl.DataFrame({"year": [2023, 2023], "value": [1.0, 2.0]})
    health = pl.DataFrame({"year": [2023], "value": [1.0]})
    with pytest.raises(ValueError, match="duplicate years"):
        combine(debt=debt, health=health)


def test_combine_drops_years_before_first_year() -> None:
    debt = pl.DataFrame({"year": [2012, 2013], "value": [1.0, 2.0]})
    health = pl.DataFrame({"year": [2012, 2013], "value": [3.0, 4.0]})
    out = combine(debt=debt, health=health, first_year=2013)
    assert out.select("Year", DEBT, HEALTH).rows() == [(2013, 2.0, 4.0)]
