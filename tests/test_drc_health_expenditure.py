"""Tests for DR Congo expenditure by source."""

import polars as pl
import pytest

from ebola_2026_outbreak import drc_health_expenditure
from ebola_2026_outbreak.drc_health_expenditure import (
    PVT,
    SOURCE_ORDER,
    TOTAL,
    expenditure_by_source,
    sum_components,
)


def ghed(values: dict[str, dict[int, float | None]]) -> pl.DataFrame:
    """Long GHED rows from {indicator_code: {year: value}}."""
    return pl.DataFrame(
        [
            {"year": year, "indicator_code": code, "value": value}
            for code, by_year in values.items()
            for year, value in by_year.items()
        ],
        schema={"year": pl.Int64, "indicator_code": pl.String, "value": pl.Float64},
    )


def full_year(year: int = 2020) -> dict[str, dict[int, float | None]]:
    """One consistent year: CHE 100, gov 20, ext 40, OOP 30, other private 10."""
    return {
        "gghed_che": {year: 20},
        "ext_che": {year: 40},
        "hf3_che": {year: 30},
        "fs4": {year: 6},
        "fs5": {year: 5},
        "fs6": {year: 3},
        "fs61": {year: 4},
        "hf1": {year: 60},
        "hf2": {year: 10},
        "hf3": {year: 30},
        "che_usd2023": {year: 2.5},
    }


def build(data: dict[str, dict[int, float | None]]) -> pl.DataFrame:
    """Run the output builder for the single test year."""
    return expenditure_by_source(ghed(data), first_year=2020, last_year=2020)


def test_output_columns_and_values() -> None:
    out = build(full_year())
    assert out.columns == ["Year", "Country", TOTAL, *SOURCE_ORDER]
    assert out.row(0) == (
        2020,
        "DR Congo",
        2_500_000.0,
        20.0,
        40.0,
        pytest.approx(10.0),
        30.0,
    )


def test_other_private_share() -> None:
    # (6 + 5 + 3 - 4) / (60 + 10 + 30) * 100
    out = build(full_year())
    assert out[PVT].item() == pytest.approx(10.0)


def test_sum_components_fills_partial_years_and_drops_empty_ones() -> None:
    df = ghed({"fs4": {2019: None, 2020: 5}, "fs5": {2019: None, 2020: None}})
    assert sum_components(df, add=["fs4", "fs5"]).rows() == [(2020, 5.0)]


def test_shares_not_summing_to_100_raise() -> None:
    data = full_year()
    data["ext_che"] = {2020: 50}
    with pytest.raises(ValueError, match="do not sum to 100"):
        build(data)


def test_absent_private_components_fail_share_check() -> None:
    data = {
        k: v for k, v in full_year().items() if not k.startswith(("fs4", "fs5", "fs6"))
    }
    with pytest.raises(ValueError, match="missing a source"):
        build(data)


def test_load_ghed_raises_on_missing_code(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = ghed(full_year()).filter(pl.col("indicator_code") != "ext_che")

    class FakeGHED:
        def get_data(self) -> object:
            return rows.with_columns(
                iso3_code=pl.lit("COD"), unit=pl.lit("Millions")
            ).to_pandas()

    monkeypatch.setattr(drc_health_expenditure, "GHED", FakeGHED)
    with pytest.raises(ValueError, match="ext_che"):
        drc_health_expenditure.load_ghed()


def test_missing_year_in_range_raises() -> None:
    with pytest.raises(ValueError, match=r"No che_usd2023 for \[2021\]"):
        expenditure_by_source(ghed(full_year()), first_year=2020, last_year=2021)


def test_years_outside_range_are_dropped() -> None:
    data = full_year(2020)
    for code, by_year in full_year(2012).items():
        data[code] = {**data[code], **by_year}
    out = expenditure_by_source(ghed(data), first_year=2020, last_year=2020)
    assert out["Year"].to_list() == [2020]


def test_load_ghed_raises_on_unexpected_usd_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = ghed(full_year()).with_columns(iso3_code=pl.lit("COD"), unit=pl.lit("Units"))

    class FakeGHED:
        def get_data(self) -> object:
            return rows.to_pandas()

    monkeypatch.setattr(drc_health_expenditure, "GHED", FakeGHED)
    with pytest.raises(ValueError, match="Expected che_usd2023 in Millions"):
        drc_health_expenditure.load_ghed()
