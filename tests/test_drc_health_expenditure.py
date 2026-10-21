"""Tests for DR Congo expenditure by source."""

import polars as pl
import pytest

from ebola_2026_outbreak import drc_health_expenditure
from ebola_2026_outbreak.drc_health_expenditure import (
    OUTPUT_COLUMNS,
    PVT,
    SHARE_UNIT,
    USD_PER_GHED_UNIT,
    USD_UNIT,
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
        "hf3": {year: 30},
        "hf2": {year: 10},
        "gghed_usd2023": {year: 2},
        "ext_usd2023": {year: 4},
        "hf3_usd2023": {year: 3},
        "fs4_usd2023": {year: 0.6},
        "fs5_usd2023": {year: 0.5},
        "fs6_usd2023": {year: 0.3},
        "fs61_usd2023": {year: 0.4},
    }


def value(df: pl.DataFrame, *, unit: str, source: str) -> float:
    """Look up one output value."""
    return df.filter(pl.col("unit") == unit, pl.col("source") == source)["value"].item()


def test_other_private_share_and_usd() -> None:
    out = expenditure_by_source(ghed(full_year()))
    # (6 + 5 + 3 - 4) / (60 + 10 + 30) * 100
    assert value(out, unit=SHARE_UNIT, source=PVT) == pytest.approx(10.0)
    # (0.6 + 0.5 + 0.3 - 0.4) million
    assert value(out, unit=USD_UNIT, source=PVT) == pytest.approx(
        1.0 * USD_PER_GHED_UNIT
    )


def test_usd_is_converted_from_millions() -> None:
    out = expenditure_by_source(ghed(full_year()))
    assert value(out, unit=USD_UNIT, source="External") == 4 * USD_PER_GHED_UNIT


def test_output_layout_matches_original() -> None:
    out = expenditure_by_source(ghed(full_year()))
    assert out.columns == OUTPUT_COLUMNS
    assert out.height == 8
    assert out["entity_name"].unique().to_list() == ["DR Congo"]
    assert out["unit"].unique(maintain_order=True).to_list() == [SHARE_UNIT, USD_UNIT]


def test_sum_components_fills_partial_years_and_drops_empty_ones() -> None:
    df = ghed({"fs4": {2019: None, 2020: 5}, "fs5": {2019: None, 2020: None}})
    out = sum_components(df, add=["fs4", "fs5"])
    assert out.rows() == [(2020, 5.0)]


def test_shares_not_summing_to_100_raise() -> None:
    data = full_year()
    data["ext_che"] = {2020: 50}
    with pytest.raises(ValueError, match="do not sum to 100"):
        expenditure_by_source(ghed(data))


def stub_ghed(monkeypatch: pytest.MonkeyPatch, rows: pl.DataFrame) -> None:
    """Make `GHED().get_data()` return `rows` (with iso3_code) as pandas."""

    class FakeGHED:
        def get_data(self) -> object:
            return rows.to_pandas()

    monkeypatch.setattr(drc_health_expenditure, "GHED", FakeGHED)


def full_rows(*, usd_unit: str = "Millions") -> pl.DataFrame:
    """GHED-shaped rows for COD with every required code."""
    return ghed(full_year()).with_columns(
        iso3_code=pl.lit("COD"),
        unit=pl.when(pl.col("indicator_code").str.ends_with("usd2023"))
        .then(pl.lit(usd_unit))
        .otherwise(pl.lit("Percentage")),
    )


def test_load_ghed_raises_on_missing_code(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_ghed(
        monkeypatch, full_rows().filter(pl.col("indicator_code") != "ext_usd2023")
    )
    with pytest.raises(ValueError, match="ext_usd2023"):
        drc_health_expenditure.load_ghed()


def test_load_ghed_raises_on_unexpected_usd_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_ghed(monkeypatch, full_rows(usd_unit="Units"))
    with pytest.raises(ValueError, match="Expected USD indicators in Millions"):
        drc_health_expenditure.load_ghed()


def test_absent_private_components_fail_share_check() -> None:
    data = {
        k: v for k, v in full_year().items() if not k.startswith(("fs4", "fs5", "fs6"))
    }
    with pytest.raises(ValueError, match="missing a source"):
        expenditure_by_source(ghed(data))
