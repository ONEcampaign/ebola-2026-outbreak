"""Tests for the ROWCA early West Africa series."""

from datetime import date

import polars as pl
import pytest

from ebola_2026_outbreak.west_africa_2014_16_early import (
    deduplicate,
    drop_entry_errors,
    early_series,
)

D1, D2, D3 = date(2014, 8, 26), date(2014, 8, 27), date(2014, 8, 28)


def rows(data: list[tuple[str, date, float]]) -> pl.DataFrame:
    """ROWCA-shaped rows from (country, date, value) tuples."""
    return pl.DataFrame(
        data,
        schema={"country": pl.String, "date": pl.Date, "value": pl.Float64},
        orient="row",
    )


def test_deduplicate_keeps_agreeing_repeats() -> None:
    out = deduplicate(rows([("Guinea", D1, 10), ("Guinea", D1, 10)]))
    assert out.height == 1


def test_deduplicate_raises_on_conflict() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate(rows([("Guinea", D1, 10), ("Guinea", D1, 11)]))


def test_drop_entry_errors_drops_large_dips_only() -> None:
    out = drop_entry_errors(
        rows(
            [
                ("Liberia", D1, 342),
                ("Liberia", D2, 22),  # entry error, dropped
                ("Liberia", D3, 392),
                ("Guinea", D1, 208),
                ("Guinea", D2, 207),  # reclassification, kept
            ]
        )
    )
    assert sorted(out.select("country", "value").rows()) == [
        ("Guinea", 207.0),
        ("Guinea", 208.0),
        ("Liberia", 342.0),
        ("Liberia", 392.0),
    ]


def test_early_series_sums_with_carry_forward() -> None:
    out = early_series(
        rows([("Guinea", D1, 100), ("Liberia", D1, 50), ("Guinea", D2, 110)])
    )
    assert out.select("date", "value").rows() == [(D1, 150.0), (D2, 160.0)]
    assert out["source"].unique().to_list() == ["OCHA ROWCA"]
    assert out["indicator"].unique().to_list() == [
        "Cumulative number of confirmed Ebola cases"
    ]
