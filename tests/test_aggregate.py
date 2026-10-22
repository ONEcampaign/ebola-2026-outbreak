"""Tests for shared aggregation helpers."""

from datetime import date

import polars as pl

from ebola_2026_outbreak.aggregate import forward_fill_cumulative

D1, D2, D3 = date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)


def test_forward_fill_carries_last_cumulative_value() -> None:
    df = pl.DataFrame(
        {
            "date": [D1, D2, D3, D2, D1],
            "province": ["A", "A", "A", "B", "A"],
            "indicator": ["Cumulative x"] * 4 + ["New x"],
            "value": [1, 2, 3, 10, 5],
        }
    )
    out = forward_fill_cumulative(df, entity="province", cumulative_prefix="Cumulative")
    b = out.filter(pl.col("province") == "B").sort("date")
    # B starts on D2: nothing before, carried onto D3.
    assert b.select("date", "value").rows() == [(D2, 10), (D3, 10)]
    # Non-cumulative indicators are left as reported.
    assert out.filter(pl.col("indicator") == "New x").height == 1
