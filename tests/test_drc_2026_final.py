"""Tests for the DRC national series."""

import logging
from datetime import date

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_final import (
    FINAL_INDICATOR,
    build_final,
    cross_check_who,
    sum_by_date_indicator,
)

D1, D2 = date(2026, 10, 17), date(2026, 10, 18)
CASES, NEW = "Cumulative confirmed cases", "New confirmed cases (24h)"


@pytest.fixture
def provinces() -> pl.DataFrame:
    """Two provinces over two dates, with one missing new-case count."""
    return pl.DataFrame(
        {
            "date": [D1, D1, D2, D2, D1, D1],
            "province": ["Ituri", "Tshopo", "Ituri", "Tshopo", "Ituri", "Tshopo"],
            "indicator": [CASES, CASES, CASES, CASES, NEW, NEW],
            "value": [100, 10, 120, 11, 5, None],
        }
    )


def test_sum_by_date_indicator(provinces: pl.DataFrame) -> None:
    out = sum_by_date_indicator(provinces)
    assert out.filter(pl.col("indicator") == CASES)["value"].to_list() == [110, 131]
    # A missing province count makes the national new-case sum unknown.
    assert out.filter(pl.col("indicator") == NEW)["value"].to_list() == [None]
    assert out["outbreak"].unique().to_list() == ["drc_2026"]


def test_build_final_relabels(provinces: pl.DataFrame) -> None:
    final = build_final(sum_by_date_indicator(provinces))
    assert final["indicator"].unique().to_list() == [FINAL_INDICATOR]
    assert final.columns == ["outbreak", "date", "indicator", "value"]


def test_cross_check_logs_mismatch(
    provinces: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    final = build_final(sum_by_date_indicator(provinces))
    who = pl.DataFrame(
        {
            "date": [D1, D2],
            "country": ["Democratic Republic of the Congo"] * 2,
            "indicator": ["Confirmed cases"] * 2,
            "value": [110.0, 130.0],
        }
    )
    with caplog.at_level(logging.INFO):
        cross_check_who(final, who)
    assert "2 shared dates, 1 match" in caplog.text
    assert "2026-10-18: sitrep total 131, WHO 130.0" in caplog.text
