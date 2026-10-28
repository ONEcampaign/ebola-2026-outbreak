"""Tests for the DRC national series."""

import logging
from datetime import date

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_final import (
    FINAL_INDICATOR,
    SOURCE_INRB,
    SOURCE_SITREP,
    backfill_from_inrb,
    build_final,
    cross_check_inrb,
    cross_check_who,
    sum_by_date_indicator,
)

D0, D1, D2 = date(2026, 10, 16), date(2026, 10, 17), date(2026, 10, 18)
DEATHS = "Cumulative confirmed deaths"
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


@pytest.fixture
def inrb() -> pl.DataFrame:
    """INRB rows on one date the sitreps miss and one they share."""
    return pl.DataFrame(
        {
            "outbreak": ["drc_2026"] * 4,
            "date": [D0, D0, D0, D2],
            "indicator": [CASES, DEATHS, NEW, CASES],
            "value": [100, 40, None, 999],
            "source_ref": ["abc"] * 4,
        }
    )


def test_backfill_from_inrb(provinces: pl.DataFrame, inrb: pl.DataFrame) -> None:
    intermediate = sum_by_date_indicator(provinces)
    out = backfill_from_inrb(intermediate=intermediate, inrb=inrb)
    added = out.filter(pl.col("source") == SOURCE_INRB)
    # Only the missing date is added, and its null row is dropped.
    assert added.select("date", "indicator", "value").rows() == [
        (D0, CASES, 100),
        (D0, DEATHS, 40),
    ]
    # The sitrep value wins on the shared date.
    shared = out.filter(pl.col("date") == D2, pl.col("indicator") == CASES)
    assert shared.select("value", "source").rows() == [(131, SOURCE_SITREP)]
    assert out.height == intermediate.height + 2


def test_build_final_after_backfill(
    provinces: pl.DataFrame, inrb: pl.DataFrame
) -> None:
    merged = backfill_from_inrb(
        intermediate=sum_by_date_indicator(provinces), inrb=inrb
    )
    final = build_final(merged)
    assert final.columns == ["outbreak", "date", "indicator", "value"]
    assert final["value"].to_list() == [100, 110, 131]


def test_cross_check_inrb_logs_mismatch(
    provinces: pl.DataFrame, inrb: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        cross_check_inrb(intermediate=sum_by_date_indicator(provinces), inrb=inrb)
    assert "1 shared rows, 0 match" in caplog.text
    assert "2026-10-18 cumulative confirmed cases: sitrep 131, INRB 999" in caplog.text


def test_backfill_from_inrb_logs_date_missing_an_indicator(
    provinces: pl.DataFrame, inrb: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        backfill_from_inrb(intermediate=sum_by_date_indicator(provinces), inrb=inrb)
    assert f"INRB backfill on 2026-10-16 has no ['{NEW}']" in caplog.text


def test_backfill_from_inrb_rejects_duplicate_date_indicator(
    provinces: pl.DataFrame, inrb: pl.DataFrame
) -> None:
    doubled = pl.concat([inrb, inrb.filter(pl.col("date") == D0)])
    with pytest.raises(ValueError, match="Duplicate \\(date, indicator\\)"):
        backfill_from_inrb(intermediate=sum_by_date_indicator(provinces), inrb=doubled)


def test_cross_check_who_raises_when_no_rows_match(provinces: pl.DataFrame) -> None:
    final = build_final(sum_by_date_indicator(provinces))
    who = pl.DataFrame(
        {
            "date": [D1],
            "country": ["DR Congo"],
            "indicator": ["Confirmed cases"],
            "value": [110.0],
        }
    )
    with pytest.raises(ValueError, match="No WHO rows"):
        cross_check_who(final, who)
