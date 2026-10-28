"""Tests for the INRB national series."""

import logging
from datetime import date

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_inrb import (
    INRB_REF,
    SCHEMA,
    parse_national,
)
from ebola_2026_outbreak.drc_2026_sitreps_parse import check_monotonic

METRIC = "national_cumulative_confirmed_cases"
CASES = "Cumulative confirmed cases"


def raw(*, noms: list[str], dates: list[str], values: list[str]) -> pl.DataFrame:
    """An INRB CSV as read with every column as a string."""
    return pl.DataFrame({"nom": noms, "date": dates, METRIC: values})


def test_parse_national() -> None:
    out = parse_national(
        raw=raw(
            noms=["DRC"] * 3,
            dates=["2026-05-18", "2026-05-14", "2026-05-17"],
            values=["33", "8", "ND"],
        ),
        metric=METRIC,
        indicator=CASES,
    )
    assert out.schema == SCHEMA
    assert out["date"].to_list() == [
        date(2026, 5, 14),
        date(2026, 5, 17),
        date(2026, 5, 18),
    ]
    assert out["value"].to_list() == [8, None, 33]
    assert out["source_ref"].unique().to_list() == [INRB_REF]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (raw(noms=["Bunia"], dates=["2026-05-14"], values=["8"]), "non-national"),
        (raw(noms=["DRC"], dates=["2026-06-25]"], values=["8"]), "invalid dates"),
        (
            raw(noms=["DRC"] * 2, dates=["2026-05-14"] * 2, values=["8", "9"]),
            "duplicate dates",
        ),
        (raw(noms=["DRC"], dates=["2026-05-14"], values=["8.5"]), "non-integer"),
    ],
)
def test_parse_national_rejects(frame: pl.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_national(raw=frame, metric=METRIC, indicator=CASES)


def test_parse_national_rejects_columns() -> None:
    frame = raw(noms=["DRC"], dates=["2026-05-14"], values=["8"]).rename(
        {METRIC: "other"}
    )
    with pytest.raises(ValueError, match="columns"):
        parse_national(raw=frame, metric=METRIC, indicator=CASES)


def test_check_monotonic_logs_fall(caplog: pytest.LogCaptureFixture) -> None:
    df = parse_national(
        raw=raw(
            noms=["DRC"] * 2, dates=["2026-05-14", "2026-05-15"], values=["9", "8"]
        ),
        metric=METRIC,
        indicator=CASES,
    )
    with caplog.at_level(logging.WARNING):
        check_monotonic(df, by=(), label="INRB")
    assert (
        "INRB cumulative confirmed cases falls from 9 (2026-05-14) to 8 (2026-05-15)"
        in caplog.text
    )
