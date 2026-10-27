"""Tests for the WHO Bundibugyo table parser."""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_who import parse_value, parse_who_table, upsert

FIXTURE = Path(__file__).parent / "fixtures" / "who_bundibugyo_2026-10-20.html"
RETRIEVED_AT = datetime(2026, 10, 20, 12, tzinfo=UTC)


@pytest.fixture
def table() -> pl.DataFrame:
    """Parsed fixture table."""
    return parse_who_table(FIXTURE.read_text(), retrieved_at=RETRIEVED_AT)


def _value(df: pl.DataFrame, country: str, indicator: str) -> float | None:
    """Look up one value."""
    row = df.filter(pl.col("country") == country, pl.col("indicator") == indicator)
    assert row.height == 1
    return row["value"].item()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("9 552", 9552.0), ("9\xa0552", 9552.0), ("48%", 48.0), ("-", None), ("0", 0.0)],
)
def test_parse_value(text: str, expected: float | None) -> None:
    assert parse_value(text) == expected


def test_parse_value_rejects_text() -> None:
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_value("n/a")


def test_countries_and_dates(table: pl.DataFrame) -> None:
    dates = dict(table.select("country", "date").unique().iter_rows())
    assert dates == {
        "Democratic Republic of the Congo": date(2026, 10, 18),
        "Uganda": date(2026, 7, 21),
        "France": date(2026, 7, 6),
        "Kenya": date(2026, 10, 20),
        "Nigeria": date(2026, 10, 20),
    }


def test_drc_values(table: pl.DataFrame) -> None:
    drc = "Democratic Republic of the Congo"
    assert _value(table, drc, "Confirmed cases") == 9552
    assert _value(table, drc, "Confirmed deaths") == 4602
    assert _value(table, drc, "Suspected cases") == 468
    assert _value(table, drc, "Recovered") == 2550
    assert _value(table, drc, "Probable cases") is None


def test_shape_and_labels(table: pl.DataFrame) -> None:
    assert table.height == 5 * 7
    assert table["outbreak"].unique().to_list() == ["drc_2026"]
    assert "Total" not in table["country"].to_list()


def test_total_mismatch_raises() -> None:
    html = FIXTURE.read_text().replace(">9 552<", ">9 553<", 1)
    with pytest.raises(ValueError, match="Confirmed cases"):
        parse_who_table(html, retrieved_at=RETRIEVED_AT)


def test_missing_section_raises() -> None:
    with pytest.raises(ValueError, match="Bundibugyo"):
        parse_who_table("<html><h2>Other</h2></html>", retrieved_at=RETRIEVED_AT)


def test_upsert_keeps_latest_capture(table: pl.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    upsert(table, path)
    later = table.with_columns(
        retrieved_at=pl.lit(datetime(2026, 10, 21, tzinfo=UTC)),
        value=pl.col("value") + 1,
    )
    merged = upsert(later, path)
    assert merged.height == table.height
    drc = "Democratic Republic of the Congo"
    assert _value(merged, drc, "Confirmed cases") == 9553


def test_upsert_keeps_stored_row_for_unchanged_value(
    table: pl.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "out.csv"
    upsert(table, path)
    before = path.read_bytes()
    recapture = table.with_columns(
        retrieved_at=pl.lit(datetime(2026, 10, 21, tzinfo=UTC))
    )
    upsert(recapture, path)
    assert path.read_bytes() == before


def test_image_only_section_raises() -> None:
    html = (
        "<html><h2>Ebola disease (Bundibugyo virus)</h2>"
        '<img src="/logo.svg">'
        '<img src="https://cdn.who.int/bundibugyo-epi-update.png"><p>*Data source</p>'
        "<h2>Other outbreak</h2><table><tr><td>x</td></tr></table></html>"
    )
    with pytest.raises(
        ValueError, match=r"published as an image: .*bundibugyo-epi-update\.png"
    ):
        parse_who_table(html, retrieved_at=RETRIEVED_AT)


def test_non_strict_total_mismatch_warns(caplog: pytest.LogCaptureFixture) -> None:
    html = FIXTURE.read_text().replace(">2 550<", ">2 560<", 1)
    df = parse_who_table(html, retrieved_at=RETRIEVED_AT)
    assert _value(df, "Democratic Republic of the Congo", "Recovered") == 2560
    assert "Recovered: countries sum to" in caplog.text


def test_unrelated_image_is_not_reported_as_table() -> None:
    html = '<html><h2>Ebola disease (Bundibugyo virus)</h2><img src="/logo.svg"></html>'
    with pytest.raises(ValueError, match="No table after heading"):
        parse_who_table(html, retrieved_at=RETRIEVED_AT)
