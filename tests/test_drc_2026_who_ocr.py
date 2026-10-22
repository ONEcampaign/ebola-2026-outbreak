"""Tests for rebuilding the WHO table from OCR tokens."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_who import parse_rows
from ebola_2026_outbreak.drc_2026_who_ocr import (
    OCR_STRICT_INDICATORS,
    OcrToken,
    run_ocr,
    tokens_to_rows,
)

FIXTURE = Path(__file__).parent / "fixtures" / "who_bundibugyo_2026-08-01.png"
RETRIEVED_AT = datetime(2026, 8, 2, tzinfo=UTC)
# Left edges of the indicator columns in the synthetic tables below.
COLS = [300, 400, 500, 600, 700, 800, 900]


def tok(x: float, y: float, text: str, score: float = 0.99) -> OcrToken:
    """A token 20 px tall whose box starts at (x, y)."""
    return OcrToken(x, y, x + 10 * len(text), y + 20, text, score)


def header(y: float = 100) -> list[OcrToken]:
    """Two-line header as WHO draws it."""
    first = ["Suspected", "Confirmed", "Confirmed", "Probable", "Probable"]
    first += ["CFR (%)", "Recovered"]
    second = ["cases", "cases", "deaths", "cases", "deaths", "confirmed"]
    return [
        tok(10, y, "Affected countries"),
        *(tok(x, y, t) for x, t in zip(COLS, first, strict=True)),
        *(tok(x, y + 25, t) for x, t in zip(COLS, second, strict=False)),
    ]


def row(y: float, name: list[str], values: list[str | None]) -> list[OcrToken]:
    """One body row; None leaves the cell empty, as OCR does for "-"."""
    names = [tok(10 + 120 * i, y, n) for i, n in enumerate(name)]
    cells = [tok(x, y, v) for x, v in zip(COLS, values, strict=True) if v]
    return names + cells


def standard_table() -> list[OcrToken]:
    """Date rows between countries, footnote markers, a footnote after Total."""
    return [
        tok(900, 20, "World Health"),
        *header(),
        tok(400, 160, "Data as of 30 July 2026"),
        *row(
            190,
            ["Democratic Republic", "*"],
            ["374", "3 605", "1 587", None, None, "44%", "651"],
        ),
        tok(400, 220, "Data as of 6 July 2026"),
        *row(250, ["France", "***"], [None, "1", "0", "0", "0", None, "1"]),
        *row(280, ["Total"], ["374", "3 606", "1 587", "0", "0", "44%", "652"]),
        tok(10, 320, "#Data source: COUSP-DRC"),
    ]


def test_tokens_to_rows_standard_layout() -> None:
    rows = tokens_to_rows(standard_table())
    assert rows[0][1:] == [
        "Suspected cases",
        "Confirmed cases",
        "Confirmed deaths",
        "Probable cases",
        "Probable deaths",
        "CFR (%) confirmed",
        "Recovered",
    ]
    assert rows[1] == ["Data as of 30 July 2026"]
    assert rows[2] == [
        "Democratic Republic",
        "374",
        "3 605",
        "1 587",
        "-",
        "-",
        "44%",
        "651",
    ]
    assert rows[4][0] == "France"
    assert rows[-1][0] == "Total"  # The footnote line is not a row.


def test_parsed_rows_validate_against_total() -> None:
    df = parse_rows(
        tokens_to_rows(standard_table()),
        retrieved_at=RETRIEVED_AT,
        source_url="https://cdn.who.int/x.png",
        strict_indicators=OCR_STRICT_INDICATORS,
    )
    assert df.filter(pl.col("indicator") == "Confirmed cases")["value"].sum() == 3606


def test_date_line_above_header_applies_to_all_rows() -> None:
    tokens = [
        tok(10, 20, "Data as of 26 May 2026"),
        *header(),
        *row(160, ["Uganda"], ["0", "7", "1", None, None, "14%", "0"]),
        *row(190, ["Total"], ["0", "7", "1", None, None, "14%", "0"]),
    ]
    rows = tokens_to_rows(tokens)
    assert rows[1] == ["Data as of 26 May 2026"]
    assert rows[2][0] == "Uganda"


def test_low_confidence_value_raises() -> None:
    tokens = [*header(), tok(400, 160, "Data as of 1 May 2026")]
    tokens += row(190, ["Uganda"], ["0", "7", "1", None, None, "14%", "0"])
    tokens[-1] = tokens[-1]._replace(score=0.5)
    with pytest.raises(ValueError, match="Low OCR confidence"):
        tokens_to_rows(tokens)


def test_low_confidence_dash_and_marker_are_ignored() -> None:
    tokens = [*header(), tok(400, 160, "Data as of 1 May 2026")]
    tokens += row(190, ["Uganda"], ["0", "7", "1", None, None, "14%", "0"])
    tokens += [tok(130, 190, "*木", score=0.5), tok(600, 190, "-", score=0.5)]
    rows = tokens_to_rows(tokens)
    assert rows[2][0] == "Uganda"
    assert rows[2][4] == "-"


def test_header_matches_despite_spacing_and_ocr_typo() -> None:
    rows = tokens_to_rows(standard_table())
    rows[0][3] = "Confrmed deaths"
    rows[0][6] = "CFR(%) confirmed"
    df = parse_rows(
        rows,
        retrieved_at=RETRIEVED_AT,
        source_url="https://cdn.who.int/x.png",
        strict_indicators=OCR_STRICT_INDICATORS,
    )
    assert {"Confirmed deaths", "CFR confirmed (%)"} <= set(df["indicator"])


def test_real_image() -> None:
    pytest.importorskip("rapidocr")
    df = parse_rows(
        tokens_to_rows(run_ocr(FIXTURE.read_bytes())),
        retrieved_at=RETRIEVED_AT,
        source_url="https://cdn.who.int/x.png",
        strict_indicators=OCR_STRICT_INDICATORS,
    )
    drc = df.filter(pl.col("country") == "Democratic Republic of the Congo")
    values = dict(drc.select("indicator", "value").iter_rows())
    assert values["Confirmed cases"] == 3605
    assert values["Confirmed deaths"] == 1587
    assert values["Probable cases"] is None
    assert set(df["country"]) == {
        "Democratic Republic of the Congo",
        "Uganda",
        "France",
    }


def test_date_row_without_space() -> None:
    rows = tokens_to_rows(standard_table())
    rows[1] = ["Data as of30 July 2026"]
    df = parse_rows(
        rows,
        retrieved_at=RETRIEVED_AT,
        source_url="https://cdn.who.int/x.png",
        strict_indicators=OCR_STRICT_INDICATORS,
    )
    drc = df.filter(pl.col("country") == "Democratic Republic")
    assert drc["date"].unique().to_list() == [datetime(2026, 7, 30).date()]
