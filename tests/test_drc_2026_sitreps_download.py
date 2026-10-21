"""Tests for sitrep header parsing and report date resolution."""

from datetime import date

import polars as pl
import pytest

from ebola_2026_outbreak.drc_2026_sitreps_download import (
    MANIFEST_SCHEMA,
    SitrepHeader,
    flag_latest,
    parse_header,
    resolve_report_date,
)

# First-page header lines as extracted from real sitreps, one per template.
HEADER_OCT = """SitRep N°157 /MVE-BDBV/18/10/2026
Date du rapport : 18 octobre 2026 Date de publication : 19 octobre 2026"""
HEADER_AUG = """SitRep N°106/MVEBDB/29/08/2026
Date de rapportage : 29 août 2026 Date de publication : 30 août 2026"""
HEADER_JUN = """SitRep N°32/MVB_15/06/2026
Date de rapportage 15 juin 2026
Date de publication 16 juin 2026"""
HEADER_MAY = """SitRep MVE N° 007/MVB_17/2026
Date de rapportage 21 mai 2026
Date de publication 22 mai 2026"""
HEADER_1ER = """SitRep N°139/MVE-BDBV/30/09/2026
Date du rapport : 30 septembre 2026 Date de publication : 1er octobre 2026"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            HEADER_OCT,
            SitrepHeader(
                157, date(2026, 10, 18), date(2026, 10, 18), date(2026, 10, 19)
            ),
        ),
        (
            HEADER_AUG,
            SitrepHeader(106, date(2026, 8, 29), date(2026, 8, 29), date(2026, 8, 30)),
        ),
        (
            HEADER_JUN,
            SitrepHeader(32, date(2026, 6, 15), date(2026, 6, 15), date(2026, 6, 16)),
        ),
        (HEADER_MAY, SitrepHeader(7, date(2026, 5, 21), None, date(2026, 5, 22))),
        (
            HEADER_1ER,
            SitrepHeader(139, date(2026, 9, 30), date(2026, 9, 30), date(2026, 10, 1)),
        ),
        ("no header here", SitrepHeader(None, None, None, None)),
    ],
)
def test_parse_header(text: str, expected: SitrepHeader) -> None:
    assert parse_header(text) == expected


ANCHORS = [(149, date(2026, 10, 10)), (151, date(2026, 10, 12))]


def test_resolve_agreeing_dates() -> None:
    header = SitrepHeader(157, date(2026, 10, 18), date(2026, 10, 18), None)
    assert resolve_report_date(header, anchors=[]) == date(2026, 10, 18)


def test_resolve_single_date() -> None:
    header = SitrepHeader(7, date(2026, 5, 21), None, None)
    assert resolve_report_date(header, anchors=[]) == date(2026, 5, 21)


def test_resolve_conflict_by_nearest_anchor() -> None:
    # Sitrep 150 prints 12 Oct, but #149 is 10 Oct and #151 is 12 Oct.
    header = SitrepHeader(150, date(2026, 10, 12), date(2026, 10, 11), None)
    assert resolve_report_date(header, anchors=ANCHORS) == date(2026, 10, 11)


def test_resolve_conflict_neither_fits() -> None:
    header = SitrepHeader(150, date(2026, 10, 20), date(2026, 10, 21), None)
    with pytest.raises(ValueError, match="Neither header date fits"):
        resolve_report_date(header, anchors=ANCHORS)


def test_resolve_no_dates() -> None:
    with pytest.raises(ValueError, match="No date"):
        resolve_report_date(SitrepHeader(None, None, None, None), anchors=ANCHORS)


def test_flag_latest_keeps_most_recent_upload() -> None:
    day = date(2026, 8, 9)
    rows = [
        {"report_date": day, "media_id": 497, "modified": "2026-08-14T10:00:00"},
        {"report_date": day, "media_id": 518, "modified": "2026-08-12T10:00:00"},
        {"report_date": date(2026, 8, 10), "media_id": 506, "modified": "2026-08-12"},
    ]
    manifest = pl.DataFrame(rows).with_columns(file_name=pl.lit("f.pdf"))
    manifest = manifest.select(
        pl.col(c) if c in manifest.columns else pl.lit(None).alias(c)
        for c in MANIFEST_SCHEMA
    ).cast(MANIFEST_SCHEMA)
    flagged = flag_latest(manifest)
    assert dict(flagged.select("media_id", "is_latest").iter_rows()) == {
        497: True,
        518: False,
        506: True,
    }
