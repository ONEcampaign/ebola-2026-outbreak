"""Tests for sitrep header parsing and report date resolution."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from ebola_2026_outbreak import drc_2026_sitreps_download as dlm
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


def test_resolve_conflict_tied_anchors_disagree() -> None:
    # #149 and #151 are equally near #150 but expect 11 and 12 Oct.
    anchors = [(149, date(2026, 10, 10)), (151, date(2026, 10, 13))]
    header = SitrepHeader(150, date(2026, 10, 12), date(2026, 10, 11), None)
    with pytest.raises(ValueError, match="expect different dates"):
        resolve_report_date(header, anchors=anchors)


def test_resolve_conflict_tied_anchors_agree() -> None:
    header = SitrepHeader(150, date(2026, 10, 12), date(2026, 10, 11), None)
    anchors = [*ANCHORS, (149, date(2026, 10, 10))]
    assert resolve_report_date(header, anchors=anchors) == date(2026, 10, 11)


def test_resolve_report_dates_keeps_stored_dates() -> None:
    # #150 was resolved to 11 Oct against #149. A new upload also numbered 150
    # with agreeing dates of 12 Oct is a nearer anchor; re-resolving the old
    # row would flip it to 12 Oct.
    rows = [
        (149, date(2026, 10, 10), date(2026, 10, 10), date(2026, 10, 10)),
        (150, date(2026, 10, 11), date(2026, 10, 12), date(2026, 10, 11)),
        (150, None, date(2026, 10, 12), date(2026, 10, 12)),
    ]
    manifest = pl.DataFrame(
        rows,
        schema=[
            "sitrep_number",
            "report_date",
            "printed_report_date",
            "reference_date",
        ],
        orient="row",
    )
    manifest = manifest.select(
        pl.col(c) if c in manifest.columns else pl.lit(None).alias(c)
        for c in MANIFEST_SCHEMA
    ).cast(MANIFEST_SCHEMA)
    resolved = dlm.resolve_report_dates(manifest)
    assert resolved["report_date"].to_list() == [
        date(2026, 10, 10),
        date(2026, 10, 11),
        date(2026, 10, 12),
    ]


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


def test_flag_latest_prefers_ministry_over_newer_insp() -> None:
    day = date(2026, 10, 19)
    rows = [
        {
            "report_date": day,
            "media_id": 712,
            "modified": "2026-10-27T09:37:50",
            "source_url": f"{dlm.MINISTRY_SITE}wp-content/uploads/a.pdf",
        },
        {
            "report_date": day,
            "media_id": 25749,
            "modified": "2026-10-28T05:05:17",
            "source_url": "https://insp.cd/wp-content/uploads/a.pdf",
        },
        {
            "report_date": date(2026, 10, 25),
            "media_id": 25776,
            "modified": "2026-10-27T08:09:30",
            "source_url": "https://insp.cd/wp-content/uploads/b.pdf",
        },
    ]
    manifest = pl.DataFrame(rows).with_columns(file_name=pl.lit("f.pdf"))
    manifest = manifest.select(
        pl.col(c) if c in manifest.columns else pl.lit(None).alias(c)
        for c in MANIFEST_SCHEMA
    ).cast(MANIFEST_SCHEMA)
    flagged = flag_latest(manifest)
    assert dict(flagged.select("media_id", "is_latest").iter_rows()) == {
        712: True,
        25749: False,
        25776: True,
    }


def test_list_media_keeps_marked_files_whatever_their_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        {
            **_item(25776, "2026-10-27"),
            "source_url": "https://insp.cd/SitRep_MVEBDB_164.pdf",
        },
        {
            **_item(24000, "2025-10-16"),
            "source_url": "https://insp.cd/SitRep_MVE_Kasai.pdf",
        },
        {
            **_item(25812, "2026-11-02"),
            "source_url": "https://insp.cd/final_draft_SitRep_MVE_RDC_N_170_31-10-2026.pdf",
        },
        # Unattached, from before the MVEBDB names.
        {
            **_item(24728, "2026-05-21"),
            "source_url": "https://insp.cd/SitRep_MVE_RDC_20260521_editing-FV.pdf",
        },
        # Unattached, one of INSP's first sitreps.
        {
            **_item(24706, "2026-05-20"),
            "source_url": "https://insp.cd/Sitrep_MVE-Ituri_N°-002_18-mai-2026.pdf",
        },
        # The 2025 outbreak's sitreps.
        {
            **_item(24111, "2025-10-16"),
            "source_url": "https://insp.cd/SITREP40_MVE16-B.pdf",
        },
    ]
    monkeypatch.setattr(dlm, "list_api_pdfs", lambda api: items)
    kept = [item["id"] for item in dlm.list_media([])]
    assert kept == [25776, 25812, 24728, 24706]


def test_list_media_keeps_renamed_upload_attached_to_sitrep_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        {
            **_item(26108, "2026-11-10"),
            "source_url": "https://insp.cd/MVE_RDC_N178.pdf",
        },
        {
            **_item(26090, "2026-11-06"),
            "source_url": "https://insp.cd/SGI_Rougeole.pdf",
        },
        {
            **_item(26131, "2026-11-10"),
            "source_url": "https://insp.cd/SitRep_N179_09-11-2026_corrige.pdf",
            "post": 26130,
        },
        # Attached with a low id and no marker.
        {
            **_item(25321, "2026-08-13"),
            "source_url": "https://insp.cd/85.pdf",
            "post": 25302,
        },
        # Attached to a post outside the SitRep category.
        {
            **_item(25230, "2026-07-31"),
            "source_url": "https://insp.cd/Bulletin-Ebola.pdf",
            "post": 25229,
        },
    ]
    monkeypatch.setattr(dlm, "list_api_pdfs", lambda api: items)
    posts = [
        _post(26130, 179, "2026-11-10T21:52:09"),
        _post(25302, 85, "2026-08-09T18:36:00"),
    ]
    assert [item["id"] for item in dlm.list_media(posts)] == [26108, 26131, 25321]


def test_list_media_warns_about_newer_rejected_upload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    items = [
        {
            **_item(25776, "2026-10-27"),
            "source_url": "https://insp.cd/SitRep_MVEBDB_164.pdf",
        },
        {
            **_item(24000, "2025-10-16"),
            "source_url": "https://insp.cd/SitRep_MVE_Kasai.pdf",
        },
        {
            **_item(25900, "2026-11-10"),
            "source_url": "https://insp.cd/sitrep_Ebola_RDC_N_180.pdf",
        },
        {
            **_item(25950, "2026-11-11"),
            "source_url": "https://insp.cd/Communique_Presse.pdf",
        },
    ]
    monkeypatch.setattr(dlm, "list_api_pdfs", lambda api: items)
    with caplog.at_level(logging.WARNING):
        dlm.list_media([])
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "sitrep_Ebola_RDC_N_180.pdf" in warnings[0]


@pytest.mark.parametrize(
    ("title", "number"),
    [
        ("SitRep N°179/MVE-BDBV/09/11/2026", 179),
        ("SitRep N°0100/MVEBDB/22/08/2026", 100),
        (" SITREP N°059 ", 59),
        ("SitRep MVE N° 014/2026", 14),
        ("SitRep N&#176;180/MVE-BDBV/10/11/2026", 180),
    ],
)
def test_parse_post_reads_sitrep_number(title: str, number: int) -> None:
    raw = {"id": 1, "date_gmt": "2026-11-10T21:52:09", "title": {"rendered": title}}
    post = dlm.parse_post(raw)
    assert post is not None
    assert post.sitrep_number == number
    assert post.published == datetime(2026, 11, 10, 21, 52, 9, tzinfo=UTC)


def test_parse_post_skips_title_without_number() -> None:
    raw = {"id": 1, "date_gmt": "2026-11-10T21:52:09", "title": {"rendered": "Note"}}
    assert dlm.parse_post(raw) is None


def _post(post_id: int, number: int, published: str) -> dlm.SitrepPost:
    """SitRep post stub; `published` is a UTC ISO timestamp."""
    return dlm.SitrepPost(
        post_id,
        number,
        f"SitRep N°{number}",
        datetime.fromisoformat(published).replace(tzinfo=UTC),
    )


NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def _attached(media_id: int, post_id: int) -> dlm.MediaItem:
    """Media item stub attached to a post."""
    return {**_item(media_id, "2026-11-10"), "post": post_id}


def test_find_posts_without_pdf_flags_gap_below_newest_number() -> None:
    posts = [
        _post(26097, 177, "2026-11-08T08:00"),
        _post(26107, 178, "2026-11-10T07:57"),
        _post(26130, 179, "2026-11-10T21:52"),
    ]
    media = [_attached(26098, 26097), _attached(26108, 26107), _attached(26131, 26130)]
    missing = dlm.find_posts_without_pdf(
        posts, media, manifest_ids={26098, 26131}, now=NOW
    )
    assert [post.post_id for post in missing] == [26107]


def test_find_posts_without_pdf_passes_when_one_attached_pdf_is_kept() -> None:
    posts = [_post(25296, 83, "2026-08-09T18:31")]
    # 25297 returned 404. Its re-upload 25325 is in the manifest.
    media = [_attached(25297, 25296), _attached(25325, 25296)]
    assert dlm.find_posts_without_pdf(posts, media, manifest_ids={25325}, now=NOW) == []


def test_find_posts_without_pdf_skips_exempt_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dlm, "POSTS_WITHOUT_PDF", {24781: "no file"})
    posts = [_post(24781, 1, "2026-05-20T10:00"), _post(26130, 179, "2026-11-10")]
    missing = dlm.find_posts_without_pdf(
        posts, [_attached(26131, 26130)], manifest_ids={26131}, now=NOW
    )
    assert missing == []


def test_find_posts_without_pdf_warns_when_exempt_post_now_has_pdf(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dlm, "POSTS_WITHOUT_PDF", {24781: "no file"})
    posts = [_post(24781, 1, "2026-05-20T10:00"), _post(26130, 179, "2026-11-10")]
    media = [_attached(24659, 24781), _attached(26131, 26130)]
    with caplog.at_level(logging.WARNING):
        missing = dlm.find_posts_without_pdf(
            posts, media, manifest_ids={24659, 26131}, now=NOW
        )
    assert missing == []
    assert "POSTS_WITHOUT_PDF lists post 24781" in caplog.text
    assert "26130" not in caplog.text


def test_find_posts_without_pdf_ignores_post_within_grace() -> None:
    posts = [
        _post(26130, 179, "2026-11-10T21:52"),
        _post(26140, 180, "2026-11-11T11:00"),
    ]
    missing = dlm.find_posts_without_pdf(
        posts, [_attached(26131, 26130)], manifest_ids={26131}, now=NOW
    )
    assert missing == []


def test_find_posts_without_pdf_rejects_no_settled_post() -> None:
    posts = [_post(26140, 180, "2026-11-11T11:00")]
    with pytest.raises(ValueError, match="grace"):
        dlm.find_posts_without_pdf(posts, [], manifest_ids=set(), now=NOW)


def test_find_gaps_flags_missing_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dlm, "KNOWN_GAPS", {})
    dates = {date(2026, 7, 15), date(2026, 7, 17)}
    assert dlm.find_gaps(dates) == [date(2026, 7, 16)]


def test_find_gaps_skips_known_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dlm, "KNOWN_GAPS", {date(2026, 7, 28): "none published"})
    dates = {date(2026, 7, 27), date(2026, 7, 29)}
    assert dlm.find_gaps(dates) == []


def test_find_gaps_warns_about_filled_known_gap(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dlm, "KNOWN_GAPS", {date(2026, 7, 28): "none published"})
    with caplog.at_level(logging.WARNING):
        dlm.find_gaps({date(2026, 7, 27), date(2026, 7, 28)})
    assert "2026-07-28" in caplog.text


def _item(media_id: int, modified: str) -> dlm.MediaItem:
    """Media item stub."""
    return {
        "id": media_id,
        "modified": modified,
        "source_url": f"https://example.org/{media_id}.pdf",
        "mime_type": "application/pdf",
        "post": None,
    }


@dataclass
class FakeApi:
    """Media API contents and the downloads requested from it."""

    items: list[dlm.MediaItem] = field(default_factory=list)
    downloads: list[tuple[int, bool]] = field(default_factory=list)


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    """Stub the media API, downloads and PDF reads."""
    api = FakeApi()

    def download(item: dlm.MediaItem, *, refresh: bool = False) -> Path:
        """Record the download request and return a stub path."""
        api.downloads.append((item["id"], refresh))
        return Path(f"{item['id']}.pdf")

    monkeypatch.setattr(dlm, "list_media", lambda posts: api.items)
    monkeypatch.setattr(dlm, "download", download)
    monkeypatch.setattr(dlm, "read_first_page", lambda path: HEADER_OCT)
    return api


def test_build_manifest_keeps_media_dropped_by_api(fake_api: FakeApi) -> None:
    fake_api.items = [_item(710, "2026-10-20")]
    first = dlm.build_manifest(None, posts=[])
    fake_api.items = []
    second = dlm.build_manifest(first, posts=[])
    assert second.select("media_id", "in_api", "is_latest").rows() == [
        (710, False, True)
    ]


def test_build_manifest_refreshes_replaced_upload(fake_api: FakeApi) -> None:
    fake_api.items = [_item(710, "2026-10-20")]
    first = dlm.build_manifest(None, posts=[])
    fake_api.items = [_item(710, "2026-10-21")]
    second = dlm.build_manifest(first, posts=[])
    assert fake_api.downloads == [(710, False), (710, True)]
    assert second["modified"].to_list() == ["2026-10-21"]


def test_build_manifest_reuses_unchanged_media(fake_api: FakeApi) -> None:
    fake_api.items = [_item(710, "2026-10-20")]
    dlm.build_manifest(dlm.build_manifest(None, posts=[]), posts=[])
    assert fake_api.downloads == [(710, False)]


def test_download_creates_missing_sitrep_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sitrep_dir = tmp_path / "data" / "raw" / "sitreps"
    monkeypatch.setattr(dlm, "SITREP_DIR", sitrep_dir)
    monkeypatch.setattr(dlm, "get", lambda url: type("R", (), {"content": b"%PDF"})())
    path = dlm.download(_item(710, "2026-10-20"))
    assert path == sitrep_dir / "710_20261020.pdf"
    assert path.read_bytes() == b"%PDF"


def test_download_names_file_by_modified_so_replaced_upload_is_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dlm, "SITREP_DIR", tmp_path)
    content = iter([b"old", b"new"])
    monkeypatch.setattr(
        dlm, "get", lambda url: type("R", (), {"content": next(content)})()
    )
    old = dlm.download(_item(710, "2026-10-20T08:15:00"))
    new = dlm.download(_item(710, "2026-10-21T09:30:00"))
    assert old != new
    assert (old.read_bytes(), new.read_bytes()) == (b"old", b"new")
    assert dlm.download(_item(710, "2026-10-21T09:30:00")) == new


def test_local_pdf_name_rejects_empty_modified() -> None:
    with pytest.raises(ValueError, match="modified"):
        dlm.local_pdf_name(_item(710, "--"))


def test_build_manifest_keeps_ministry_files_unlisted(fake_api: FakeApi) -> None:
    ministry = {**_item(710, "2026-10-20"), "source_url": f"{dlm.MINISTRY_SITE}157.pdf"}
    fake_api.items = [ministry]
    first = dlm.build_manifest(None, posts=[])
    fake_api.items = [
        {**_item(25746, "2026-10-21"), "source_url": "https://insp.cd/157.pdf"}
    ]
    second = dlm.build_manifest(first, posts=[])
    assert second.select("media_id", "in_api", "is_latest").rows() == [
        (710, False, True),
        (25746, True, False),
    ]


def test_build_manifest_rejects_insp_id_of_a_ministry_file(fake_api: FakeApi) -> None:
    fake_api.items = [
        {**_item(710, "2026-10-20"), "source_url": f"{dlm.MINISTRY_SITE}a.pdf"}
    ]
    first = dlm.build_manifest(None, posts=[])
    fake_api.items = [
        {**_item(710, "2026-10-28"), "source_url": "https://insp.cd/b.pdf"}
    ]
    with pytest.raises(ValueError, match="clash"):
        dlm.build_manifest(first, posts=[])
