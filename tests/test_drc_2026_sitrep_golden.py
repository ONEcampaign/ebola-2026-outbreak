"""Offline golden-set test of the sitrep extraction pipeline.

For each golden report, the recorded Gemini and Mistral responses in
`fixtures/sitrep_golden/recorded/` and the report's word dump go through
`judge` with no review, with text-layer arbitration on and off. No accepted
value may differ from the truth, and the share of cells held for review must
stay under a ceiling, so a rise in flags fails here with its count instead of
landing on reviewers.

Record the fixtures with `drc_2026_sitrep_golden.py record`.
"""

import hashlib
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import pytest

from ebola_2026_outbreak import drc_2026_sitrep_golden as golden
from ebola_2026_outbreak.drc_2026_sitrep_golden import (
    NO_TRUTH_ROW,
    WORDS_DIR,
    Report,
    Score,
    WrongCell,
    compare,
    load_reports,
    load_truth,
    read_recorded,
    score,
    truth_cells,
    write_recorded,
)
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    ModelResult,
    RowKey,
    load_gazetteer,
)
from ebola_2026_outbreak.drc_2026_sitreps_extract import Records

GAZETTEER = load_gazetteer()
REPORTS = load_reports()
TRUTH = load_truth()

# Share of truth cells held for review, across the golden reports. Observed with
# the recorded fixtures: 15 of 7,197 (0.21%) with arbitration, 182 of 7,197
# (2.5%) without. The ceilings sit above both, at 0.4% and 3.2%.
FLAG_RATE_CEILING = {True: 0.004, False: 0.032}


@cache
def outcome(report: Report, *, arbitration: bool) -> Score:
    """Score a golden report with its recorded model answers, cached per call."""
    records = Records(
        read_recorded("gemini", report),
        read_recorded("mistral", report),
        WORDS_DIR / report.words,
    )
    return score(records, gazetteer=GAZETTEER, arbitration=arbitration)


@pytest.mark.parametrize("arbitration", [True, False], ids=["arbitration", "strict"])
@pytest.mark.parametrize("report", REPORTS, ids=lambda r: f"SitRep_{r.sitrep_no}")
def test_no_accepted_value_is_wrong(report: Report, arbitration: bool) -> None:
    accepted = outcome(report, arbitration=arbitration).accepted
    assert compare(accepted, TRUTH[report.sitrep_no]) == []


@pytest.mark.parametrize("arbitration", [True, False], ids=["arbitration", "strict"])
def test_flag_rate_under_ceiling(arbitration: bool) -> None:
    flagged = sum(outcome(r, arbitration=arbitration).flagged for r in REPORTS)
    total = sum(truth_cells(TRUTH[r.sitrep_no]) for r in REPORTS)
    assert flagged / total <= FLAG_RATE_CEILING[arbitration], (
        f"{flagged} of {total} cells flagged ({flagged / total:.1%})"
    )


def test_arbitration_reaches_judge() -> None:
    flagged = {
        arbitration: sum(outcome(r, arbitration=arbitration).flagged for r in REPORTS)
        for arbitration in (True, False)
    }
    assert flagged[False] > flagged[True]


def test_every_report_has_truth_and_its_word_dump() -> None:
    assert len(REPORTS) == 13
    for report in REPORTS:
        assert (WORDS_DIR / report.words).exists()
        assert {k.section for k in TRUTH[report.sitrep_no]} == {
            "kpis",
            "provinces",
            "zone_subtotals",
            "zone_total",
            "zones",
        }


def test_compare_lists_each_wrong_cell() -> None:
    zone = RowKey("zones", "Ituri", "Adja")
    kpis = RowKey("kpis", "", "")
    truth = {
        zone: {"cum_confirmed": 11, "new_confirmed": None, "cte_deaths": 0},
        kpis: {"cfr_pct": 47.6},
    }
    other = RowKey("zones", "Ituri", "Bunia")
    accepted = {
        zone: {"cum_confirmed": 11, "new_confirmed": 0, "cte_deaths": None},
        kpis: {"cfr_pct": 47.6},
        other: {"cum_deaths": 3},
    }
    assert compare(accepted, truth) == [
        WrongCell(zone, "new_confirmed", 0, None),
        WrongCell(zone, "cte_deaths", None, 0),
        WrongCell(other, "cum_deaths", 3, NO_TRUTH_ROW),
    ]


def test_oversized_record_drops_page_markdown(tmp_path: Path) -> None:
    result = ModelResult(
        extraction_raw={"sitrep_no": 157},
        model_id_requested="mistral-ocr-4-1",
        model_id_returned="mistral-ocr-4-1",
        usage={"pages_processed": 13},
        requested_at="2026-10-22T00:00:00+00:00",
        response={"model": "mistral-ocr-4-1", "pages": ["x" * 200_000]},
    )
    write_recorded("mistral", REPORTS[0], result, root=tmp_path)
    kept = read_recorded("mistral", REPORTS[0], root=tmp_path)
    assert kept == result._replace(response={"model": "mistral-ocr-4-1"})


def test_run_logs_a_failed_report_and_scores_the_rest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failing, *rest = REPORTS[:3]
    by_name = {r.report_date: r for r in REPORTS}

    def failing_score(records: Records, **kwargs: Any) -> Score:
        """Score `records`, failing on the report named `failing`."""
        if records.words == WORDS_DIR / failing.words:
            raise ValueError("bad row")
        return score(records, **kwargs)

    monkeypatch.setattr(golden, "fetch_pdf", lambda report, _: Path(report.report_date))
    monkeypatch.setattr(
        golden,
        "dump_words",
        lambda pdf, _: WORDS_DIR / by_name[pdf.name].words,
    )
    monkeypatch.setattr(golden, "score", failing_score)
    monkeypatch.setattr(
        golden,
        "EXTRACT",
        {
            e: lambda pdf, e=e: read_recorded(e, by_name[pdf.name])
            for e in ("gemini", "mistral")
        },
    )

    assert golden.run(limit=3) == 1
    captured = capsys.readouterr()
    assert f"SitRep {failing.sitrep_no}" in captured.err
    assert "scoring failed: bad row" in captured.err
    for report in rest:
        assert (
            f"SitRep {report.sitrep_no} ({report.report_date}): 0 accepted-wrong"
            in (captured.out)
        )


def _fake_get(served: dict[str, bytes]) -> Any:
    """A `fetch.get` serving `served` by URL and failing on any other."""

    def get(url: str) -> httpx.Response:
        """Serve `served` by URL, failing on any other."""
        if url not in served:
            raise httpx.ConnectError(f"gone: {url}")
        return httpx.Response(200, content=served[url])

    return get


def _report(ministry: bytes, insp: bytes | None) -> Report:
    """A report row for SitRep 157 with the given ministry and INSP bytes."""
    return Report(
        "2026-10-18",
        157,
        "https://ministry/157.pdf",
        hashlib.sha256(ministry).hexdigest(),
        "SitRep_157.json",
        "https://insp/157.pdf" if insp is not None else "",
        hashlib.sha256(insp).hexdigest() if insp is not None else "",
    )


def test_fetch_falls_back_to_the_insp_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _report(b"ministry", b"insp")
    monkeypatch.setattr(golden.fetch, "get", _fake_get({report.insp_url: b"insp"}))

    assert golden.fetch_pdf(report, tmp_path).read_bytes() == b"insp"


def test_fetch_names_every_failed_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _report(b"ministry", b"insp")
    served = {report.source_url: b"changed"}
    monkeypatch.setattr(golden.fetch, "get", _fake_get(served))

    with pytest.raises(golden.SourceError) as e:
        golden.fetch_pdf(report, tmp_path)
    assert "SOURCE CHANGED" in str(e.value) and "SOURCE GONE" in str(e.value)
    assert not list(tmp_path.iterdir())


def test_local_insp_copy_is_used_before_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _report(b"ministry", b"insp")
    (tmp_path / "25746.pdf").write_bytes(b"insp")
    monkeypatch.setattr(golden.fetch, "get", _fake_get({}))

    local = golden._local_pdfs([tmp_path])
    assert golden._pdf(report, local, tmp_path) == tmp_path / "25746.pdf"


def test_golden_reports_name_insp_copies_by_url() -> None:
    insp = [r for r in REPORTS if r.insp_url]
    assert insp, "no golden report lists an INSP copy"
    for report in insp:
        assert report.insp_url.startswith("https://insp.cd/")
        assert len(report.insp_sha256) == 64
