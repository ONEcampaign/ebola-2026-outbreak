"""Tests for sitrep extraction, release, the status ledger and issues."""

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from ebola_2026_outbreak import drc_2026_sitrep_gemini as gemini
from ebola_2026_outbreak import drc_2026_sitrep_mistral as mistral
from ebola_2026_outbreak import drc_2026_sitrep_review as review
from ebola_2026_outbreak import drc_2026_sitreps_extract as ex
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    EXTRACTORS,
    ModelResult,
    RowKey,
    TextRow,
    cache_path,
    load_gazetteer,
    write_model_result,
    write_text_rows,
)
from ebola_2026_outbreak.drc_2026_sitrep_textlayer import read_rows
from ebola_2026_outbreak.drc_2026_sitreps_download import MANIFEST_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"
GEMINI_96 = FIXTURES / "sitrep_models" / "gemini_response.json"
MISTRAL_157 = FIXTURES / "sitrep_models" / "mistral_157_response.json"
WORDS_157 = FIXTURES / "sitrep_words" / "SitRep_157.json"

DAY = date(2026, 8, 19)
NOW = "2026-10-20T12:00:00+00:00"
SHA_A, SHA_B = "aa" * 32, "bb" * 32

ZONE = RowKey("zones", "Tshopo", "Makiso-Kisangani")
# Text-layer rows matching the SitRep 96 Gemini fixture; CTE deaths is blank.
TEXT_96 = {
    RowKey("provinces", "Tshopo", ""): "1 12 6 50,0 % 1/23",
    RowKey("zone_subtotals", "Tshopo", ""): "12 6 50,0 % 1 0 0",
    RowKey("zone_total", "", ""): "12 6 50,0 % 1 0 0",
    ZONE: "12 6 50,0 % 1 0 0",
}


def _result(raw: dict[str, Any], model: str) -> ModelResult:
    return ModelResult(raw, model, model, {}, "2026-10-19T08:00:00+00:00", {})


def _raw_96() -> dict[str, Any]:
    return gemini.parse_response(json.loads(GEMINI_96.read_text()))


def _raw_157() -> dict[str, Any]:
    return mistral.parse_annotation(json.loads(MISTRAL_157.read_text()))


def _cache(sha: str, raw: dict[str, Any], text: dict[RowKey, TextRow]) -> None:
    """Cache one report as both models read it, with its text layer."""
    write_model_result(
        sha, "gemini", _result(raw, gemini.GEMINI_MODEL), root=ex.CACHE_DIR
    )
    write_model_result(
        sha, "mistral", _result(raw, mistral.MISTRAL_OCR_MODEL), root=ex.CACHE_DIR
    )
    write_text_rows(sha, text.values(), root=ex.CACHE_DIR)


def _text_96(*, located: bool = True) -> dict[RowKey, TextRow]:
    """SitRep 96 text rows; with `located` False the zone row is not found."""
    rows = {key: TextRow(key, printed, True, 3) for key, printed in TEXT_96.items()}
    if not located:
        rows[ZONE] = TextRow(ZONE, None, False, None)
    return rows


def _manifest(*uploads: tuple[int, str]) -> None:
    """Write a manifest whose last upload is the latest for `DAY`."""
    rows = [
        {
            "outbreak": "drc_2026",
            "report_date": DAY,
            "sitrep_number": 96,
            "is_latest": i == len(uploads) - 1,
            "in_api": True,
            "media_id": media_id,
            "modified": modified,
            "source_url": f"https://example.org/{media_id}.pdf",
        }
        for i, (media_id, modified) in enumerate(uploads)
    ]
    pl.DataFrame(rows).select(
        pl.col(c) if c in rows[0] else pl.lit(None).alias(c) for c in MANIFEST_SCHEMA
    ).cast(MANIFEST_SCHEMA).write_csv(ex.MANIFEST_PATH)


def _ledger(*rows: tuple[int, str, str]) -> None:
    """Write a ledger of extracted, not yet released uploads (media, modified, sha)."""
    ex.write_ledger(
        {
            (media_id, modified): ex.LedgerRow(
                DAY, media_id, modified, sha, "pending", *[None] * 7
            )
            for media_id, modified, sha in rows
        }
    )


@pytest.fixture(autouse=True)
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, path in {
        "MANIFEST_PATH": "manifest.csv",
        "CACHE_DIR": "cache",
        "REVIEW_DIR": "review",
        "BY_ZONE_PATH": "processed/by_zone.csv",
        "HEADLINE_PATH": "processed/headline.csv",
        "STATUS_PATH": "processed/status.csv",
    }.items():
        monkeypatch.setattr(ex, name, tmp_path / path)
    return tmp_path


@pytest.fixture
def issues(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Calls to the issue wrappers; each new issue is numbered from 7."""
    calls: list[tuple[str, Any]] = []

    def create(*, title: str, body: str, dry_run: bool = False) -> int:
        calls.append(("create", title))
        return 6 + sum(c[0] == "create" for c in calls)

    def close(number: int, *, comment: str, dry_run: bool = False) -> None:
        calls.append(("close", number))

    monkeypatch.setattr(ex, "create_issue", create)
    monkeypatch.setattr(ex, "close_issue", close)
    return calls


def _status() -> dict[int, dict[str, Any]]:
    return {r["media_id"]: r for r in pl.read_csv(ex.STATUS_PATH).to_dicts()}


def _fill_review(
    path: Path, values: dict[str, str], *, section: str | None = None
) -> None:
    """Type `values` by field into the rows of `section`, or of every section."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if section in (None, row["section"]):
            row["reviewed_value"] = values[row["field"]]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# Release


def test_clean_release_writes_outputs_and_ledger(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _cache(SHA_A, _raw_96(), _text_96())

    ex.run("release", now=NOW)

    by_zone = pl.read_csv(ex.BY_ZONE_PATH, schema=ex.BY_ZONE_SCHEMA)
    assert by_zone.columns == list(ex.BY_ZONE_SCHEMA)
    assert by_zone.height == 3 * 6
    levels = by_zone.select("level", "province", "zone").unique().sort("level")
    assert levels.rows() == [
        ("national", None, None),
        ("province", "Tshopo", None),
        ("zone", "Tshopo", "Makiso-Kisangani"),
    ]
    zone = dict(by_zone.filter(level="zone").select("indicator", "value").iter_rows())
    assert zone["CTE deaths (24h)"] is None
    assert zone["Community deaths (24h)"] == 0
    # A printed blank is an empty field; a printed 0 is "0".
    lines = ex.BY_ZONE_PATH.read_text().splitlines()
    assert any(",CTE deaths (24h),,96," in line for line in lines)
    assert any(",Community deaths (24h),0,96," in line for line in lines)

    headline = pl.read_csv(ex.HEADLINE_PATH, schema=ex.HEADLINE_SCHEMA)
    assert headline.columns == list(ex.HEADLINE_SCHEMA)
    assert dict(headline.select("indicator", "value").iter_rows()) == {
        "Cumulative confirmed cases": 12.0,
        "Cumulative confirmed deaths": 6.0,
        "Case fatality rate (%)": 50.0,
        "Patients in isolation": 3.0,
        "Cumulative recoveries": 4.0,
        "Contacts followed up (%)": 87.5,
    }

    row = _status()[1]
    assert row["status"] == "released"
    assert row["released_at"] == NOW
    assert row["flagged_cells"] == 0
    assert row["gemini_model"] == gemini.GEMINI_MODEL
    assert not ex.REVIEW_DIR.exists()
    assert issues == []


def test_flagged_report_is_held_once(issues: list[tuple[str, Any]]) -> None:
    raw = _raw_157()
    gazetteer = load_gazetteer()
    _manifest((710, "t1"))
    _ledger((710, "t1", SHA_A))
    _cache(SHA_A, raw, read_rows(WORDS_157, gazetteer))

    ex.run("release", now=NOW)

    assert _status()[710]["status"] == "held"
    assert _status()[710]["issue_number"] == 7
    assert issues == [("create", "Review 1 held DRC sitrep")]
    (path,) = ex.REVIEW_DIR.iterdir()
    assert path.name == f"{DAY}_{SHA_A[:8]}.csv"
    rows = review.read_review(path)
    assert len(rows) == _status()[710]["flagged_cells"] > 0
    # Mistral returned 4 of the 8 subtotal rows.
    assert any(r["reason"] == "section_missing" for r in rows)
    assert not ex.BY_ZONE_PATH.exists() or pl.read_csv(ex.BY_ZONE_PATH).is_empty()

    # A second run neither rewrites the file nor opens another issue.
    review.write_review(path, [{**rows[0], "pdf_url": "edited"}, *rows[1:]])
    before = path.read_text()
    ex.run("release", now=NOW)
    assert path.read_text() == before
    assert issues == [("create", "Review 1 held DRC sitrep")]
    assert _status()[710]["issue_number"] == 7


def _hold_96(sha: str = SHA_A) -> Path:
    """Cache SitRep 96 with its zone row missing from the text layer."""
    _cache(sha, _raw_96(), _text_96(located=False))
    return ex.REVIEW_DIR / f"{DAY}_{sha[:8]}.csv"


TRUE_ZONE = {
    "cum_confirmed": "12",
    "cum_deaths": "6",
    "cfr_pct": "50.0",
    "new_confirmed": "1",
    "community_deaths": "0",
    "cte_deaths": "blank",
    "total_deaths_day": "0",
}


def test_completed_review_releases_and_closes_issue(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    path = _hold_96()
    ex.run("release", now=NOW)
    assert _status()[1]["status"] == "held"
    assert {r["reason"] for r in review.read_review(path)} == {"row_not_located"}

    _fill_review(path, TRUE_ZONE)
    ex.run("release", now=NOW)

    row = _status()[1]
    assert row["status"] == "released"
    assert row["flagged_cells"] == 7
    assert row["issue_number"] == 7
    assert issues == [("create", "Review 1 held DRC sitrep"), ("close", 7)]
    by_zone = pl.read_csv(ex.BY_ZONE_PATH, schema=ex.BY_ZONE_SCHEMA)
    zone = by_zone.filter(level="zone", indicator="CTE deaths (24h)")
    assert zone["value"].to_list() == [None]


def test_review_file_grows_when_flags_change(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    path = _hold_96()
    ex.run("release", now=NOW)
    # The reviewer starts on the zone row, typing a wrong count.
    _fill_review(path, {**TRUE_ZONE, "cum_confirmed": "99"})
    before = path.read_text()

    # A changed text layer moves the flag from the zone row to its subtotal.
    text = _text_96()
    subtotal = RowKey("zone_subtotals", "Tshopo", "")
    text[subtotal] = TextRow(subtotal, None, False, None)
    write_text_rows(SHA_A, text.values(), root=ex.CACHE_DIR)
    lines = ex.run("release", now=NOW)

    after = path.read_text()
    assert after.startswith(before)
    rows = review.read_review(path)
    assert [r["section"] for r in rows] == ["zones"] * 7 + ["zone_subtotals"] * 7
    assert all(r["reviewed_value"] == "" for r in rows[7:])
    assert f"Wrote 7 open cells to the end of {path}" in lines
    assert _status()[1]["status"] == "held"

    # Once the new rows are filled the report releases; the zone row is no
    # longer flagged, so its typed values are ignored.
    _fill_review(path, TRUE_ZONE, section="zone_subtotals")
    ex.run("release", now=NOW)
    assert _status()[1]["status"] == "released"
    by_zone = pl.read_csv(ex.BY_ZONE_PATH, schema=ex.BY_ZONE_SCHEMA)
    zone = by_zone.filter(level="zone", indicator="Cumulative confirmed cases")
    assert zone["value"].to_list() == [12]
    assert issues == [("create", "Review 1 held DRC sitrep"), ("close", 7)]


def test_review_of_replaced_upload_is_superseded(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    old_review = _hold_96(SHA_A)
    ex.run("release", now=NOW)

    # The upload is replaced in place; the new PDF reads clean.
    _manifest((1, "t2"))
    ledger = ex.read_ledger()
    ledger[1, "t2"] = ledger[1, "t1"]._replace(modified="t2", sha256=SHA_B)
    ex.write_ledger(ledger)
    _cache(SHA_B, _raw_96(), _text_96())
    _fill_review(old_review, TRUE_ZONE)
    ex.run("release", now=NOW)

    statuses = {
        (r["modified"], r["status"]) for r in pl.read_csv(ex.STATUS_PATH).to_dicts()
    }
    assert statuses == {("t1", "superseded"), ("t2", "released")}
    assert issues == [("create", "Review 1 held DRC sitrep"), ("close", 7)]
    assert pl.read_csv(ex.BY_ZONE_PATH)["modified"].unique().to_list() == ["t2"]


def test_held_reupload_keeps_released_values(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _cache(SHA_A, _raw_96(), _text_96())
    ex.run("release", now=NOW)

    _manifest((1, "t1"), (2, "t2"))
    ledger = ex.read_ledger()
    ledger[2, "t2"] = ex.LedgerRow(DAY, 2, "t2", SHA_B, "pending", *[None] * 7)
    ex.write_ledger(ledger)
    _hold_96(SHA_B)
    ex.run("release", now=NOW)

    assert {m: r["status"] for m, r in _status().items()} == {
        1: "released",
        2: "held",
    }
    by_zone = pl.read_csv(ex.BY_ZONE_PATH, schema=ex.BY_ZONE_SCHEMA)
    assert by_zone["media_id"].unique().to_list() == [1]
    assert by_zone.height == 18


def test_release_refuses_to_drop_a_released_date() -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _cache(SHA_A, _raw_96(), _text_96())
    ex.run("release", now=NOW)
    written = {p: p.read_text() for p in ex.STATUS_PATH.parent.iterdir()}

    # The released upload's cache is lost: its date would drop out.
    for extractor in EXTRACTORS:
        cache_path(SHA_A, extractor, root=ex.CACHE_DIR).unlink()
    with pytest.raises(ValueError, match="Dates would drop out"):
        ex.run("release", now=NOW)
    assert {p: p.read_text() for p in ex.STATUS_PATH.parent.iterdir()} == written


# Extract


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Fake clients answering with SitRep 157; `fail` maps a model to its error."""
    state: dict[str, Any] = {"calls": [], "fail": {}, "model": {}}

    def fake(extractor: str, pin: str):
        def extract(pdf_path: Path) -> ModelResult:
            state["calls"].append(extractor)
            if error := state["fail"].get(extractor):
                raise error
            result = _result(_raw_157(), pin)
            return result._replace(model_id_returned=state["model"].get(extractor, pin))

        return extract

    monkeypatch.setattr(gemini, "extract", fake("gemini", gemini.GEMINI_MODEL))
    monkeypatch.setattr(mistral, "extract", fake("mistral", mistral.MISTRAL_OCR_MODEL))
    pdf = tmp_path / "710.pdf"
    pdf.write_bytes(WORDS_157.read_bytes())
    # read_rows reads a word dump the way it reads a PDF.
    monkeypatch.setattr(ex, "read_rows", lambda _, g: read_rows(WORDS_157, g))
    monkeypatch.setattr(ex, "download", lambda item, refresh=False: pdf)
    _manifest((710, "t1"))
    return state


def test_extract_fills_cache_once(
    clients: dict[str, Any], issues: list[tuple[str, Any]]
) -> None:
    ex.run("extract", now=NOW)
    assert clients["calls"] == ["gemini", "mistral"]
    row = _status()[710]
    sha = row["sha256"]
    assert row["gemini_model"] == gemini.GEMINI_MODEL
    for name in (*EXTRACTORS, "textlayer"):
        assert cache_path(sha, name, root=ex.CACHE_DIR).exists()

    ex.run(None, now=NOW)
    assert clients["calls"] == ["gemini", "mistral"]
    assert _status()[710]["status"] == "held"


@pytest.mark.parametrize(
    ("fail", "model"),
    [
        ({"gemini": httpx.ConnectError("down")}, {}),
        ({"mistral": httpx.HTTPStatusError("429", request=None, response=None)}, {}),
        ({"gemini": gemini.GeminiError("finishReason: MAX_TOKENS")}, {}),
        ({"mistral": ValueError("no document_annotation")}, {}),
        ({}, {"mistral": "mistral-ocr-5-0"}),
    ],
    ids=["transport", "status", "gemini", "annotation", "model-id"],
)
def test_extractor_failure_leaves_upload_pending(
    clients: dict[str, Any],
    issues: list[tuple[str, Any]],
    fail: dict[str, Exception],
    model: dict[str, str],
) -> None:
    clients["fail"], clients["model"] = fail, model
    ex.run(None, now=NOW)

    row = _status()[710]
    assert row["status"] == "pending"
    assert row["extracted_at"] == NOW
    failed = next(iter(fail or model))
    assert not cache_path(row["sha256"], failed, root=ex.CACHE_DIR).exists()
    assert issues == []

    # The next run retries only the missing answer, and releases.
    clients["fail"], clients["model"], clients["calls"] = {}, {}, []
    ex.run(None, now=NOW)
    assert clients["calls"] == [failed]
    assert _status()[710]["status"] == "held"


def test_pending_upload_opens_an_issue_on_day_three(
    clients: dict[str, Any], issues: list[tuple[str, Any]]
) -> None:
    clients["fail"] = {"gemini": httpx.ConnectError("down")}
    for day in ("2026-10-20", "2026-10-21"):
        ex.run(None, now=f"{day}T12:00:00+00:00")
    assert issues == []

    ex.run(None, now="2026-10-22T12:00:00+00:00")
    assert issues == [("create", "DRC sitrep extraction pending for 1 upload")]
    row = _status()[710]
    assert (row["status"], row["issue_number"]) == ("pending", 7)
    assert row["extracted_at"] == "2026-10-20T12:00:00+00:00"

    ex.run(None, now="2026-10-23T12:00:00+00:00")
    assert len(issues) == 1

    # Extraction recovers: the report is held, so the pending issue closes
    # and the review issue opens.
    clients["fail"] = {}
    ex.run(None, now="2026-10-24T12:00:00+00:00")
    assert issues[1:] == [("create", "Review 1 held DRC sitrep"), ("close", 7)]
    assert _status()[710]["issue_number"] == 8


@pytest.fixture
def no_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on any gh call, as if inside GitHub Actions."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("gh called in a dry run")

    monkeypatch.setattr(review.subprocess, "run", refuse)


@pytest.mark.usefixtures("no_gh")
def test_dry_run_without_keys_calls_and_writes_nothing(
    clients: dict[str, Any], paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in (gemini, mistral):
        monkeypatch.delenv(module.API_KEY_ENV, raising=False)
    monkeypatch.setattr(ex, "download", lambda *_, **__: pytest.fail("downloaded"))
    before = sorted(paths.rglob("*"))

    lines = ex.run(None, now=NOW, dry_run=True)

    assert clients["calls"] == []
    assert sorted(paths.rglob("*")) == before
    assert lines[0] == (
        "Extract: would extract 1 uploads; API keys not set, so nothing was called"
    )
    assert "Release: 0 released, 0 held, 1 pending, 0 superseded" in lines


@pytest.mark.usefixtures("no_gh")
def test_dry_run_with_keys_caches_answers_and_writes_no_release(
    clients: dict[str, Any], paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in (gemini, mistral):
        monkeypatch.setenv(module.API_KEY_ENV, "key")

    lines = ex.run(None, now=NOW, dry_run=True)

    assert clients["calls"] == ["gemini", "mistral"]
    (cached,) = ex.CACHE_DIR.iterdir()
    assert sorted(p.name for p in cached.iterdir()) == [
        "gemini.json",
        "mistral.json",
        "textlayer.json",
    ]
    assert not ex.STATUS_PATH.parent.exists()
    assert not ex.REVIEW_DIR.exists()
    # Release read the fresh cache through the ledger held in memory.
    assert "Release: 0 released, 1 held, 0 pending, 0 superseded" in lines
    assert any(line.startswith("Would write review file") for line in lines)
    assert any(line.startswith("Would open an issue for 1 held") for line in lines)
