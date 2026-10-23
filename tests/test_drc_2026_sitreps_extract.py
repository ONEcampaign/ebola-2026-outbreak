"""Tests for sitrep extraction, release, the status ledger and issues."""

import csv
import json
import shutil
import subprocess
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
    GAZETTEER_PATH,
    ModelResult,
    RowKey,
    cache_path,
    write_model_result,
    write_words,
)
from ebola_2026_outbreak.drc_2026_sitreps_download import MANIFEST_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"
GEMINI_96 = FIXTURES / "sitrep_models" / "gemini_response.json"
MISTRAL_157 = FIXTURES / "sitrep_models" / "mistral_157_response.json"
WORDS_157 = FIXTURES / "sitrep_words" / "SitRep_157.json"

DAY = date(2026, 8, 19)
NOW = "2026-10-20T12:00:00+00:00"
SHA_A, SHA_B = "aa" * 32, "bb" * 32

ZONE = RowKey("zones", "Tshopo", "Makiso-Kisangani")
# A zone-table row matching the SitRep 96 Gemini fixture; CTE deaths is blank.
ZONE_ROW_96 = ("12", "6", "50,0 %", "1", "0", None, "0")


def _result(raw: dict[str, Any], model: str) -> ModelResult:
    return ModelResult(raw, model, model, {}, "2026-10-19T08:00:00+00:00", {})


def _raw_96() -> dict[str, Any]:
    return gemini.parse_response(json.loads(GEMINI_96.read_text()))


def _raw_157() -> dict[str, Any]:
    return mistral.parse_annotation(json.loads(MISTRAL_157.read_text()))


def _cache(
    sha: str,
    raw: dict[str, Any],
    words: Path,
    *,
    mistral_raw: dict[str, Any] | None = None,
) -> None:
    """Cache one report as both models read it, or as Mistral reads it
    differently, with its word dump."""
    write_model_result(
        sha, "gemini", _result(raw, gemini.GEMINI_MODEL), root=ex.CACHE_DIR
    )
    write_model_result(
        sha,
        "mistral",
        _result(mistral_raw or raw, mistral.MISTRAL_OCR_MODEL),
        root=ex.CACHE_DIR,
    )
    write_words(sha, words, root=ex.CACHE_DIR)


def _header(*words: str) -> list[str]:
    """Dump words of a table header at top 100, column i at x 100 + 50 i."""
    return [f"100 {100 + 50 * i} {120 + 50 * i} {w}" for i, w in enumerate(words)]


def _line(top: int, label: str, cells: tuple[str | None, ...]) -> list[str]:
    """Dump words of one table row: its label, then each cell's words under
    its column's header, None leaving the cell blank."""
    words = [f"{top} 40 60 {label}"]
    for i, cell in enumerate(cells):
        for j, t in enumerate((cell or "").split()):
            x = 100 + 50 * i + 10 * j
            words.append(f"{top} {x} {x + 8} {t}")
    return words


def _words_96(
    *,
    zone: str | None = ZONE.zone,
    subtotal: bool = True,
    zero_zone: str | None = None,
    name: str = "words",
) -> Path:
    """A word dump printing the rows of the SitRep 96 Gemini fixture.

    Page 2 holds the province table, new cases first, and page 3 the zone
    table: the Tshopo subtotal row (left out when `subtotal` is False), a
    zone row labelled `zone` (left out when None), a row of zeros labelled
    `zero_zone` (when given), and the Total row.
    """
    province_row = ("1", "12", "6", "50,0 %", "1/23")
    province = [
        *_header("Nouveaux", "Cas", "Décès", "Létalité", "touchées"),
        *_line(130, "Tshopo", province_row),
        *_line(150, "Total", province_row),
    ]
    zones = _header("(n)", "(n)", "Létalité", "Nouv.", "(Swab+)", "intra-CTE", "Total")
    if subtotal:
        zones += _line(130, "Tshopo", ZONE_ROW_96)
    if zone is not None:
        zones += _line(150, zone, ZONE_ROW_96)
    if zero_zone is not None:
        zones += _line(160, zero_zone, ("0", "0", None, "0", "0", None, "0"))
    zones += _line(170, "Total", ZONE_ROW_96)
    path = ex.CACHE_DIR.parent / f"{name}.json"
    path.write_text(json.dumps({"pages": {"2": province, "3": zones}}))
    return path


def _manifest(*uploads: tuple[int, str], report_date: date = DAY) -> None:
    """Write a manifest whose last upload is the latest for `report_date`."""
    rows = [
        {
            "outbreak": "drc_2026",
            "report_date": report_date,
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

    monkeypatch.setattr(ex, "open_issues", lambda *, dry_run=False: {})
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
    _cache(SHA_A, _raw_96(), _words_96())

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
    _manifest((710, "t1"))
    _ledger((710, "t1", SHA_A))
    _cache(SHA_A, _raw_157(), WORDS_157)

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
    _cache(sha, _raw_96(), _words_96(zone=None))
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
    write_words(SHA_A, _words_96(subtotal=False), root=ex.CACHE_DIR)
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


def test_sum_opened_by_a_review_releases_as_a_discrepancy(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    path = _hold_96()
    ex.run("release", now=NOW)

    # The PDF prints 13 cases for the zone and 12 for its province: the
    # subtotal check runs once the zone is typed in, and fails.
    _fill_review(path, {**TRUE_ZONE, "cum_confirmed": "13"})
    ex.run("release", now=NOW)
    assert _status()[1]["status"] == "held"
    rows = review.read_review(path)
    assert [(r["section"], r["field"], r["reason"]) for r in rows[7:]] == [
        ("zone_subtotals", "cum_confirmed", "sum_mismatch")
    ]

    # The reviewer confirms the subtotal as printed. Both failures rest on
    # verified values only: the subtotal sum, and the zone's CFR (6 of 13
    # is not 50.0).
    _fill_review(path, {**TRUE_ZONE, "cum_confirmed": "12"}, section="zone_subtotals")
    ex.run("release", now=NOW)
    row = _status()[1]
    assert (row["status"], row["flagged_cells"], row["discrepancies"]) == (
        "released",
        8,
        2,
    )
    by_zone = pl.read_csv(ex.BY_ZONE_PATH, schema=ex.BY_ZONE_SCHEMA)
    cases = by_zone.filter(indicator="Cumulative confirmed cases")
    assert dict(cases.select("level", "value").iter_rows()) == {
        "national": 12,
        "province": 12,
        "zone": 13,
    }


MISSPELT = "Makiso-Kisangany"


@pytest.mark.parametrize(
    ("printed", "verified"),
    [
        # The text layer, read with the alias, confirms the models' row, so
        # the typed values are not needed.
        (MISSPELT, 0),
        # The text layer cannot confirm it, so the typed values verify it.
        (None, 7),
    ],
    ids=["pdf-prints-alias", "row-not-in-text"],
)
def test_alias_and_values_in_one_review_release(
    issues: list[tuple[str, Any]],
    paths: Path,
    monkeypatch: pytest.MonkeyPatch,
    printed: str | None,
    verified: int,
) -> None:
    gazetteer = paths / "zones.csv"
    gazetteer.write_text(GAZETTEER_PATH.read_text())
    monkeypatch.setattr(ex, "GAZETTEER_PATH", gazetteer)
    raw = _raw_96()
    raw["zones"][0]["zone"] = MISSPELT
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _cache(SHA_A, raw, _words_96(zone=printed))
    ex.run("release", now=NOW)
    path = ex.REVIEW_DIR / f"{DAY}_{SHA_A[:8]}.csv"
    assert {(r["zone"], r["reason"]) for r in review.read_review(path)} == {
        (MISSPELT, "unknown_name")
    }

    # One pull request adds the spelling as an alias and types the values.
    canonical = f"Tshopo,{ZONE.zone},\n"
    gazetteer.write_text(
        gazetteer.read_text().replace(canonical, f"Tshopo,{ZONE.zone},{MISSPELT}\n")
    )
    _fill_review(path, TRUE_ZONE)
    ex.run("release", now=NOW)

    row = _status()[1]
    assert (row["status"], row["flagged_cells"]) == ("released", verified)
    by_zone = pl.read_csv(ex.BY_ZONE_PATH, schema=ex.BY_ZONE_SCHEMA)
    assert by_zone.filter(level="zone")["zone"].unique().to_list() == [ZONE.zone]


# A misreading of Lubunga, a Tshopo zone.
MISSPELT_ZERO = "Lubinga"


@pytest.mark.parametrize(
    ("both_misread", "printed", "zero_zone", "status"),
    [
        # Gemini's row matches the page; Mistral's misread is dropped.
        (False, ZONE.zone, False, "released"),
        (True, ZONE.zone, False, "held"),
        # The page prints the name, so it may be a zone new to the gazetteer.
        (False, MISSPELT, False, "held"),
        # Mistral misreads Lubunga, a row of zeros Gemini left out, so the
        # misread is its only copy and no sum would miss it.
        (False, ZONE.zone, True, "held"),
    ],
    ids=["one-model-unprinted", "both-models", "one-model-printed", "only-copy"],
)
def test_unknown_zone_name_of_one_model_is_dropped_unless_printed(
    issues: list[tuple[str, Any]],
    both_misread: bool,
    printed: str,
    zero_zone: bool,
    status: str,
) -> None:
    misread = _raw_96()
    if zero_zone:
        zeros = dict.fromkeys(misread["zones"][0], 0)
        misread["zones"].append(
            {**zeros, "province": "Tshopo", "zone": MISSPELT_ZERO, "cfr_pct": None}
        )
    else:
        misread["zones"][0]["zone"] = MISSPELT
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    gemini_raw = misread if both_misread else _raw_96()
    words = _words_96(zone=printed, zero_zone="Lubunga" if zero_zone else None)
    _cache(SHA_A, gemini_raw, words, mistral_raw=misread)

    ex.run("release", now=NOW)

    assert _status()[1]["status"] == status
    path = ex.REVIEW_DIR / f"{DAY}_{SHA_A[:8]}.csv"
    unknown = {
        r["zone"]
        for r in (review.read_review(path) if path.exists() else [])
        if r["reason"] == "unknown_name"
    }
    expected = MISSPELT_ZERO if zero_zone else MISSPELT
    assert unknown == (set() if status == "released" else {expected})


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
    _cache(SHA_B, _raw_96(), _words_96())
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
    _cache(SHA_A, _raw_96(), _words_96())
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


def test_upload_replaced_with_another_date_drops_its_old_date(
    issues: list[tuple[str, Any]],
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _cache(SHA_A, _raw_96(), _words_96())
    ex.run("release", now=NOW)

    # The ministry replaces the upload in place with the next day's report,
    # which waits for review.
    later = date(2026, 8, 20)
    _manifest((1, "t2"), report_date=later)
    ledger = ex.read_ledger()
    ledger[1, "t2"] = ex.LedgerRow(later, 1, "t2", SHA_B, "pending", *[None] * 7)
    ex.write_ledger(ledger)
    _hold_96(SHA_B)
    lines = ex.run("release", now=NOW)

    statuses = {
        (r["modified"], r["status"]) for r in pl.read_csv(ex.STATUS_PATH).to_dicts()
    }
    assert statuses == {("t1", "superseded"), ("t2", "held")}
    assert pl.read_csv(ex.BY_ZONE_PATH).is_empty()
    assert (
        f"Dropped {DAY} from the outputs: the ministry replaced the upload with a "
        "report of another date"
    ) in lines


def test_release_refuses_to_drop_a_released_date() -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _cache(SHA_A, _raw_96(), _words_96())
    ex.run("release", now=NOW)
    written = {p: p.read_text() for p in ex.STATUS_PATH.parent.iterdir()}

    # The released upload's cache is lost: its date would drop out.
    for extractor in EXTRACTORS:
        cache_path(SHA_A, extractor, root=ex.CACHE_DIR).unlink()
    with pytest.raises(ex.OutputError) as e:
        ex.run("release", now=NOW)
    assert e.value.code == "dates_dropped"
    assert {p: p.read_text() for p in ex.STATUS_PATH.parent.iterdir()} == written


# Extract


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Fake clients answering with SitRep 157.

    `fail` maps a model to its error, `model` to the model id it returns and
    `answer` to the raw extraction it returns.
    """
    state: dict[str, Any] = {"calls": [], "fail": {}, "model": {}, "answer": {}}

    def fake(extractor: str, pin: str):
        def extract(pdf_path: Path) -> ModelResult:
            state["calls"].append(extractor)
            if error := state["fail"].get(extractor):
                raise error
            result = _result(state["answer"].get(extractor, _raw_157()), pin)
            return result._replace(model_id_returned=state["model"].get(extractor, pin))

        return extract

    monkeypatch.setattr(gemini, "extract", fake("gemini", gemini.GEMINI_MODEL))
    monkeypatch.setattr(mistral, "extract", fake("mistral", mistral.MISTRAL_OCR_MODEL))
    pdf = tmp_path / "710.pdf"
    pdf.write_bytes(WORDS_157.read_bytes())
    monkeypatch.setattr(
        ex, "dump_words", lambda _, out: shutil.copyfile(WORDS_157, out)
    )
    monkeypatch.setattr(ex, "download", lambda item: pdf)
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
    for name in (*EXTRACTORS, "words"):
        assert cache_path(sha, name, root=ex.CACHE_DIR).exists()

    ex.run(None, now=NOW)
    assert clients["calls"] == ["gemini", "mistral"]
    assert _status()[710]["status"] == "held"


def _without_zones() -> dict[str, Any]:
    return {k: v for k, v in _raw_157().items() if k != "zones"}


@pytest.mark.parametrize(
    ("fail", "model", "answer"),
    [
        ({"gemini": httpx.ConnectError("down")}, {}, {}),
        (
            {"mistral": httpx.HTTPStatusError("429", request=None, response=None)},
            {},
            {},
        ),
        ({"gemini": gemini.GeminiError("finishReason: MAX_TOKENS")}, {}, {}),
        ({"mistral": ValueError("no document_annotation")}, {}, {}),
        ({}, {"mistral": "mistral-ocr-5-0"}, {}),
        ({}, {}, {"gemini": _without_zones()}),
    ],
    ids=["transport", "status", "gemini", "annotation", "model-id", "malformed"],
)
def test_extractor_failure_leaves_upload_pending(
    clients: dict[str, Any],
    issues: list[tuple[str, Any]],
    fail: dict[str, Exception],
    model: dict[str, str],
    answer: dict[str, dict[str, Any]],
) -> None:
    clients["fail"], clients["model"], clients["answer"] = fail, model, answer
    ex.run(None, now=NOW)

    row = _status()[710]
    assert row["status"] == "pending"
    assert row["extracted_at"] == NOW
    failed = next(iter(fail or model or answer))
    assert not cache_path(row["sha256"], failed, root=ex.CACHE_DIR).exists()
    assert issues == []

    # The next run retries only the missing answer, and releases.
    clients["fail"], clients["model"], clients["answer"] = {}, {}, {}
    clients["calls"] = []
    ex.run(None, now=NOW)
    assert clients["calls"] == [failed]
    assert _status()[710]["status"] == "held"


class PDFSyntaxError(Exception):
    """An error outside `EXTRACTOR_ERRORS`, as pdfminer raises on a bad PDF."""


def test_word_dump_failure_leaves_upload_pending(
    clients: dict[str, Any],
    issues: list[tuple[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def damaged(pdf: Path, out: Path) -> Path:
        raise PDFSyntaxError("damaged PDF")

    monkeypatch.setattr(ex, "dump_words", damaged)
    lines = ex.run(None, now=NOW)

    assert "Extract: 1 uploads lacked records; 0 complete, 1 pending" in lines
    row = _status()[710]
    assert (row["status"], row["extracted_at"]) == ("pending", NOW)
    assert row["sha256"] is not None
    assert clients["calls"] == []


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
        "words.json",
    ]
    assert not ex.STATUS_PATH.parent.exists()
    assert not ex.REVIEW_DIR.exists()
    # Release read the fresh cache through the ledger held in memory.
    assert "Release: 0 released, 1 held, 0 pending, 0 superseded" in lines
    assert any(line.startswith("Would write review file") for line in lines)
    assert any(line.startswith("Would open an issue for 1 held") for line in lines)


# Issues through gh


@pytest.fixture
def gh(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake gh inside Actions; `open` maps open issue numbers to bodies."""
    state: dict[str, Any] = {"open": {}, "calls": []}
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        state["calls"].append(args[1:3])
        out = {
            ("issue", "list"): json.dumps(
                [{"number": n, "body": b} for n, b in state["open"].items()]
            ),
            ("issue", "create"): "https://github.com/o/r/issues/99\n",
        }.get(tuple(args[1:3]), "")
        return subprocess.CompletedProcess(args, 0, out)

    monkeypatch.setattr(review.subprocess, "run", run)
    return state


@pytest.mark.parametrize(
    ("orphan", "number"), [(False, 99), (True, 5)], ids=["new", "adopted"]
)
def test_held_report_adopts_an_orphan_issue(
    gh: dict[str, Any], orphan: bool, number: int
) -> None:
    _manifest((1, "t1"))
    _ledger((1, "t1", SHA_A))
    _hold_96()
    if orphan:
        # Opened by a run whose ledger was never committed.
        gh["open"] = {5: f"Review\n\n{review.held_marker(SHA_A)}\n"}

    ex.run("release", now=NOW)

    assert _status()[1]["issue_number"] == number
    assert (["issue", "create"] in gh["calls"]) is not orphan


def test_pending_upload_adopts_an_orphan_issue(
    clients: dict[str, Any], gh: dict[str, Any]
) -> None:
    clients["fail"] = {"gemini": httpx.ConnectError("down")}
    for day in ("2026-10-20", "2026-10-21"):
        ex.run(None, now=f"{day}T12:00:00+00:00")
    gh["open"] = {
        4: f"Review\n\n{review.held_marker(SHA_A)}\n",
        5: f"Pending\n\n{review.pending_marker(710, 't1')}\n",
    }

    ex.run(None, now="2026-10-22T12:00:00+00:00")

    assert _status()[710]["issue_number"] == 5
    assert ["issue", "create"] not in gh["calls"]
