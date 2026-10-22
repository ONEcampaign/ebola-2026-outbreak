"""Tests for review files, issues, applying a review and the PR check."""

import csv
import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ebola_2026_outbreak import drc_2026_sitrep_review as review
from ebola_2026_outbreak import drc_2026_sitreps_extract as ex
from ebola_2026_outbreak.drc_2026_sitrep_review import (
    HeldReport,
    ReviewRow,
    apply,
    issue_body,
    parse_value,
    read_review,
    review_rows,
    write_review,
)
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    CACHE_DIR,
    FIELDS,
    GAZETTEER_PATH,
    REVIEW_DIR,
    Flag,
    ModelResult,
    RowKey,
    Section,
    Status,
    TextRow,
    Values,
    load_gazetteer,
    normalise,
    write_model_result,
    write_words,
)
from ebola_2026_outbreak.drc_2026_sitrep_validate import check_sums, reconcile
from ebola_2026_outbreak.drc_2026_sitreps_download import (
    MANIFEST_PATH,
    MANIFEST_SCHEMA,
    ROOT,
)

SHA = "ab" * 32
DATE = "2026-10-14"
PDF = "https://example.org/SitRep_152.pdf"

KPIS = RowKey("kpis", "", "")
TOTAL = RowKey("zone_total", "", "")
ITURI = RowKey("zone_subtotals", "Ituri", "")
ITURI_PROVINCE = RowKey("provinces", "Ituri", "")
A = RowKey("zones", "Ituri", "A")
B = RowKey("zones", "Ituri", "B")

SUBTOTAL: Values = {
    "cum_confirmed": 30,
    "cum_deaths": 7,
    "cfr_pct": 23.3,
    "new_confirmed": 1,
    "community_deaths": 1,
    "cte_deaths": 1,
    "total_deaths_day": 2,
}
# A consistent report: one province of two zones, with a blank.
TRUTH: dict[RowKey, Values] = {
    KPIS: {
        "cum_confirmed": 30,
        "cum_deaths": 7,
        "cfr_pct": 23.3,
        "in_isolation": 4,
        "cum_recovered": 10,
        "contact_followup_pct": 95.0,
    },
    ITURI_PROVINCE: {
        "new_confirmed": 1,
        "cum_confirmed": 30,
        "cum_deaths": 7,
        "cfr_pct": 23.3,
        "zs_affected": 2,
        "zs_total": 2,
    },
    ITURI: SUBTOTAL,
    TOTAL: SUBTOTAL,
    A: {
        "cum_confirmed": 10,
        "cum_deaths": 2,
        "cfr_pct": 20.0,
        "new_confirmed": 1,
        "community_deaths": None,
        "cte_deaths": 1,
        "total_deaths_day": 1,
    },
    B: {
        "cum_confirmed": 20,
        "cum_deaths": 5,
        "cfr_pct": 25.0,
        "new_confirmed": 0,
        "community_deaths": 1,
        "cte_deaths": 0,
        "total_deaths_day": 1,
    },
}


def match(text: TextRow, values: Values, section: Section) -> bool:
    """Stub matcher: the text layer "prints" the true row as JSON."""
    return text.printed == json.dumps(dict(values), sort_keys=True)


def text_layer() -> dict[RowKey, TextRow]:
    return {
        key: TextRow(key, json.dumps(values, sort_keys=True), True, 3, None)
        for key, values in TRUTH.items()
        if key.section != "kpis"
    }


def raw(rows: dict[RowKey, Values]) -> dict[str, Any]:
    """A model's raw extraction holding `rows`."""

    def listed(section: Section, *names: str) -> list[dict[str, Any]]:
        return [
            {**dict(zip(names, key[1:], strict=False)), **values}
            for key, values in rows.items()
            if key.section == section
        ]

    return {
        "sitrep_no": 152,
        "report_date": DATE,
        "kpis": rows.get(KPIS),
        "zone_total": rows.get(TOTAL),
        "provinces": listed("provinces", "province"),
        "zone_subtotals": listed("zone_subtotals", "province"),
        "zones": listed("zones", "province", "zone"),
    }


def write_gazetteer(path: Path, rows: str = "Ituri,A,\nIturi,B,\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("province,zone,aliases\n" + rows)
    return path


@pytest.fixture
def gazetteer(tmp_path: Path):
    return load_gazetteer(write_gazetteer(tmp_path / "zones.csv"))


def row(
    key: RowKey, field: str, value: str, reason: str = "text_mismatch"
) -> ReviewRow:
    return {
        "report_date": DATE,
        "sha256": SHA,
        "section": key.section,
        "province": key.province,
        "zone": key.zone,
        "field": field,
        "page": 3,
        "pdf_url": f"{PDF}#page=3",
        "reason": reason,
        "reviewed_value": value,
    }


# Review files


def test_review_rows_round_trip(tmp_path: Path) -> None:
    flags = [
        Flag(A, "cum_deaths", "text_mismatch", 4),
        Flag(KPIS, "in_isolation", "models_disagree"),
        Flag(ITURI, "cum_deaths", "sum_mismatch"),
        Flag(A, "cum_deaths", "sum_mismatch"),
        Flag(RowKey("provinces", "", ""), None, "section_missing"),
    ]
    rows = review_rows(
        flags, report_date=DATE, sha256=SHA, pdf_url=PDF, text_rows=text_layer()
    )

    cells = [
        (r["section"], r["zone"], r["field"], r["page"], r["pdf_url"]) for r in rows
    ]
    assert cells == [
        ("kpis", "", "in_isolation", None, PDF),
        ("provinces", "", "", None, PDF),
        ("zone_subtotals", "", "cum_deaths", 3, f"{PDF}#page=3"),
        ("zones", "A", "cum_deaths", 4, f"{PDF}#page=4"),
    ]
    assert rows[3]["reason"] == "text_mismatch"
    assert all(r["reviewed_value"] == "" for r in rows)
    assert read_review(write_review(tmp_path / "r.csv", rows)) == rows


@pytest.mark.parametrize(
    ("text", "field", "value"),
    [
        ("12", "cum_confirmed", 12),
        ("0", "cum_confirmed", 0),
        ("48.3", "cfr_pct", 48.3),
        ("48", "cfr_pct", 48.0),
        ("blank", "cum_deaths", None),
        ("Blank", "cfr_pct", None),
    ],
)
def test_parse_value(text: str, field: str, value: float | None) -> None:
    parsed = parse_value(text, field)
    assert parsed == value
    assert type(parsed) is type(value)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("48,3", "cfr_pct"),
        ("1 234", "cum_confirmed"),
        ("12.0", "cum_confirmed"),
        ("-1", "cum_confirmed"),
        ("NA", "cum_deaths"),
        ("absent", "cum_deaths"),
    ],
)
def test_parse_value_rejects(text: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        parse_value(text, field)


def test_read_review_names_the_bad_line(tmp_path: Path) -> None:
    path = write_review(
        tmp_path / "r.csv",
        [row(A, "cum_confirmed", "10"), row(A, "cum_deaths", "two")],
    )
    with pytest.raises(ValueError, match=r"r\.csv line 3: .*'two'"):
        read_review(path)


# Applying a review


@pytest.mark.parametrize(
    "reason",
    [
        "models_disagree",
        "text_mismatch",
        "row_not_located",
        "duplicate_row",
        "sum_mismatch",
        "wrong_province",
    ],
)
def test_apply_sets_and_verifies_the_cell(reason: str, gazetteer) -> None:
    accepted = {**TRUTH, A: {"cum_confirmed": 99}}
    values, verified = apply(
        accepted,
        [row(A, "cum_confirmed", "10", reason), row(A, "community_deaths", "blank")],
        gazetteer=gazetteer,
    )
    assert values[A] == {"cum_confirmed": 10, "community_deaths": None}
    assert verified == {(A, "cum_confirmed"), (A, "community_deaths")}
    assert accepted[A] == {"cum_confirmed": 99}


def test_apply_creates_a_missing_row_and_skips_unfilled(gazetteer) -> None:
    accepted = {k: v for k, v in TRUTH.items() if k != KPIS}
    review_ = [
        row(KPIS, "cum_confirmed", "30", "section_missing"),
        row(KPIS, "cum_deaths", "", "section_missing"),
        row(RowKey("zones", "Ituri", ""), "", "", "section_missing"),
    ]
    values, verified = apply(accepted, review_, gazetteer=gazetteer)
    assert values[KPIS] == {"cum_confirmed": 30}
    assert verified == {(KPIS, "cum_confirmed")}


def test_added_rows_need_a_known_name(tmp_path: Path, gazetteer) -> None:
    typed = row(A, "cum_deaths", "2", "")
    other = row(RowKey("zones", "Nord-Kivu", "D"), "cum_deaths", "", "")

    values, verified = apply({}, [typed], gazetteer=gazetteer)
    assert values == {A: {"cum_deaths": 2}}
    assert verified == {(A, "cum_deaths")}
    review.check_added([typed], gazetteer=gazetteer)
    with pytest.raises(ValueError, match="Nord-Kivu / D is not in the gazetteer"):
        review.check_added([typed, other], gazetteer=gazetteer)
    placeholder = row(RowKey("zones", "Ituri", ""), "", "12", "section_missing")
    path = write_review(tmp_path / "r.csv", [placeholder])
    with pytest.raises(ValueError, match="'done'"):
        read_review(path)


def test_apply_moves_a_zone_to_the_gazetteer_province(gazetteer) -> None:
    printed = RowKey("zones", "Nord-Kivu", "A")
    values, verified = apply(
        TRUTH, [row(printed, "cum_deaths", "2", "wrong_province")], gazetteer=gazetteer
    )
    assert verified == {(A, "cum_deaths")}
    assert printed not in values


def test_apply_resolves_an_unknown_name_after_an_alias(tmp_path: Path) -> None:
    printed = RowKey("zones", "Ituri", "Bee")
    reviewed = [row(printed, f, "1", "unknown_name") for f in ("new_confirmed",)]
    before = load_gazetteer(write_gazetteer(tmp_path / "before.csv"))
    with pytest.raises(ValueError, match="Ituri / Bee is not in the gazetteer"):
        apply(TRUTH, reviewed, gazetteer=before)

    after = load_gazetteer(
        write_gazetteer(tmp_path / "after.csv", "Ituri,A,\nIturi,B,Bee|Bé\n")
    )
    values, verified = apply(TRUTH, reviewed, gazetteer=after)
    assert verified == {(B, "new_confirmed")}
    assert values[B]["new_confirmed"] == 1
    assert printed not in values


def test_apply_drops_a_row_dismissed_as_absent(gazetteer) -> None:
    garbled = RowKey("zones", "Ituri", "Xyz")
    accepted = {**TRUTH, garbled: {}}
    dismiss = [row(garbled, f, "absent", "unknown_name") for f in FIELDS["zones"]]

    values, verified = apply(accepted, dismiss, gazetteer=gazetteer)
    assert values == TRUTH
    assert verified == set()

    with pytest.raises(ValueError, match="also given values"):
        apply(
            accepted,
            [*dismiss[1:], row(garbled, "cum_confirmed", "1")],
            gazetteer=gazetteer,
        )
    real = [row(A, "cum_deaths", "absent", "unknown_name")]
    with pytest.raises(ValueError, match="real zone"):
        apply(TRUTH, real, gazetteer=gazetteer)
    duplicate = [row(A, "cum_deaths", "absent", "duplicate_row")]
    with pytest.raises(ValueError, match="only for rows flagged unknown_name"):
        apply(TRUTH, duplicate, gazetteer=gazetteer)


def test_apply_rejects_a_cell_reviewed_twice_differently(gazetteer) -> None:
    with pytest.raises(ValueError, match="reviewed twice"):
        apply(
            TRUTH,
            [row(A, "cum_deaths", "2"), row(A, "cum_deaths", "3")],
            gazetteer=gazetteer,
        )


# Issues


def test_issue_body() -> None:
    held = [
        HeldReport(DATE, SHA, 8, "data/review/drc_2026/2026-10-14_abababab.csv", PDF),
        HeldReport("2026-10-15", "cd" * 32, 1, "data/review/x.csv", "https://x/2.pdf"),
    ]
    body = issue_body(held, run_url="https://github.com/o/r/actions/runs/1")

    assert body.startswith("@mharoruiz @jm-rivera")
    assert f"| {DATE} | 8 | `data/review/drc_2026/2026-10-14_abababab.csv` |" in body
    assert "[PDF](https://x/2.pdf)" in body
    assert "`Refs #N`" in body
    assert "Do not write `Closes`" in body
    assert "aliases" in body
    assert "https://github.com/o/r/actions/runs/1" in body


@pytest.fixture
def gh_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("shell") is not True
        body = args[args.index("--body-file") + 1] if "--body-file" in args else None
        calls.append([*args, Path(body).read_text() if body else ""])
        return subprocess.CompletedProcess(
            args, 0, "https://github.com/o/r/issues/42\n"
        )

    monkeypatch.setattr(review.subprocess, "run", run)
    return calls


@pytest.mark.parametrize(("actions", "dry_run"), [(False, False), (True, True)])
def test_gh_is_skipped_outside_actions_or_in_dry_run(
    gh_calls: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    actions: bool,
    dry_run: bool,
) -> None:
    if actions:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")

    assert review.create_issue(title="t", body="b", dry_run=dry_run) is None
    assert review.open_issues(dry_run=dry_run) == {}
    review.comment_issue(1, body="b", dry_run=dry_run)
    review.close_issue(1, comment="c", dry_run=dry_run)
    assert gh_calls == []


def test_gh_runs_in_actions(
    gh_calls: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    body = '$(rm -rf /) `x` "quoted"'

    assert review.create_issue(title="Review; echo", body=body) == 42
    review.close_issue(42, comment="Released.")

    assert [c[:3] for c in gh_calls] == [
        ["gh", "label", "create"],
        ["gh", "issue", "create"],
        ["gh", "issue", "comment"],
        ["gh", "issue", "close"],
    ]
    assert "--force" in gh_calls[0]
    assert gh_calls[1][3:7] == ["--title", "Review; echo", "--label", review.LABEL]
    assert gh_calls[1][-1] == body
    assert gh_calls[2][-1] == "Released."


# Pull request check (drc_2026_sitreps_extract.py check)


@pytest.fixture
def check_pr(monkeypatch: pytest.MonkeyPatch):
    """Run the review check on a repository, reading the stub text layer."""
    monkeypatch.setattr(ex, "MATCH", match)
    monkeypatch.setattr(ex, "read_rows", lambda words, gazetteer: text_layer())

    def run(repo: Path) -> list[str]:
        return ex.check_review(base="HEAD", repo=repo)

    return run


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(
    tmp_path: Path,
    gemini: dict[RowKey, Values],
    mistral: dict[RowKey, Values],
    *,
    status: Status = "held",
) -> tuple[Path, Path]:
    """A repository holding one report in its manifest and ledger.

    The report has its cache, the gazetteer and, if anything is flagged, its
    review file.
    """
    repo = tmp_path / "repo"
    manifest = repo / MANIFEST_PATH.relative_to(ROOT)
    manifest.parent.mkdir(parents=True)
    upload = {
        "report_date": date.fromisoformat(DATE),
        "sitrep_number": 152,
        "is_latest": True,
        "media_id": 1,
        "modified": "t1",
        "source_url": PDF,
    }
    pl.DataFrame(
        [{c: upload.get(c) for c in MANIFEST_SCHEMA}], schema=MANIFEST_SCHEMA
    ).write_csv(manifest)
    ledger = ex.LedgerRow(date.fromisoformat(DATE), 1, "t1", SHA, status, *[None] * 7)
    ex.ledger_frame({(1, "t1"): ledger}).write_csv(
        repo / ex.STATUS_PATH.relative_to(ROOT)
    )
    cache = repo / CACHE_DIR.relative_to(ROOT)
    gazetteer = load_gazetteer(write_gazetteer(repo / GAZETTEER_PATH.relative_to(ROOT)))
    for name, rows in (("gemini", gemini), ("mistral", mistral)):
        result = ModelResult(raw(rows), name, name, {}, "2026-10-15T12:00:00Z", {})
        write_model_result(SHA, name, result, root=cache)
    # The check reads the stub text layer in place of these words.
    words = tmp_path / "words.json"
    words.write_text("{}\n")
    write_words(SHA, words, root=cache)
    (repo / "README.md").write_text("Readme\n")

    reconciled = reconcile(
        normalise(raw(gemini), gazetteer),
        normalise(raw(mistral), gazetteer),
        text_layer(),
        match=match,
    )
    sum_flags, _ = check_sums(reconciled.accepted, set())
    path = review.review_path(DATE, SHA, root=repo / REVIEW_DIR.relative_to(ROOT))
    if flags := [*reconciled.flags, *sum_flags]:
        write_review(
            path,
            review_rows(
                flags,
                report_date=DATE,
                sha256=SHA,
                pdf_url=PDF,
                text_rows=text_layer(),
            ),
        )

    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "held")
    return repo, path


def printed(value: float | None) -> str:
    return "blank" if value is None else str(value)


def fill(path: Path, **overrides: str) -> None:
    """Fill empty cells from the truth and mark placeholders done.

    `overrides` maps zone_field to the text to type instead.
    """
    rows = read_review(path)
    for r in rows:
        cell = f"{r['zone']}_{r['field']}"
        if not r["field"]:
            r["reviewed_value"] = review.DONE
        elif cell in overrides or not r["reviewed_value"]:
            key = RowKey(r["section"], r["province"], r["zone"])
            r["reviewed_value"] = (
                overrides[cell]
                if cell in overrides
                else printed(TRUTH[key][r["field"]])
            )
    write_review(path, rows)


def wrong_twice() -> tuple[dict[RowKey, Values], dict[RowKey, Values]]:
    """Both models misread zone A's deaths; the models disagree on one KPI."""
    gemini = {**TRUTH, A: {**TRUTH[A], "cum_deaths": 3}}
    mistral = {**gemini, KPIS: {**TRUTH[KPIS], "in_isolation": 5}}
    return gemini, mistral


def test_check_passes_a_filled_review(tmp_path: Path, check_pr: Any) -> None:
    repo, path = make_repo(tmp_path, *wrong_twice())
    assert len(read_review(path)) == 8
    fill(path)
    assert check_pr(repo) == []


def added(key: RowKey, printed_as: RowKey | None = None) -> list[ReviewRow]:
    """The rows a reviewer types in for a missing row, as `printed_as`."""
    return [
        {
            **row(printed_as or key, f, printed(v), ""),
            "report_date": "",
            "sha256": "",
            "page": None,
            "pdf_url": "",
        }
        for f, v in TRUTH[key].items()
    ]


def test_check_passes_rows_added_for_cells_a_sum_opens(
    tmp_path: Path, check_pr: Any
) -> None:
    repo, path = make_repo(tmp_path, *wrong_twice())
    # The PDF prints 11 cases for A. Typing that in runs the Ituri subtotal
    # sum, which fails and opens cells the review file has no row for.
    fill(path, A_cum_confirmed="11")
    assert [e.split(": ", 1)[1].split(" is open")[0] for e in check_pr(repo)] == [
        "zone_subtotals / Ituri cum_confirmed",
        "zones / Ituri / B cum_confirmed",
    ]

    # The reviewer adds rows for them in the same pull request.
    write_review(
        path,
        [
            *read_review(path),
            row(ITURI, "cum_confirmed", "30", ""),
            row(B, "cum_confirmed", "20", ""),
        ],
    )
    assert check_pr(repo) == []
    cache = repo / CACHE_DIR.relative_to(ROOT)
    verdict = ex.judge(
        ex.read_records(SHA, root=cache),
        gazetteer=load_gazetteer(repo / GAZETTEER_PATH.relative_to(ROOT)),
        review_file=path,
    )
    assert (verdict.open_flags, verdict.unapplied) == ([], ())
    # Released as printed: the subtotal sum, and A's CFR (2 of 11 is not 20.0).
    assert {(d.key, d.field) for d in verdict.discrepancies} == {
        (ITURI, "cum_confirmed"),
        (A, "cfr_pct"),
    }


@pytest.mark.parametrize(
    ("missing", "placeholders", "typed"),
    [
        pytest.param(
            [ITURI_PROVINCE],
            [("provinces", "")],
            added(ITURI_PROVINCE),
            id="whole-section",
        ),
        pytest.param(
            [A, B],
            # With no zone rows at all, the section is missing too.
            [("zones", ""), ("zones", "Ituri")],
            # B typed as printed, under a spelling the PR adds as an alias.
            [*added(A), *added(B, RowKey("zones", "Ituri", "Bee"))],
            id="province-zones-with-alias",
        ),
    ],
)
def test_check_passes_a_section_typed_in(
    tmp_path: Path,
    missing: list[RowKey],
    placeholders: list[tuple[str, str]],
    typed: list[ReviewRow],
    check_pr: Any,
) -> None:
    without = {k: v for k, v in TRUTH.items() if k not in missing}
    repo, path = make_repo(tmp_path, without, without)
    rows = read_review(path)
    assert [
        (r["section"], r["province"]) for r in rows if not r["field"]
    ] == placeholders
    gazetteer = repo / GAZETTEER_PATH.relative_to(ROOT)
    write_gazetteer(gazetteer, "Ituri,A,\nIturi,B,Bee\n")

    write_review(path, [*rows, *typed])
    assert check_pr(repo) == []


def test_check_passes_a_missing_row_filled_field_by_field(
    tmp_path: Path, check_pr: Any
) -> None:
    without = {k: v for k, v in TRUTH.items() if k != TOTAL}
    repo, path = make_repo(tmp_path, without, without)
    rows = read_review(path)
    assert {(r["section"], r["reason"]) for r in rows} == {
        ("zone_total", "section_missing")
    }
    assert len(rows) == len(FIELDS["zone_total"])
    fill(path)
    assert check_pr(repo) == []


def test_check_passes_a_garbled_zone_dismissed_as_absent(
    tmp_path: Path, check_pr: Any
) -> None:
    gemini, mistral = wrong_twice()
    garbled = RowKey("zones", "Ituri", "Xyz")
    extra = {garbled: {**TRUTH[A], "cum_confirmed": 3}}
    repo, path = make_repo(tmp_path, {**gemini, **extra}, {**mistral, **extra})
    assert {r["reason"] for r in read_review(path) if r["zone"] == "Xyz"} == {
        "unknown_name"
    }

    fill(path, **{f"Xyz_{f}": "absent" for f in FIELDS["zones"]})
    assert check_pr(repo) == []


def edit_column(path: Path, column: str, value: str) -> None:
    with path.open() as f:
        records = list(csv.DictReader(f))
    records[0][column] = value
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            lambda repo, path: fill(path, A_cum_deaths=""),
            r"zones / Ituri / A cum_deaths is open \(text_mismatch\): type the "
            r"value printed in the PDF",
            id="unfilled",
        ),
        pytest.param(
            lambda repo, path: (repo / "README.md").write_text("changed\n"),
            r"README\.md: a review pull request may change only",
            id="other-file",
        ),
        pytest.param(
            lambda repo, path: edit_column(path, "page", "9"),
            r"changed a column other than reviewed_value",
            id="other-column",
        ),
        pytest.param(
            lambda repo, path: fill(path, A_cum_confirmed="11"),
            r"zone_subtotals / Ituri cum_confirmed is open \(sum_mismatch\): the "
            r"table sums do not hold",
            id="failing-sum",
        ),
        pytest.param(
            lambda repo, path: fill(path, _in_isolation="absent"),
            r"line \d+: reviewed_value 'absent' for in_isolation is only for rows "
            r"flagged "
            r"unknown_name",
            id="absent-on-models-disagree",
        ),
        pytest.param(
            lambda repo, path: write_review(
                path, [*read_review(path), row(B, "cum_deaths", "5", "")]
            ),
            r"zones / Ituri / B cum_deaths is not a cell under review; remove the row",
            id="added-row-not-under-review",
        ),
    ],
)
def test_check_fails(tmp_path: Path, check_pr: Any, change: Any, message: str) -> None:
    repo, path = make_repo(tmp_path, *wrong_twice())
    fill(path)
    change(repo, path)
    errors = check_pr(repo)
    assert any(re.search(message, e) for e in errors), errors


@pytest.mark.parametrize("typed", ["", "99"])
def test_check_ignores_a_row_no_longer_flagged(
    tmp_path: Path, check_pr: Any, typed: str
) -> None:
    repo, path = make_repo(tmp_path, *wrong_twice())
    # Release appended a row for a cell that is no longer flagged.
    stale = row(B, "cum_confirmed", "")
    write_review(path, [*read_review(path), stale])
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "grow")

    fill(path, B_cum_confirmed=typed)
    assert read_review(path)[-1]["reviewed_value"] == typed
    assert check_pr(repo) == []


def test_check_fails_a_gazetteer_edit_that_unreleases_a_report(
    tmp_path: Path, check_pr: Any
) -> None:
    repo, path = make_repo(tmp_path, TRUTH, TRUTH, status="released")
    assert not path.exists()
    assert check_pr(repo) == []

    # Moving zone A to another province flags the released report's row.
    write_gazetteer(repo / GAZETTEER_PATH.relative_to(ROOT), "Tshopo,A,\nIturi,B,\n")
    errors = check_pr(repo)
    assert (
        f"{DATE} upload (media 1, modified t1) was released and would now be held"
        in errors
    )
    assert any(e.startswith("Release would fail: Dates would drop out") for e in errors)
