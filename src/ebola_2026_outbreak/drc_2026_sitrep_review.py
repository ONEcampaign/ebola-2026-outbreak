"""Hold flagged sitrep values for a person to review, and check the review.

A report with any flagged value is held. Its flags become a review CSV,
`data/review/drc_2026/<report_date>_<sha8>.csv`, one row per flagged cell,
naming the row, the field, the PDF page and the reason, with an empty
`reviewed_value`. The models' values are not shown, so the reviewer reads the
PDF instead of confirming a guess. A flag with no field (a whole section, or
a province's zone rows, is missing) becomes one placeholder row with an empty
`field`. The reviewer adds one row per cell of that section and province,
with an empty `reason`; the placeholder's `reviewed_value` may be set to
`done`. A row flagged `unknown_name` that the table does not print (a
misread zone name) is dismissed by typing `absent` in its cells. A missing
row of a known shape (the headline indicators, the zone table's Total, a
province subtotal) is flagged per field, so its cells are ordinary rows to
fill.

The reviewer types each value as printed (a whole number, a decimal with a
point, or `blank` for an empty cell) and opens a pull request. An unknown
zone name is settled by adding it, or an alias, to the gazetteer in the same
pull request. `apply` merges the reviewed values into the accepted ones.
Every reviewed cell counts as verified for `check_sums`, so a sum that fails
on verified values only is a publisher discrepancy, released as printed.

`check_diff` holds a review pull request to its shape: it may change only
`reviewed_value` cells of existing review CSVs (and add rows under a
placeholder) and add or edit rows of the gazetteer, and every value typed
must read. Whether the reviewed report would then be released is decided by
`drc_2026_sitreps_extract.py check`, through the same function release uses.

One issue per run lists the held reports and tags the reviewers. Issues are
opened, commented on and closed with `gh`, only inside GitHub Actions. Each
issue carries the label `sitrep-review` and a hidden marker per upload it
tracks (`held_marker`, `pending_marker`), so a run that finds an open issue
already tracking an upload adopts it instead of opening another, whether or
not the ledger recorded its number.
"""

import csv
import json
import logging
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple, TypedDict

from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    FIELDS,
    GAZETTEER_PATH,
    PERCENT_FIELDS,
    REVIEW_DIR,
    SECTIONS,
    Flag,
    Gazetteer,
    HealthZone,
    Number,
    RowKey,
    Section,
    TextRow,
    Values,
    name_key,
    normalise,
)
from ebola_2026_outbreak.drc_2026_sitrep_validate import Cell
from ebola_2026_outbreak.drc_2026_sitreps_download import ROOT

logger = logging.getLogger(__name__)

REVIEWERS = ("@mharoruiz", "@jm-rivera")
COLUMNS = (
    "report_date",
    "sha256",
    "section",
    "province",
    "zone",
    "field",
    "page",
    "pdf_url",
    "reason",
    "reviewed_value",
)
BLANK = "blank"
# `reviewed_value` of a flagged row that is not in the PDF's table at all.
ABSENT = "absent"
# `reviewed_value` of a placeholder whose section the reviewer has typed in.
DONE = "done"
# Label of every issue the pipeline opens, and what it searches for.
LABEL = "sitrep-review"
_INTEGER = re.compile(r"\d+")
_DECIMAL = re.compile(r"\d+\.\d+")


class ReviewRow(TypedDict):
    """One row of a review CSV: one flagged cell, or a missing section.

    `field` is "" for a missing section's placeholder. A row the reviewer
    added under a placeholder has `reason` "". `reviewed_value` is the text
    the reviewer typed, "" until filled.
    """

    report_date: str
    sha256: str
    section: Section
    province: str
    zone: str
    field: str
    page: int | None
    pdf_url: str
    reason: str
    reviewed_value: str


class HeldReport(NamedTuple):
    """A report held for review, as listed in the run's issue.

    `review_path` is relative to the repository root.
    """

    report_date: str
    sha256: str
    cells: int
    review_path: str
    pdf_url: str


# Review files


def review_path(report_date: str, sha256: str, *, root: Path = REVIEW_DIR) -> Path:
    """Path of a report's review CSV."""
    return root / f"{report_date}_{sha256[:8]}.csv"


def review_rows(
    flags: Iterable[Flag],
    *,
    report_date: str,
    sha256: str,
    pdf_url: str,
    text_rows: dict[RowKey, TextRow],
) -> list[ReviewRow]:
    """One review row per flagged cell, in table order.

    A cell flagged twice gets one row, with the first flag's reason. A flag
    with no page takes its row's page from the text layer, and the PDF link
    opens at that page.

    Args:
        flags: From `reconcile` and `check_sums`.
        report_date: The report's date, `YYYY-MM-DD`.
        sha256: The PDF's sha256.
        pdf_url: The PDF's source URL.
        text_rows: Text-layer rows by key, for pages.

    Returns:
        Rows with empty `reviewed_value`.
    """
    cells: dict[tuple[RowKey, str], ReviewRow] = {}
    for flag in flags:
        field = flag.field or ""
        if (flag.key, field) in cells:
            continue
        page = flag.page
        if page is None and (text := text_rows.get(flag.key)) is not None:
            page = text.page
        cells[flag.key, field] = {
            "report_date": report_date,
            "sha256": sha256,
            "section": flag.key.section,
            "province": flag.key.province,
            "zone": flag.key.zone,
            "field": field,
            "page": page,
            "pdf_url": f"{pdf_url}#page={page}" if page else pdf_url,
            "reason": flag.reason,
            "reviewed_value": "",
        }

    def order(cell: tuple[RowKey, str]) -> tuple[int, str, str, int]:
        key, field = cell
        fields = FIELDS[key.section]
        return (
            SECTIONS.index(key.section),
            key.province,
            key.zone,
            fields.index(field) if field in fields else -1,
        )

    return [cells[c] for c in sorted(cells, key=order)]


def write_review(path: Path, rows: Iterable[ReviewRow]) -> Path:
    """Write review rows as CSV, creating the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "page": "" if row["page"] is None else row["page"]})
    return path


def read_review(path: Path) -> list[ReviewRow]:
    """Read and validate a review CSV.

    Raises:
        ValueError: If the columns differ from `COLUMNS`, or a row names an
            unknown section or field, has a page that is not a number, has a
            `reviewed_value` that is not a whole number, a decimal with a
            point, or `blank` (`done` on a placeholder), or was added (has no
            reason) for a section and province with no placeholder. The
            message names the file and line.
    """
    records = _records(path.read_text(), path.name)
    rows: list[ReviewRow] = []
    for line, record in records:
        try:
            rows.append(_review_row(record))
        except ValueError as e:
            raise ValueError(f"{path.name} line {line}: {e}") from None
    for i in _unplaced(rows):
        raise ValueError(f"{path.name} line {records[i][0]}: {_UNPLACED}")
    return rows


_UNPLACED = (
    "a row without a reason may be added only for a section and province "
    "whose placeholder row (empty field) is in the file"
)


def _unplaced(rows: list[ReviewRow]) -> list[int]:
    """Indices of added rows with no placeholder for their section and province."""
    placeholders = [(r["section"], r["province"]) for r in rows if not r["field"]]
    return [
        i
        for i, r in enumerate(rows)
        if r["field"]
        and not r["reason"]
        and (r["section"], "") not in placeholders
        and (r["section"], r["province"]) not in placeholders
    ]


def _records(text: str, name: str) -> list[tuple[int, dict[str, str]]]:
    """A review CSV's rows as raw strings, with their line numbers."""
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(f"{name}: columns must be {', '.join(COLUMNS)}")
    return [(i, record) for i, record in enumerate(reader, start=2)]


def _review_row(record: dict[str, str]) -> ReviewRow:
    section = record["section"]
    if section not in SECTIONS:
        raise ValueError(f"unknown section {section!r}")
    field = record["field"]
    if field and field not in FIELDS[section]:
        raise ValueError(f"unknown field {field!r} for section {section}")
    page = record["page"].strip()
    if page and not _INTEGER.fullmatch(page):
        raise ValueError(f"page {page!r} is not a number")
    row: ReviewRow = {
        "report_date": record["report_date"],
        "sha256": record["sha256"],
        "section": section,
        "province": record["province"],
        "zone": record["zone"],
        "field": field,
        "page": int(page) if page else None,
        "pdf_url": record["pdf_url"],
        "reason": record["reason"],
        "reviewed_value": record["reviewed_value"].strip(),
    }
    if not field and row["reviewed_value"] not in ("", DONE):
        raise ValueError(f"a placeholder's reviewed_value must be {DONE!r} or empty")
    if row["reviewed_value"] and field and not _dismissed(row):
        parse_value(row["reviewed_value"], field)
    return row


def parse_value(text: str, field: str) -> Number | None:
    """The value a reviewer typed for a field.

    Args:
        text: A whole number ("1234"), a decimal with a point ("48.3"), or
            "blank" for a cell empty in the PDF.
        field: The field it fills; a count takes only whole numbers.

    Raises:
        ValueError: For anything else, such as "48,3", "1 234" or "", and
            for "absent", which dismisses a row instead of giving a value.
    """
    if text.casefold() == BLANK:
        return None
    if text.casefold() == ABSENT:
        raise ValueError(
            f"reviewed_value {ABSENT!r} for {field} is only for rows flagged "
            "unknown_name; type the value printed in the PDF"
        )
    if field in PERCENT_FIELDS and (
        _INTEGER.fullmatch(text) or _DECIMAL.fullmatch(text)
    ):
        return float(text)
    if _INTEGER.fullmatch(text):
        return int(text)
    expected = "a number" if field in PERCENT_FIELDS else "a whole number"
    raise ValueError(
        f"reviewed_value {text!r} for {field} must be {expected}, or {BLANK!r}"
    )


# Applying a review


def apply(
    reconciled_values: dict[RowKey, Values],
    review: Iterable[ReviewRow],
    *,
    gazetteer: Gazetteer,
) -> tuple[dict[RowKey, Values], set[Cell]]:
    """Merge a review's values into the accepted values.

    Each filled row with a field sets that cell to the reviewer's value,
    replacing any accepted value and creating the row if it is absent: a
    row flagged `section_missing` field by field, or a row the reviewer
    added under a placeholder. Placeholders and rows with an empty
    `reviewed_value` are skipped. Every cell set is verified.

    A row flagged `unknown_name` whose cells read `absent` is not in the PDF:
    its key, as printed in the review, is dropped from the values and none of
    its cells is verified, so `check_sums` runs on the rest. A row the
    gazetteer resolves is a real zone and cannot be dismissed. A
    `duplicate_row` row takes the real row's values. `absent` there would
    drop the canonical row and is refused.

    A row's names are re-normalised with `gazetteer`, the way `normalise`
    reads a model's row, so an `unknown_name` row whose zone the reviewer has
    since added to the gazetteer lands under the canonical zone and province,
    and a `wrong_province` row lands under the gazetteer's province. The
    gazetteer decides the row, the reviewer the value.

    Args:
        reconciled_values: `Reconciled.accepted`, from a `reconcile` run
            with the same gazetteer.
        review: From `read_review`.
        gazetteer: From `load_gazetteer`.

    Returns:
        The merged values and the verified cells, for `check_sums`.

    Raises:
        ValueError: If a row's name does not resolve in the gazetteer, an
            added row has no placeholder, a row is both dismissed and given
            values, a resolved name is dismissed, `absent` is typed on a row
            with another reason, or two rows give one cell different values.
    """
    review = list(review)
    for i in _unplaced(review):
        raise ValueError(f"{describe(_key(review[i]))}: {_UNPLACED}")
    dismissed = {_key(row) for row in review if _dismissed(row)}
    for row in review:
        if _dismissed(row) and resolve(_key(row), gazetteer) is not None:
            raise ValueError(
                f"{describe(_key(row))} is in the gazetteer, so it is a real zone; "
                f"type its values instead of {ABSENT!r}"
            )
    values = {
        key: dict(row) for key, row in reconciled_values.items() if key not in dismissed
    }
    verified: dict[Cell, Number | None] = {}
    for row in review:
        field = row["field"]
        if not field or not row["reviewed_value"] or _dismissed(row):
            continue
        printed = _key(row)
        key = resolve(printed, gazetteer)
        if printed in dismissed or key in dismissed:
            raise ValueError(
                f"{describe(printed)} is marked {ABSENT!r} and also given values"
            )
        if key is None:
            raise ValueError(
                f"{describe(printed)} is not in the gazetteer: add the zone, or "
                f"its spelling as an alias, to {GAZETTEER_PATH.name}, or type "
                f"{ABSENT!r} if the table does not print it"
            )
        value = parse_value(row["reviewed_value"], field)
        if verified.get((key, field), value) != value:
            raise ValueError(f"{describe(key)} {field} is reviewed twice")
        verified[key, field] = value
        values.setdefault(key, {})[field] = value
    return values, set(verified)


def resolve(key: RowKey, gazetteer: Gazetteer) -> RowKey | None:
    """The canonical key of a row, read as `normalise` reads a model's row.

    Returns None when a province or zone name is unknown, or a zone name
    belongs to several provinces, none of them the printed one.
    """
    if key.section in ("kpis", "zone_total"):
        return key if not key.province and not key.zone else None
    names = {"province": key.province}
    if key.section == "zones":
        names["zone"] = key.zone
    raw = {
        "sitrep_no": None,
        "report_date": None,
        "kpis": None,
        "zone_total": None,
        "provinces": [],
        "zone_subtotals": [],
        "zones": [],
        key.section: [{**names, **dict.fromkeys(FIELDS[key.section])}],
    }
    normalised = normalise(raw, gazetteer)
    if len(normalised.extraction.rows) != 1:
        # A Total row, which `normalise` drops.
        return None
    (canonical,) = normalised.extraction.rows
    if canonical.section == "zones":
        # A wrong_province flag still resolves when the gazetteer names one
        # province for the zone; `normalise` has already keyed it there.
        zone = HealthZone(canonical.province, canonical.zone)
        return canonical if zone in gazetteer.get(name_key(zone.zone), ()) else None
    return None if normalised.name_flags else canonical


def _dismissed(row: ReviewRow) -> bool:
    """Whether a flagged row's cell says the row is not in the PDF."""
    return (
        bool(row["field"])
        and row["reviewed_value"].casefold() == ABSENT
        and row["reason"] == "unknown_name"
    )


def _key(row: ReviewRow) -> RowKey:
    return RowKey(row["section"], row["province"], row["zone"])


def describe(key: RowKey) -> str:
    """A row key as a reader names it: "zones / Tshopo / Makiso-Kisangani"."""
    return " / ".join(part for part in key if part)


# Issues


def issue_title(count: int) -> str:
    """Title of a run's issue."""
    noun = "sitrep" if count == 1 else "sitreps"
    return f"Review {count} held DRC {noun}"


def issue_body(held: Sequence[HeldReport], *, run_url: str) -> str:
    """The body of the issue listing a run's held reports.

    Args:
        held: The reports held in the run.
        run_url: The Actions run that held them.
    """
    count = f"{len(held)} sitrep {'report is' if len(held) == 1 else 'reports are'}"
    lines = [
        f"{' '.join(REVIEWERS)}: {count} held. Each has "
        "values the two models read differently, or that the PDF text or the "
        "table sums do not confirm. They are released once a person has typed "
        "those values from the PDF.",
        "",
        "| Report date | Cells | Review file | PDF |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.report_date} | {r.cells} | `{r.review_path}` | [PDF]({r.pdf_url}) |"
        for r in held
    ]
    lines += [
        "",
        "To review a report:",
        "",
        "1. Open its review file. Each row is one cell of the report. Open the "
        "PDF at the page in `pdf_url`, find the row and column, and type the "
        "value printed in `reviewed_value`: a whole number without spaces "
        f"(`1234`), a decimal with a point (`48.3`), or `{BLANK}` when the cell "
        "is empty. Change no other column.",
        "2. A row with an empty `field` means the whole section is missing, or, "
        "for `zones` with a province, that province's zone rows. Add one row "
        "per zone and field below it, filling `section`, `province`, `zone`, "
        "`field` and `reviewed_value` and leaving `reason` empty.",
        "3. A row with reason `unknown_name` names a zone the gazetteer does not "
        "know. Add the zone, or this spelling to its `aliases` (separated by "
        f"`|`), in `data/reference/{GAZETTEER_PATH.name}`, and fill its values. "
        f"If the name is a misreading and the table has no such row, type "
        f"`{ABSENT}` in each of its cells instead. A zone the misreading hid "
        "shows up as its own flag.",
        "4. Open a pull request and write `Refs #N` in its description, where N "
        "is this issue's number. Do not write `Closes`, because this issue "
        "closes itself once every report in it is released.",
        "5. The pull request's check confirms the report would be released "
        "with the values typed, meaning every open cell is filled and the table "
        "sums hold. A row whose cell is not flagged needs no value. Merging "
        "the pull request releases the report.",
        "",
        f"Run: {run_url}",
        "",
        *(held_marker(r.sha256) for r in held),
    ]
    return "\n".join(lines) + "\n"


def held_marker(sha256: str) -> str:
    """The hidden line marking an issue as tracking a held report's PDF."""
    return f"<!-- sitrep-upload:{sha256} -->"


def pending_marker(media_id: int, modified: str) -> str:
    """The hidden line marking an issue as tracking a pending upload."""
    return f"<!-- sitrep-pending:{media_id}:{modified} -->"


def _gh(args: list[str], *, body: str | None, dry_run: bool) -> str | None:
    """Run `gh` with a body passed as a file, only inside GitHub Actions.

    Returns gh's output, or None when skipped.
    """
    if dry_run or not os.environ.get("GITHUB_ACTIONS"):
        logger.info("Skipping gh %s", " ".join(args[:2]))
        return None
    with tempfile.TemporaryDirectory() as tmp:
        if body is not None:
            body_path = Path(tmp) / "body.md"
            body_path.write_text(body)
            args = [*args, "--body-file", str(body_path)]
        result = subprocess.run(
            ["gh", *args], check=True, capture_output=True, text=True
        )
    return result.stdout.strip()


def open_issues(*, dry_run: bool = False) -> dict[int, str]:
    """The bodies of open issues labelled `LABEL`, by number; empty when skipped."""
    out = _gh(
        [
            "issue",
            "list",
            "--label",
            LABEL,
            "--state",
            "open",
            "--json",
            "number,body",
            "--limit",
            "1000",
        ],
        body=None,
        dry_run=dry_run,
    )
    return {} if out is None else {i["number"]: i["body"] for i in json.loads(out)}


def create_issue(*, title: str, body: str, dry_run: bool = False) -> int | None:
    """Open an issue labelled `LABEL` and return its number, or None when skipped.

    `gh issue create --label` refuses a label the repository lacks, so the
    label is created first. `--force` makes that a no-op when it exists.
    """
    _gh(
        ["label", "create", LABEL, "--force", "--description", "DRC sitrep review"],
        body=None,
        dry_run=dry_run,
    )
    url = _gh(
        ["issue", "create", "--title", title, "--label", LABEL],
        body=body,
        dry_run=dry_run,
    )
    return None if url is None else int(url.rstrip("/").rsplit("/", 1)[1])


def comment_issue(number: int, *, body: str, dry_run: bool = False) -> None:
    """Comment on an issue."""
    _gh(["issue", "comment", str(number)], body=body, dry_run=dry_run)


def close_issue(number: int, *, comment: str, dry_run: bool = False) -> None:
    """Comment on an issue, then close it."""
    comment_issue(number, body=comment, dry_run=dry_run)
    _gh(["issue", "close", str(number)], body=None, dry_run=dry_run)


# Pull request check


class ReviewDiff(NamedTuple):
    """A review pull request's changes against its base.

    `reviews` are the changed review files, as paths under the repository.
    `errors` holds one message per rule the changes break. `merge_base` is
    the commit the changes are measured from.
    """

    reviews: list[Path]
    errors: list[str]
    merge_base: str


def check_diff(*, base: str = "origin/main", repo: Path = ROOT) -> ReviewDiff:
    """Hold a review pull request's working tree to the review file rules.

    Diffs tracked files against the merge base of `base` and HEAD. A pull
    request may change only `reviewed_value` cells of existing review files,
    add rows under a placeholder, and add or edit gazetteer rows; each review
    file must read with `read_review`.

    Args:
        base: The branch the pull request merges into.
        repo: The repository root.
    """
    merge_base = git(repo, "merge-base", base, "HEAD")
    changes = [
        line.split("\t", 1)
        for line in git(
            repo, "diff", "--name-status", "--no-renames", merge_base
        ).splitlines()
    ]
    review_dir = REVIEW_DIR.relative_to(ROOT).as_posix()
    gazetteer_file = GAZETTEER_PATH.relative_to(ROOT).as_posix()

    errors: list[str] = []
    reviews: list[Path] = []
    for status, path in changes:
        if status == "M" and path == gazetteer_file:
            errors += _check_gazetteer(repo, merge_base, path)
        elif (
            status == "M"
            and path.startswith(f"{review_dir}/")
            and path.endswith(".csv")
        ):
            errors += _check_review(repo, merge_base, path)
            reviews.append(repo / path)
        else:
            errors.append(
                f"{path}: a review pull request may change only `reviewed_value` "
                f"in existing files of {review_dir}/ and rows of {gazetteer_file}"
            )
    return ReviewDiff(reviews, errors, merge_base)


def git(repo: Path, *args: str) -> str:
    """Run git in `repo` and return its output, stripped."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _check_gazetteer(repo: Path, merge_base: str, path: str) -> list[str]:
    """The gazetteer may gain or change rows, and lose none."""
    old = git(repo, "show", f"{merge_base}:{path}").splitlines()
    new = (repo / path).read_text().splitlines()
    if new[:1] != old[:1]:
        return [f"{path}: the header changed"]
    if len(new) < len(old):
        return [f"{path}: rows were removed (add aliases instead)"]
    return []


def _check_review(repo: Path, merge_base: str, path: str) -> list[str]:
    """Problems with the shape of one changed review file."""
    try:
        old = _records(git(repo, "show", f"{merge_base}:{path}"), path)
        new = _records((repo / path).read_text(), path)
    except ValueError as e:
        return [str(e)]
    errors = [f"{path}: {e}" for e in _changed_columns(old, new)]
    if errors:
        return errors
    try:
        if not read_review(repo / path):
            return [f"{path}: the file has no rows"]
    except ValueError as e:
        return [f"{path}: {e}"]
    return []


def _changed_columns(
    old: list[tuple[int, dict[str, str]]], new: list[tuple[int, dict[str, str]]]
) -> list[str]:
    """Rows removed or changed in a column other than `reviewed_value`.

    Rows may be added for the section, and the province unless the
    placeholder's is empty, of a placeholder row (empty field).
    """

    def identity(record: dict[str, str]) -> tuple[str, ...]:
        return tuple(record[c] for c in COLUMNS if c != "reviewed_value")

    placeholders = [r for _, r in old if not r["field"]]
    added = Counter(identity(r) for _, r in new) - Counter(identity(r) for _, r in old)
    removed = Counter(identity(r) for _, r in old) - Counter(
        identity(r) for _, r in new
    )
    errors = [
        f"the row for {' / '.join(p for p in r[2:6] if p)} was removed or changed in a "
        "column other than reviewed_value"
        for r in removed.elements()
    ]
    for line, record in new:
        if added[identity(record)] <= 0:
            continue
        added[identity(record)] -= 1
        if not any(
            record["field"]
            and not record["reason"]
            and record["section"] == p["section"]
            and p["province"] in ("", record["province"])
            for p in placeholders
        ):
            errors.append(
                f"line {line}: changed a column other than reviewed_value. Rows may "
                "be added, with an empty reason, only for the section and province "
                "of a row with an empty field"
            )
    return errors
