"""Extract zone-level sitrep data with two models and release what checks out.

Two steps, run in this order by default:

- `extract` needs the API keys and the PDFs. For every latest upload of a
  report date from `ZONE_START` onwards, it downloads the PDF if absent and
  fills whatever its cache lacks: the words of its table pages, Gemini's
  answer and Mistral's answer, keyed on the PDF's sha256. An upload whose
  cache is complete costs no API call. An API error, an answer that breaks
  the schema, or a returned model id other than the pin leaves the upload
  `pending`; the next run tries again, and the answer is not cached. Any
  other error on an upload (a damaged PDF, say) is logged with its traceback
  and leaves that upload pending too, keeping what was cached before it.
- `release` needs neither keys nor PDFs. It is a function of committed files
  (the manifest, the cache, the review CSVs, the gazetteer and the status
  ledger). For every cached latest upload it normalises both answers and
  reads the text-layer rows from the cached words, both with the current
  gazetteer, reconciles the answers against the rows, applies a completed
  review whose sha256 matches, and runs the sum checks. A report with
  nothing open is released. Otherwise it is held and its review CSV is
  written. An existing review CSV is not rewritten, since a reviewer may be
  filling it. Open cells it lacks (after a gazetteer edit, or a sum that
  runs only once a held row is typed in) are appended as new rows, and
  filled rows for cells that are not open are ignored.

`by_zone.csv` and `headline.csv` are rebuilt from every released upload. A
report date whose latest upload is held or pending keeps the values of the
upload released before it, so no date drops out while a re-upload waits. That
older upload is marked `superseded` once the new one is released. The
ledger (`extraction_status.csv`) keeps one row per upload. Its `extracted_at`
is when the cached answers were requested, or, for a pending upload, when
extraction was first tried, which is what the three-day issue rule counts
from.

`check` tests a review pull request. After the review file rules of
`drc_2026_sitrep_review.check_diff`, it runs the whole of release on the
pull request's tree without writing or opening anything, so the check passes
exactly when release would succeed there. It fails when release would raise,
when an upload released at the merge base would no longer be released or
superseded (a gazetteer edit can do that to a report nobody is reviewing),
and on each cell a changed review file leaves open. `--repo` points it, and
every path it reads, at another checkout's data.

A run that newly holds reports opens one issue listing them, and a run that
finds uploads pending for `PENDING_DAYS` days running opens one issue for
those. The ledger records each upload's issue, and each issue carries a
hidden marker per upload, so no run opens a second issue for an upload
already tracked. An open issue whose marker names the upload is adopted even
when its number never reached the ledger. An issue is commented on and
closed once none of its uploads is held or pending.

Release computes everything first, then checks that no output key repeats
and no previously released date drops out, and only then opens issues and
writes files. `--dry-run` governs what release writes: no output, ledger
or review file, and no issue. Extract always caches what it fetches, since
the cache is content-addressed and an answer paid for is worth keeping.
Without both API keys, a dry run lists the uploads it would extract and
calls nothing.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, NamedTuple

import httpx
import polars as pl

from ebola_2026_outbreak import drc_2026_sitrep_gemini as gemini
from ebola_2026_outbreak import drc_2026_sitrep_mistral as mistral
from ebola_2026_outbreak.drc_2026_sitrep_review import (
    REVIEWERS,
    HeldReport,
    ReviewRow,
    apply,
    check_diff,
    close_issue,
    create_issue,
    describe,
    git,
    held_marker,
    issue_body,
    issue_title,
    open_issues,
    pending_marker,
    read_review,
    resolve,
    review_path,
    review_rows,
    write_review,
)
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    CACHE_DIR,
    EXTRACTORS,
    FIELDS,
    GAZETTEER_PATH,
    HEADLINE_INDICATORS,
    LEVELS,
    REVIEW_DIR,
    ZONE_INDICATORS,
    ZONE_START,
    Discrepancy,
    Extractor,
    Flag,
    Gazetteer,
    ModelResult,
    RowKey,
    Status,
    TextRow,
    Values,
    load_gazetteer,
    normalise,
    read_model_result,
    read_words,
    sha256_file,
    write_model_result,
    write_words,
)
from ebola_2026_outbreak.drc_2026_sitrep_textlayer import (
    dump_words,
    read_rows,
    row_matches,
)
from ebola_2026_outbreak.drc_2026_sitrep_validate import (
    Cell,
    Matcher,
    check_sums,
    reconcile,
)
from ebola_2026_outbreak.drc_2026_sitreps_download import (
    MANIFEST_PATH,
    MANIFEST_SCHEMA,
    OUTBREAK,
    PDF_MIME,
    ROOT,
    MediaItem,
    download,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"
BY_ZONE_PATH = PROCESSED_DIR / f"{OUTBREAK}_sitrep_by_zone.csv"
HEADLINE_PATH = PROCESSED_DIR / f"{OUTBREAK}_sitrep_headline.csv"
STATUS_PATH = PROCESSED_DIR / f"{OUTBREAK}_sitrep_extraction_status.csv"

# Days an upload stays pending, counting the first, before an issue opens.
PENDING_DAYS = 3
# Statuses that keep an upload's issue open.
OPEN: frozenset[Status] = frozenset({"held", "pending"})
# Errors that leave an upload pending: HTTP failures, unusable or
# schema-breaking answers (ValueError), a missing key, GeminiError or a
# changed model id (RuntimeError).
EXTRACTOR_ERRORS = (httpx.HTTPError, ValueError, RuntimeError)
KPIS = RowKey("kpis", "", "")
# How `judge` reconciles, for release and the review check alike. Text-layer
# arbitration stays on only while the golden set shows no accepted-wrong value.
MATCH: Matcher = row_matches
ARBITRATION = True

BY_ZONE_SCHEMA = pl.Schema(
    {
        "outbreak": pl.String,
        "date": pl.Date,
        "level": pl.String,
        "province": pl.String,
        "zone": pl.String,
        "indicator": pl.String,
        "value": pl.Int64,
        "sitrep_number": pl.Int64,
        "media_id": pl.Int64,
        "modified": pl.String,
    }
)
HEADLINE_SCHEMA = pl.Schema(
    {
        "outbreak": pl.String,
        "date": pl.Date,
        "indicator": pl.String,
        "value": pl.Float64,
        "sitrep_number": pl.Int64,
        "media_id": pl.Int64,
        "modified": pl.String,
    }
)
STATUS_SCHEMA = pl.Schema(
    {
        "report_date": pl.Date,
        "media_id": pl.Int64,
        "modified": pl.String,
        "sha256": pl.String,
        "status": pl.String,
        "flagged_cells": pl.Int64,
        "discrepancies": pl.Int64,
        "gemini_model": pl.String,
        "mistral_model": pl.String,
        "extracted_at": pl.String,
        "released_at": pl.String,
        "issue_number": pl.Int64,
    }
)

# An upload is a (media_id, modified) pair, as in the manifest.
type UploadKey = tuple[int, str]


class Upload(NamedTuple):
    """A manifest row: the latest upload of one report date."""

    report_date: date
    sitrep_number: int | None
    media_id: int
    modified: str
    source_url: str

    @property
    def key(self) -> UploadKey:
        return (self.media_id, self.modified)


class LedgerRow(NamedTuple):
    """One row of the status ledger.

    `flagged_cells` counts the open cells of a held report and the cells a
    reviewer verified in a released one.
    """

    report_date: date
    media_id: int
    modified: str
    sha256: str | None
    status: Status
    flagged_cells: int | None
    discrepancies: int | None
    gemini_model: str | None
    mistral_model: str | None
    extracted_at: str | None
    released_at: str | None
    issue_number: int | None


type Ledger = dict[UploadKey, LedgerRow]


class Records(NamedTuple):
    """What the cache holds for one PDF, with None where a record is missing.

    `words` is the cached word dump of the PDF's table pages, or any source
    `read_rows` reads, such as the PDF itself.
    """

    gemini: ModelResult | None
    mistral: ModelResult | None
    words: Path | None

    @property
    def complete(self) -> bool:
        return None not in (self.gemini, self.mistral, self.words)


class Verdict(NamedTuple):
    """A report's values after reconciling, review and sum checks.

    `text_rows` are the text-layer rows the report was reconciled with.
    `review_error` says why a review file was not applied, if it was not.
    """

    values: dict[RowKey, Values]
    open_flags: list[Flag]
    verified: int
    discrepancies: list[Discrepancy]
    text_rows: dict[RowKey, TextRow]
    review_error: str | None = None

    @property
    def open_cells(self) -> set[Cell]:
        """Distinct open cells. A missing section counts once."""
        return _cells(self.open_flags)


class Release(NamedTuple):
    """Everything a release run would write, before issues are handled.

    `published` maps each report date to the upload whose values are in the
    outputs. `reviews` holds the rows of new review files, `appended` the
    rows to add to existing ones.
    """

    ledger: Ledger
    by_zone: pl.DataFrame
    headline: pl.DataFrame
    reviews: dict[Path, list[ReviewRow]]
    held: dict[UploadKey, HeldReport]
    published: dict[date, UploadKey]
    appended: dict[Path, list[ReviewRow]]


class Extracted(NamedTuple):
    """Result of `extract`.

    Attributes:
        ledger: The ledger with a row for every upload in scope.
        tried: Uploads whose cache lacked a record.
        failed: Uploads among them still lacking one.
    """

    ledger: Ledger
    tried: list[UploadKey]
    failed: list[UploadKey]


class IssuePlan(NamedTuple):
    """Issues a run opens and closes."""

    held: list[UploadKey]
    stale: list[UploadKey]
    close: dict[int, list[UploadKey]]


# Inputs


def _in(repo: Path | None, path: Path) -> Path:
    """`path`, or where it sits in the checkout `repo`."""
    return path if repo is None else repo / path.relative_to(ROOT)


def read_manifest(*, repo: Path | None = None) -> pl.DataFrame:
    """The sitrep manifest, of this checkout or of `repo`."""
    return pl.read_csv(_in(repo, MANIFEST_PATH), schema=MANIFEST_SCHEMA)


def latest_uploads(
    manifest: pl.DataFrame, *, report_date: date | None = None
) -> list[Upload]:
    """Latest upload of each report date from `ZONE_START`, or of one date."""
    rows = manifest.filter(
        pl.col("is_latest"),
        pl.col("report_date") >= ZONE_START,
        pl.lit(True) if report_date is None else pl.col("report_date") == report_date,
    ).sort("report_date")
    return [
        Upload(*row)
        for row in rows.select(
            "report_date", "sitrep_number", "media_id", "modified", "source_url"
        ).iter_rows()
    ]


def read_ledger(*, repo: Path | None = None) -> Ledger:
    """The status ledger by upload, of this checkout or of `repo`.

    Empty before the first run.
    """
    path = _in(repo, STATUS_PATH)
    return _parse_ledger(path.read_bytes()) if path.exists() else {}


def _parse_ledger(csv: bytes) -> Ledger:
    rows = pl.read_csv(csv, schema=STATUS_SCHEMA).iter_rows(named=True)
    return {(r["media_id"], r["modified"]): LedgerRow(**r) for r in rows}


def read_records(sha256: str | None, *, root: Path | None = None) -> Records:
    """The cached records of a PDF, from `root` or `CACHE_DIR`."""
    if sha256 is None:
        return Records(None, None, None)
    root = root or CACHE_DIR
    gemini_result, mistral_result = (
        read_model_result(sha256, e, root=root) for e in EXTRACTORS
    )
    return Records(gemini_result, mistral_result, read_words(sha256, root=root))


def _new_row(upload: Upload) -> LedgerRow:
    return LedgerRow(
        report_date=upload.report_date,
        media_id=upload.media_id,
        modified=upload.modified,
        sha256=None,
        status="pending",
        flagged_cells=None,
        discrepancies=None,
        gemini_model=None,
        mistral_model=None,
        extracted_at=None,
        released_at=None,
        issue_number=None,
    )


def _with_records(row: LedgerRow, records: Records, *, now: str) -> LedgerRow:
    """Record the cached model ids and when extraction happened.

    A complete cache dates from its latest request, an incomplete one from
    the first attempt, kept from earlier runs.
    """
    results = [r for r in (records.gemini, records.mistral) if r is not None]
    return row._replace(
        gemini_model=records.gemini.model_id_returned
        if records.gemini
        else row.gemini_model,
        mistral_model=records.mistral.model_id_returned
        if records.mistral
        else row.mistral_model,
        extracted_at=max(r.requested_at for r in results)
        if records.complete
        else row.extracted_at or now,
    )


# Extract


def extract(
    manifest: pl.DataFrame,
    ledger: Ledger,
    *,
    gazetteer: Gazetteer,
    now: str,
    report_date: date | None = None,
    call: bool = True,
) -> Extracted:
    """Fill the cache for every latest upload that lacks a record.

    Args:
        manifest: From `read_manifest`.
        ledger: From `read_ledger`; it holds each known upload's sha256, so
            an upload whose cache is complete needs no PDF.
        gazetteer: From `load_gazetteer`, to check each answer against the
            schema.
        now: UTC timestamp of the run, ISO 8601.
        report_date: Extract only this date.
        call: Download and call the APIs. False only lists what would be
            extracted.
    """
    ledger = dict(ledger)
    tried: list[UploadKey] = []
    failed: list[UploadKey] = []
    for upload in latest_uploads(manifest, report_date=report_date):
        row = ledger.get(upload.key) or _new_row(upload)
        records = read_records(row.sha256)
        if not records.complete:
            tried.append(upload.key)
            if not call:
                failed.append(upload.key)
                ledger[upload.key] = row
                continue
            # Any error is scoped to this upload. It stays pending, keeps what
            # its cache gained, and the run moves on to the next upload.
            try:
                # A new upload may replace a PDF of the same media id in place.
                pdf = download(_media_item(upload), refresh=row.sha256 is None)
                sha256 = sha256_file(pdf)
                row = row._replace(sha256=sha256)
                records = _fill(pdf, sha256, read_records(sha256), gazetteer=gazetteer)
            except Exception:
                logger.exception(
                    "Extracting %s (media %s, modified %s) failed",
                    upload.report_date,
                    *upload.key,
                )
                records = read_records(row.sha256)
            if not records.complete:
                failed.append(upload.key)
        ledger[upload.key] = _with_records(row, records, now=now)
    return Extracted(ledger, tried, failed)


def _media_item(upload: Upload) -> MediaItem:
    return {
        "id": upload.media_id,
        "modified": upload.modified,
        "source_url": upload.source_url,
        "mime_type": PDF_MIME,
    }


def _fill(pdf: Path, sha256: str, records: Records, *, gazetteer: Gazetteer) -> Records:
    """Compute and cache each missing record of one PDF.

    Each record is cached as soon as it is made, and a model that fails is
    logged and left missing, so one model's outage leaves the other's answer
    in the cache. An answer that breaks the extraction schema, or names a
    model other than the pin, counts as a failure.
    """
    if records.words is None:
        with tempfile.TemporaryDirectory() as tmp:
            dump = dump_words(pdf, Path(tmp) / "words.json")
            records = records._replace(words=write_words(sha256, dump, root=CACHE_DIR))
    for extractor in EXTRACTORS:
        if getattr(records, extractor) is not None:
            continue
        try:
            result = _call(extractor, pdf)
            normalise(result.extraction_raw, gazetteer)
        except EXTRACTOR_ERRORS as e:
            logger.error("%s failed on %s: %s", extractor, pdf.name, e)
            continue
        records = records._replace(**{extractor: result})
        write_model_result(sha256, extractor, result, root=CACHE_DIR)
    return records


def _call(extractor: Extractor, pdf: Path) -> ModelResult:
    """One model's answer, checked against the pinned model id.

    Raises:
        RuntimeError: If the response names another model, or none.
    """
    client, pin = {
        "gemini": (gemini.extract, gemini.GEMINI_MODEL),
        "mistral": (mistral.extract, mistral.MISTRAL_OCR_MODEL),
    }[extractor]
    result = client(pdf)
    if result.model_id_returned != pin:
        raise RuntimeError(
            f"{extractor} answered as {result.model_id_returned!r}, pinned to {pin!r}"
        )
    return result


# Release


def judge(
    records: Records,
    *,
    gazetteer: Gazetteer,
    review_file: Path | None,
    arbitration: bool = ARBITRATION,
) -> Verdict:
    """Reconcile a report's cached answers, apply its review, check its sums.

    Release, the review pull request check and the golden scorer all decide
    a report here. The answers are normalised, and the text-layer rows read
    from the cached words, with `gazetteer`.

    Review rows apply to a fixed point. The rows whose cell is open apply,
    the open cells are worked out again, and the rows of newly opened cells
    apply too, until no new row applies. A sum whose operands were held is
    checked only once they are typed in, so its failure opens cells after
    the first pass. A row's key is read through the gazetteer, so a row typed
    under a spelling the gazetteer has since gained answers the canonical
    row's cell. A filled row whose cell is not open (after a gazetteer
    edit, say) is ignored. A review that does not read or apply is ignored,
    with the reason in `review_error`, so the report stays held. A row that
    lacks a field without a flag on it gets one, so a missing value cannot
    be released as a printed blank.

    Args:
        records: A complete cache.
        gazetteer: From `load_gazetteer`.
        review_file: The report's review CSV, if it exists.
        arbitration: Accept the one model's row the text layer confirms when
            the models disagree, as in `reconcile`.
    """
    assert records.gemini and records.mistral and records.words is not None
    text_rows = read_rows(records.words, gazetteer)
    reconciled = reconcile(
        normalise(records.gemini.extraction_raw, gazetteer),
        normalise(records.mistral.extraction_raw, gazetteer),
        text_rows,
        match=MATCH,
        arbitration=arbitration,
    )
    values, verified = reconciled.accepted, set()
    review_error = None
    if review_file is not None:
        try:
            values, verified = _apply_review(
                reconciled.flags,
                reconciled.accepted,
                read_review(review_file),
                gazetteer=gazetteer,
            )
        except ValueError as e:
            review_error = str(e)
            logger.warning("Not applying %s: %s", review_file.name, e)
    open_flags, discrepancies = _open(reconciled.flags, values, verified)
    return Verdict(
        values, open_flags, len(verified), discrepancies, text_rows, review_error
    )


def _apply_review(
    flags: list[Flag],
    accepted: dict[RowKey, Values],
    review: list[ReviewRow],
    *,
    gazetteer: Gazetteer,
) -> tuple[dict[RowKey, Values], set[Cell]]:
    """Apply the review rows of every cell ever open; see `judge`.

    The cells reached only grow, so the loop ends.
    """
    values, verified = accepted, set()
    reached: set[Cell] = set()
    applied = 0
    while True:
        reached |= _cells(_open(flags, values, verified)[0])
        rows = [r for r in review if _review_cell(r, gazetteer) in reached]
        if len(rows) == applied:
            return values, verified
        values, verified = apply(accepted, rows, gazetteer=gazetteer)
        applied = len(rows)


def _open(
    flags: list[Flag], values: dict[RowKey, Values], verified: set[Cell]
) -> tuple[list[Flag], list[Discrepancy]]:
    """The open flags and publisher discrepancies of reviewed values.

    `flags` are the reconcile flags, and one on a cell a reviewer verified is
    closed.
    """
    # A flagged row missing from the merged values was dismissed as absent.
    open_flags = [
        f for f in flags if f.key in values and (f.key, f.field) not in verified
    ]
    sum_flags, discrepancies = check_sums(values, verified)
    open_flags += sum_flags
    flagged = _cells(open_flags)
    open_flags += [
        Flag(key, field, "section_missing")
        for key, row in sorted(values.items())
        for field in FIELDS[key.section]
        if field not in row and (key, field) not in flagged
    ]
    return open_flags, discrepancies


def release(
    manifest: pl.DataFrame,
    ledger: Ledger,
    *,
    gazetteer: Gazetteer,
    now: str,
    repo: Path | None = None,
) -> Release:
    """Decide every upload's status and build the outputs.

    Reads the cache and the review files; writes nothing.

    Args:
        manifest: From `read_manifest`.
        ledger: The ledger after `extract`, holding each upload's sha256.
        gazetteer: From `load_gazetteer`.
        now: UTC timestamp of the run, ISO 8601.
        repo: Read the cache and review files of this checkout instead.

    Raises:
        ValueError: If an upload released before no longer releases while
            its date has no newer released upload.
    """
    out: Ledger = {}
    reviews: dict[Path, list[ReviewRow]] = {}
    held: dict[UploadKey, HeldReport] = {}
    appended: dict[Path, list[ReviewRow]] = {}
    published: dict[date, tuple[Upload, dict[RowKey, Values]]] = {}
    cache, review_dir = _in(repo, CACHE_DIR), _in(repo, REVIEW_DIR)

    uploads = latest_uploads(manifest)
    for upload in uploads:
        row = ledger.get(upload.key) or _new_row(upload)
        sha256 = row.sha256
        records = read_records(sha256, root=cache)
        if sha256 is None or not records.complete:
            out[upload.key] = row._replace(status="pending")
            continue

        path = review_path(upload.report_date.isoformat(), sha256, root=review_dir)
        verdict = judge(
            records, gazetteer=gazetteer, review_file=path if path.exists() else None
        )
        _log_discrepancies(str(upload.report_date), verdict)
        row = _with_records(row, records, now=now)._replace(
            discrepancies=len(verdict.discrepancies)
        )
        if not verdict.open_flags:
            out[upload.key] = row._replace(
                status="released",
                flagged_cells=verdict.verified,
                released_at=row.released_at if row.status == "released" else now,
            )
            published[upload.report_date] = (upload, verdict.values)
            continue

        out[upload.key] = row._replace(
            status="held", flagged_cells=len(verdict.open_cells)
        )
        held[upload.key] = HeldReport(
            report_date=upload.report_date.isoformat(),
            sha256=sha256,
            cells=len(verdict.open_cells),
            review_path=_relative(path),
            pdf_url=upload.source_url,
        )
        rows = review_rows(
            verdict.open_flags,
            report_date=upload.report_date.isoformat(),
            sha256=sha256,
            pdf_url=upload.source_url,
            text_rows=verdict.text_rows,
        )
        if not path.exists():
            reviews[path] = rows
            continue
        try:
            present = {_review_cell(r, gazetteer) for r in read_review(path)}
        except ValueError as e:
            logger.warning("Not extending %s: %s", path.name, e)
            continue
        if missing := [r for r in rows if _review_cell(r, gazetteer) not in present]:
            logger.warning(
                "%s lacks %d open cells; appending them", path.name, len(missing)
            )
            appended[path] = missing

    # Of the uploads that are no longer latest, the last released one of a date
    # whose latest upload is not released stays published and every other is
    # superseded.
    by_date = {u.report_date: u for u in uploads}
    older = sorted(
        (r for k, r in ledger.items() if k not in out),
        key=lambda r: r.modified,
        reverse=True,
    )
    for row in older:
        key = (row.media_id, row.modified)
        if row.status != "released" or row.report_date in published:
            out[key] = row._replace(status="superseded")
            continue
        records = read_records(row.sha256, root=cache)
        path = review_path(
            row.report_date.isoformat(), row.sha256 or "", root=review_dir
        )
        verdict = (
            judge(
                records,
                gazetteer=gazetteer,
                review_file=path if path.exists() else None,
            )
            if records.complete
            else None
        )
        if verdict is None or verdict.open_flags:
            raise ValueError(
                f"{row.report_date} upload {key} was released and no longer "
                "releases from its cache and review"
            )
        latest = by_date.get(row.report_date)
        # A re-upload is the same report, so it takes the latest upload's
        # sitrep number.
        upload = Upload(
            row.report_date,
            latest.sitrep_number if latest else None,
            row.media_id,
            row.modified,
            latest.source_url if latest else "",
        )
        published[row.report_date] = (upload, verdict.values)
        out[key] = row

    return Release(
        ledger=out,
        by_zone=_by_zone(published.values()),
        headline=_headline(published.values()),
        reviews=reviews,
        held=held,
        published={d: u.key for d, (u, _) in published.items()},
        appended=appended,
    )


def _log_discrepancies(report: str, verdict: Verdict) -> None:
    for d in verdict.discrepancies:
        logger.info(
            "Publisher discrepancy in %s: %s %s printed %s, sums to %s",
            report,
            describe(d.key),
            d.field,
            d.printed,
            d.computed,
        )


def _cells(flags: Sequence[Flag]) -> set[Cell]:
    return {(f.key, f.field or "") for f in flags}


def _review_cell(row: ReviewRow, gazetteer: Gazetteer) -> Cell:
    """The flagged cell a review row answers.

    A flagged row answers its row's cell, read through the gazetteer as
    `apply` reads it. A row a reviewer added under a placeholder answers the
    placeholder: its province's missing zones, or the whole section.
    """
    key = RowKey(row["section"], row["province"], row["zone"])
    if row["reason"]:
        return (resolve(key, gazetteer) or key, row["field"])
    if row["section"] == "zones" and row["province"]:
        return (RowKey("zones", row["province"], ""), "")
    return (RowKey(row["section"], "", ""), "")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _by_zone(published: Iterable[tuple[Upload, dict[RowKey, Values]]]) -> pl.DataFrame:
    records = [
        {
            "outbreak": OUTBREAK,
            "date": upload.report_date,
            "level": LEVELS[key.section],
            "province": key.province or None,
            "zone": key.zone or None,
            "indicator": indicator,
            "value": row[field],
            "sitrep_number": upload.sitrep_number,
            "media_id": upload.media_id,
            "modified": upload.modified,
        }
        for upload, values in published
        for key, row in values.items()
        if key.section in LEVELS
        for field, indicator in ZONE_INDICATORS.items()
    ]
    return pl.DataFrame(records, schema=BY_ZONE_SCHEMA).sort(
        "date", "level", "province", "zone", "indicator"
    )


def _headline(published: Iterable[tuple[Upload, dict[RowKey, Values]]]) -> pl.DataFrame:
    records = [
        {
            "outbreak": OUTBREAK,
            "date": upload.report_date,
            "indicator": indicator,
            "value": values[KPIS][field],
            "sitrep_number": upload.sitrep_number,
            "media_id": upload.media_id,
            "modified": upload.modified,
        }
        for upload, values in published
        for field, indicator in HEADLINE_INDICATORS.items()
    ]
    return pl.DataFrame(records, schema=HEADLINE_SCHEMA).sort("date", "indicator")


def check_outputs(result: Release, *, previous_dates: set[date]) -> None:
    """Refuse outputs with a repeated key, or without a date released before.

    Raises:
        ValueError: On either.
    """
    keys = {
        "by_zone": (result.by_zone, ["date", "level", "province", "zone", "indicator"]),
        "headline": (result.headline, ["date", "indicator"]),
    }
    for name, (df, key) in keys.items():
        if df.select(key).is_duplicated().any():
            raise ValueError(f"Duplicate {', '.join(key)} rows in {name}")
    lost = previous_dates - set(result.published)
    if lost:
        raise ValueError(f"Dates would drop out of the outputs: {sorted(lost)}")


def previous_dates() -> set[date]:
    """Report dates in the outputs already written."""
    dates: set[date] = set()
    for path, schema in (
        (BY_ZONE_PATH, BY_ZONE_SCHEMA),
        (HEADLINE_PATH, HEADLINE_SCHEMA),
    ):
        if path.exists():
            dates.update(pl.read_csv(path, schema=schema)["date"].to_list())
    return dates


# Issues


def plan_issues(
    before: Ledger, after: Ledger, *, today: date
) -> tuple[Ledger, IssuePlan]:
    """Decide which issues to open and close.

    An upload keeps its issue while it stays in the status the issue was
    opened for. A newly held upload goes in the run's held issue, and an
    upload pending for `PENDING_DAYS` days with no issue goes in the run's
    pending issue. An issue none of whose uploads is held or pending closes.

    Args:
        before: The ledger the run started from.
        after: The ledger the run would write, issue numbers as carried.
        today: The run's date.

    Returns:
        `after` with the issue numbers of open uploads settled (None where a
        new issue is due), and the plan.
    """
    after = dict(after)
    held: list[UploadKey] = []
    stale: list[UploadKey] = []
    for key, row in after.items():
        if row.status not in OPEN:
            continue
        prev = before.get(key)
        if prev is not None and prev.status == row.status and prev.issue_number:
            after[key] = row._replace(issue_number=prev.issue_number)
            continue
        after[key] = row._replace(issue_number=None)
        if row.status == "held":
            held.append(key)
        elif _pending_days(row, today=today) >= PENDING_DAYS:
            stale.append(key)

    still_open = {
        r.issue_number for r in after.values() if r.status in OPEN and r.issue_number
    }
    close: dict[int, list[UploadKey]] = {}
    for key, prev in sorted(before.items()):
        if (
            prev.status in OPEN
            and prev.issue_number
            and prev.issue_number not in still_open
        ):
            close.setdefault(prev.issue_number, []).append(key)
    return after, IssuePlan(held, stale, close)


def _pending_days(row: LedgerRow, *, today: date) -> int:
    """Days an upload has been pending, counting the first; 0 if unknown."""
    if row.extracted_at is None:
        return 0
    return (today - datetime.fromisoformat(row.extracted_at).date()).days + 1


def handle_issues(
    result: Release, plan: IssuePlan, ledger: Ledger, *, dry_run: bool
) -> Ledger:
    """Open and close the planned issues; record new issue numbers.

    An upload due an issue first adopts an open issue whose marker names it,
    one opened by a run whose ledger was never written, and only the rest go
    in a new issue. `gh` runs only inside GitHub Actions and not in a dry
    run, so elsewhere new uploads keep no issue number and a later run opens
    their issue.
    """
    ledger = dict(ledger)
    run_url = _run_url()
    existing = {
        n: body
        for n, body in open_issues(dry_run=dry_run).items()
        if n not in plan.close
    }

    def adopt(
        keys: list[UploadKey], marker: Callable[[UploadKey], str]
    ) -> list[UploadKey]:
        """Adopt an open issue for each upload it marks; return the rest."""
        untracked = []
        for key in keys:
            number = next((n for n, b in existing.items() if marker(key) in b), None)
            if number is None:
                untracked.append(key)
            else:
                ledger[key] = ledger[key]._replace(issue_number=number)
        return untracked

    if held := adopt(plan.held, lambda k: held_marker(result.held[k].sha256)):
        reports = [result.held[k] for k in held]
        number = create_issue(
            title=issue_title(len(reports)),
            body=issue_body(reports, run_url=run_url),
            dry_run=dry_run,
        )
        for key in held:
            ledger[key] = ledger[key]._replace(issue_number=number)
    if stale := adopt(plan.stale, lambda k: pending_marker(*k)):
        rows = [ledger[k] for k in stale]
        number = create_issue(
            title=pending_title(len(rows)),
            body=pending_body(rows, run_url=run_url),
            dry_run=dry_run,
        )
        for key in stale:
            ledger[key] = ledger[key]._replace(issue_number=number)
    for number, keys in plan.close.items():
        lines = [
            f"- {ledger[k].report_date} (media {k[0]}): {ledger[k].status}"
            for k in keys
            if k in ledger
        ]
        close_issue(
            number,
            comment="No report listed here is held or pending now:\n\n"
            + "\n".join(lines)
            + f"\n\nRun: {run_url}\n",
            dry_run=dry_run,
        )
    return ledger


def pending_title(count: int) -> str:
    """Title of an issue on uploads that fail to extract."""
    noun = "upload" if count == 1 else "uploads"
    return f"DRC sitrep extraction pending for {count} {noun}"


def pending_body(rows: Sequence[LedgerRow], *, run_url: str) -> str:
    """Body of an issue on uploads pending for `PENDING_DAYS` days."""
    lines = [
        f"{' '.join(REVIEWERS)}: extraction has failed for {PENDING_DAYS} days "
        "running on the uploads below, through an API error, an answer that "
        "breaks the schema, or a model id other than its pin. The run log "
        "names the error. Each run tries again. This issue closes itself once "
        "every upload is extracted.",
        "",
        "| Report date | Media id | Modified | First tried |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.report_date} | {r.media_id} | {r.modified} | {r.extracted_at} |"
        for r in rows
    ]
    lines += ["", f"Run: {run_url}", ""]
    lines += [pending_marker(r.media_id, r.modified) for r in rows]
    return "\n".join(lines) + "\n"


def _run_url() -> str:
    """The GitHub Actions run, or "a local run"."""
    env = os.environ
    if all(
        env.get(v) for v in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    ):
        return (
            f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}"
            f"/actions/runs/{env['GITHUB_RUN_ID']}"
        )
    return "a local run"


# Writing


def ledger_frame(ledger: Ledger) -> pl.DataFrame:
    """The ledger as a frame, one row per upload."""
    return pl.DataFrame(
        [r._asdict() for r in ledger.values()], schema=STATUS_SCHEMA
    ).sort("report_date", "media_id", "modified")


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(path)


def write_ledger(ledger: Ledger) -> None:
    """Write the status ledger."""
    frame = ledger_frame(ledger)
    if frame.select("media_id", "modified").is_duplicated().any():
        raise ValueError("Duplicate uploads in the status ledger")
    _write_csv(frame, STATUS_PATH)


def write_release(result: Release, ledger: Ledger) -> None:
    """Write review files, outputs and the ledger."""
    for path, rows in result.reviews.items():
        tmp = path.with_suffix(".csv.tmp")
        write_review(tmp, rows)
        tmp.replace(path)
    for path, rows in result.appended.items():
        # Rows go after the existing ones, which stay byte for byte, so a
        # reviewer's open pull request still merges.
        tmp = path.with_suffix(".csv.tmp")
        write_review(tmp, rows)
        _, new = tmp.read_text().split("\n", 1)
        existing = path.read_text()
        tmp.write_text(existing + ("" if existing.endswith("\n") else "\n") + new)
        tmp.replace(path)
    _write_csv(result.by_zone, BY_ZONE_PATH)
    _write_csv(result.headline, HEADLINE_PATH)
    write_ledger(ledger)


# Review pull request check


def check_review(*, base: str = "origin/main", repo: Path = ROOT) -> list[str]:
    """Check a review pull request, which passes when release would succeed.

    The changes must follow the review file rules (`check_diff`). Then
    release runs on the working tree of `repo`, writing nothing and opening
    no issue, and its outputs go through `check_outputs` against the dates
    released at the merge base. Each report whose review file changed goes
    through `judge`, as release reads it, so each cell it leaves open,
    publisher discrepancies included, gets its own message.

    Args:
        base: The branch the pull request merges into.
        repo: The checkout whose data (manifest, ledger, cache, review files
            and gazetteer) is checked.

    Returns:
        One message per problem: a broken rule, release raising, an upload
        released at the merge base that would no longer be released or
        superseded, a review that does not apply, or a cell still open.
        Empty when the pull request passes.
    """
    diff = check_diff(base=base, repo=repo)
    if diff.errors:
        return diff.errors
    gazetteer_path = _in(repo, GAZETTEER_PATH)
    try:
        gazetteer = load_gazetteer(gazetteer_path)
    except ValueError as e:
        return [f"{gazetteer_path.relative_to(repo)}: {e}"]

    errors = _check_release(repo, merge_base=diff.merge_base, gazetteer=gazetteer)
    cache = _in(repo, CACHE_DIR)
    for path in diff.reviews:
        name = path.relative_to(repo).as_posix()
        sha256 = next((r["sha256"] for r in read_review(path) if r["reason"]), None)
        records = read_records(sha256, root=cache)
        if not records.complete:
            errors.append(f"{name}: no cached extraction for {sha256}")
            continue
        verdict = judge(records, gazetteer=gazetteer, review_file=path)
        if verdict.review_error is not None:
            errors.append(f"{name}: {verdict.review_error}")
            continue
        errors += [f"{name}: {_open_message(f)}" for f in verdict.open_flags]
        _log_discrepancies(name, verdict)
    return errors


def _check_release(repo: Path, *, merge_base: str, gazetteer: Gazetteer) -> list[str]:
    """What would stop release on the working tree of `repo`.

    Release raising, or an upload `released` in the merge base's ledger that
    would be neither released nor superseded.
    """
    base = _base_ledger(repo, merge_base)
    try:
        result = release(
            read_manifest(repo=repo),
            read_ledger(repo=repo),
            gazetteer=gazetteer,
            now=datetime.now(UTC).isoformat(timespec="seconds"),
            repo=repo,
        )
    except ValueError as e:
        return [f"Release would fail: {e}"]

    errors: list[str] = []
    for key, row in sorted(base.items()):
        after = result.ledger.get(key)
        status = "dropped from the ledger" if after is None else after.status
        if row.status == "released" and status not in ("released", "superseded"):
            errors.append(
                f"{row.report_date} upload (media {key[0]}, modified {key[1]}) "
                f"was released and would now be {status}"
            )
    released = {r.report_date for r in base.values() if r.status == "released"}
    try:
        check_outputs(result, previous_dates=released)
    except ValueError as e:
        errors.append(f"Release would fail: {e}")
    return errors


def _base_ledger(repo: Path, merge_base: str) -> Ledger:
    """The ledger committed at `merge_base`; empty if it had none."""
    path = STATUS_PATH.relative_to(ROOT).as_posix()
    if not git(repo, "ls-tree", "--name-only", merge_base, "--", path):
        return {}
    return _parse_ledger(git(repo, "show", f"{merge_base}:{path}").encode())


def _open_message(flag: Flag) -> str:
    """What a reviewer must do about a cell still open."""
    if flag.field is None:
        return (
            f"{describe(flag.key)} rows are missing; add one row per cell under "
            "the row with an empty field"
        )
    cell = f"{describe(flag.key)} {flag.field}"
    if flag.reason == "sum_mismatch":
        return (
            f"{cell} is open (sum_mismatch): the table sums do not hold with the "
            "values typed; check them against the PDF"
        )
    return f"{cell} is open ({flag.reason}): type the value printed in the PDF"


# Command line


def summarise_extract(result: Extracted, *, called: bool) -> list[str]:
    """Lines describing what an extract run did, or would do."""
    if not called:
        return [
            f"Extract: would extract {len(result.tried)} uploads; API keys not "
            "set, so nothing was called"
        ]
    done = len(result.tried) - len(result.failed)
    return [
        f"Extract: {len(result.tried)} uploads lacked records; {done} complete, "
        f"{len(result.failed)} pending"
    ]


def summarise_release(
    result: Release, plan: IssuePlan, ledger: Ledger, *, dry_run: bool
) -> list[str]:
    """Lines describing what a release run did, or would do."""
    counts = {
        s: sum(r.status == s for r in ledger.values())
        for s in ("released", "held", "pending", "superseded")
    }
    write, open_, close = (
        ("Would write", "Would open", "Would close")
        if dry_run
        else ("Wrote", "Opened", "Closed")
    )
    lines = [
        "Release: " + ", ".join(f"{n} {s}" for s, n in counts.items()),
        f"{write} {result.by_zone.height} by-zone rows and "
        f"{result.headline.height} headline rows for {len(result.published)} dates",
    ]
    lines += [
        f"{write} review file {_relative(p)} ({len(rows)} cells)"
        for p, rows in result.reviews.items()
    ]
    lines += [
        f"{write} {len(rows)} open cells to the end of {_relative(p)}"
        for p, rows in result.appended.items()
    ]
    for keys, what in ((plan.held, "held reports"), (plan.stale, "pending uploads")):
        if keys:
            dates = ", ".join(str(ledger[k].report_date) for k in keys)
            lines.append(f"{open_} an issue for {len(keys)} {what}: {dates}")
    lines += [f"{close} issue #{n}" for n in plan.close]
    return lines


type Command = Literal["extract", "release"]


def run(
    command: Command | None,
    *,
    now: str,
    report_date: date | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Run extract, release, or both, and describe what was done.

    Args:
        command: One step, or None for both.
        now: UTC timestamp of the run, ISO 8601.
        report_date: Extract only this date.
        dry_run: Write no output, ledger or review file, and open no issue.
            Extract still caches what it fetches, and calls nothing unless
            both API keys are set.

    Returns:
        Summary lines.

    Raises:
        ValueError: If the outputs would repeat a key or lose a date
            released before; nothing is written.
    """
    gazetteer = load_gazetteer(GAZETTEER_PATH)
    manifest = read_manifest()
    ledger = read_ledger()
    lines: list[str] = []

    if command in (None, "extract"):
        call = not dry_run or all(
            os.environ.get(m.API_KEY_ENV) for m in (gemini, mistral)
        )
        extracted = extract(
            manifest,
            ledger,
            gazetteer=gazetteer,
            now=now,
            report_date=report_date,
            call=call,
        )
        ledger = extracted.ledger
        if not dry_run:
            write_ledger(ledger)
        lines += summarise_extract(extracted, called=call)
    if command == "extract":
        return lines

    result = release(manifest, ledger, gazetteer=gazetteer, now=now)
    check_outputs(result, previous_dates=previous_dates())
    after, plan = plan_issues(
        ledger, result.ledger, today=datetime.fromisoformat(now).date()
    )
    after = handle_issues(result, plan, after, dry_run=dry_run)
    if not dry_run:
        write_release(result, after)
    return lines + summarise_release(result, plan, after, dry_run=dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return 1 when the review check fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        nargs="?",
        choices=("extract", "release", "check"),
        help="default: extract then release; check tests a review pull request",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write no output, ledger or review file and open no issue; extract "
        "still caches what it fetches, and calls nothing without both API keys",
    )
    parser.add_argument(
        "--report-date",
        type=date.fromisoformat,
        help="extract only this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--base", default="origin/main", help="branch a review pull request merges into"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="checkout whose data the check reads (default: this one)",
    )
    args = parser.parse_args(argv)
    if args.command == "check":
        try:
            errors = check_review(base=args.base, repo=args.repo.resolve())
        except subprocess.CalledProcessError as e:
            print(
                f"git {' '.join(e.cmd[3:])} failed: {e.stderr.strip()}", file=sys.stderr
            )
            return 1
        for error in errors:
            print(error, file=sys.stderr)
        if errors:
            print(f"{len(errors)} problems; fix them and push again.", file=sys.stderr)
            return 1
        print("Review check passed: the report would be released.")
        return 0

    lines = run(
        args.command,
        now=datetime.now(UTC).isoformat(timespec="seconds"),
        report_date=args.report_date,
        dry_run=args.dry_run,
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
