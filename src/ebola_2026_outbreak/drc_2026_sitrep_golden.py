"""Check the sitrep extraction pipeline against 13 hand-checked reports.

The golden set is 13 current-layout reports between SitReps 96 and 157
(`tests/fixtures/sitrep_golden/reports.csv`: report date, number, ministry
URL, the PDF's sha256 and its word dump). Their truth is
`tests/fixtures/sitrep_words/truth.csv` for the table rows, read from the
text layer and checked against the PDFs, and
`tests/fixtures/sitrep_golden/kpis.csv` for the headline indicators, checked
against the rendered first page.

A report is scored by `judge`, the function release decides with, given no
review. A value is accepted when `judge` leaves its cell closed, so a value
the sum checks flag is held. Closed cells of a held report still count as
accepted, so the score covers every report, released or not. `compare` lists
every accepted value that differs from the truth. Truth was adjudicated only
where earlier extractions disagreed, so 0 accepted-wrong values on this set
is a regression pin, not a measured error rate.

Subcommands:

- `record` runs the production clients (or one, with `--extractor`) on the 13
  PDFs and saves each raw `ModelResult` as
  `recorded/<extractor>/<report_date>.json`, the fixtures of the offline test
  (`tests/test_drc_2026_sitrep_golden.py`).
- `run` is the live check, run weekly by `.github/workflows/sitrep_golden.yml`.
  It downloads each PDF by URL, checks its sha256, runs both clients without
  the cache, and prints accepted-wrong values, the flag rate and the returned
  model ids. It exits 1 on any accepted-wrong value, a returned model id that
  differs from the recorded one, or an extractor error, and 2 when it could
  only fail on its sources: a URL that no longer serves the PDF, or a PDF
  whose sha256 changed.
"""

import argparse
import csv
import json
import logging
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import httpx

from ebola_2026_outbreak import fetch
from ebola_2026_outbreak.drc_2026_sitrep_gemini import extract as gemini_extract
from ebola_2026_outbreak.drc_2026_sitrep_mistral import extract as mistral_extract
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    EXTRACTORS,
    FIELDS,
    PERCENT_FIELDS,
    SECTIONS,
    Extractor,
    Gazetteer,
    ModelResult,
    Number,
    RowKey,
    Section,
    Values,
    load_gazetteer,
    sha256_file,
)
from ebola_2026_outbreak.drc_2026_sitreps_download import ROOT
from ebola_2026_outbreak.drc_2026_sitreps_extract import ARBITRATION, Records, judge

logger = logging.getLogger(__name__)

FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN_DIR = FIXTURES / "sitrep_golden"
WORDS_DIR = FIXTURES / "sitrep_words"
REPORTS_PATH = GOLDEN_DIR / "reports.csv"
KPIS_PATH = GOLDEN_DIR / "kpis.csv"
TRUTH_PATH = WORDS_DIR / "truth.csv"
RECORDED_DIR = GOLDEN_DIR / "recorded"

# A recorded response above this size drops its per-page OCR output, which
# normalise does not read.
MAX_RECORD_BYTES = 150_000

EXTRACT: dict[Extractor, Callable[[Path], ModelResult]] = {
    "gemini": gemini_extract,
    "mistral": mistral_extract,
}

_SECTIONS: dict[str, Section] = {s: s for s in SECTIONS}

# `WrongCell.expected` for an accepted row the truth does not have.
NO_TRUTH_ROW = "no truth row"


class Report(NamedTuple):
    """A golden report: where to fetch it and how to recognise it."""

    report_date: str
    sitrep_no: int
    source_url: str
    sha256: str
    words: str


class Score(NamedTuple):
    """A report's accepted values and the number of cells held for review."""

    accepted: dict[RowKey, Values]
    flagged: int


class WrongCell(NamedTuple):
    """An accepted value that differs from the truth."""

    key: RowKey
    field: str
    accepted: Number | None
    expected: Number | str | None


class SourceError(RuntimeError):
    """A golden PDF cannot be fetched, or differs from the recorded file."""


# Fixtures


def load_reports(path: Path = REPORTS_PATH) -> list[Report]:
    """The golden reports, in report date order."""
    with path.open() as f:
        return [
            Report(
                r["report_date"],
                int(r["sitrep_no"]),
                r["source_url"],
                r["sha256"],
                r["words"],
            )
            for r in csv.DictReader(f)
        ]


def _number(text: str, field: str) -> Number | None:
    if text == "":
        return None
    return float(text) if field in PERCENT_FIELDS else int(text)


def load_truth(
    rows_path: Path = TRUTH_PATH, kpis_path: Path = KPIS_PATH
) -> dict[int, dict[RowKey, Values]]:
    """Truth values by sitrep number, then row key.

    A blank cell is a null value. Columns a section does not have are
    ignored.
    """
    truth: dict[int, dict[RowKey, Values]] = {}
    with rows_path.open() as f:
        for r in csv.DictReader(f):
            key = RowKey(_SECTIONS[r["section"]], r["province"], r["zone"])
            truth.setdefault(int(r["sitrep"]), {})[key] = {
                field: _number(r[field], field) for field in FIELDS[key.section]
            }
    with kpis_path.open() as f:
        for r in csv.DictReader(f):
            truth.setdefault(int(r["sitrep"]), {})[RowKey("kpis", "", "")] = {
                field: _number(r[field], field) for field in FIELDS["kpis"]
            }
    return truth


def recorded_path(
    extractor: Extractor, report: Report, *, root: Path = RECORDED_DIR
) -> Path:
    """Path of a report's recorded response from one extractor."""
    return root / extractor / f"{report.report_date}.json"


def write_recorded(
    extractor: Extractor,
    report: Report,
    result: ModelResult,
    *,
    root: Path = RECORDED_DIR,
) -> Path:
    """Save a `ModelResult` in the cache record format.

    A record over `MAX_RECORD_BYTES` drops `response["pages"]`, Mistral's
    per-page markdown; `extraction_raw` and the model ids stay.
    """
    record = result._asdict()
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if len(text.encode()) > MAX_RECORD_BYTES and "pages" in result.response:
        record["response"] = {k: v for k, v in result.response.items() if k != "pages"}
        text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    path = recorded_path(extractor, report, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def read_recorded(
    extractor: Extractor, report: Report, *, root: Path = RECORDED_DIR
) -> ModelResult:
    """Read a report's recorded response.

    Raises:
        FileNotFoundError: If the extractor has not been recorded for the
            report, naming the command that records it.
    """
    path = recorded_path(extractor, report, root=root)
    if not path.exists():
        raise FileNotFoundError(
            f"No recorded {extractor} response for {report.report_date} at {path}; "
            f"run `drc_2026_sitrep_golden.py record --extractor {extractor}`"
        )
    return ModelResult(**json.loads(path.read_text()))


# Scoring


def score(
    records: Records, *, gazetteer: Gazetteer, arbitration: bool = ARBITRATION
) -> Score:
    """Judge one report with no review, as release first would.

    Args:
        records: Both recorded answers and the report's word dump or PDF.
        gazetteer: From `load_gazetteer`.
        arbitration: Passed to `judge`; release's setting by default.

    Returns:
        The values `judge` leaves no cell open on, and the count of its open
        cells, a whole missing section counting once.
    """
    verdict = judge(
        records, gazetteer=gazetteer, review_file=None, arbitration=arbitration
    )
    held = verdict.open_cells
    accepted = {
        key: {f: v for f, v in values.items() if (key, f) not in held}
        for key, values in verdict.values.items()
    }
    return Score(accepted, len(held))


def compare(
    accepted: Mapping[RowKey, Values], truth: Mapping[RowKey, Values]
) -> list[WrongCell]:
    """Every accepted value that differs from the truth, in row key order.

    Values compare exactly, so a null accepted for a printed 0 is wrong.
    """
    wrong: list[WrongCell] = []
    for key in sorted(accepted):
        expected_row = truth.get(key)
        for field, value in accepted[key].items():
            if expected_row is None:
                wrong.append(WrongCell(key, field, value, NO_TRUTH_ROW))
            elif expected_row[field] != value:
                wrong.append(WrongCell(key, field, value, expected_row[field]))
    return wrong


def truth_cells(truth: Mapping[RowKey, Values]) -> int:
    """Cells in a report's truth, the denominator of its flag rate."""
    return sum(len(values) for values in truth.values())


# Sources


def fetch_pdf(report: Report, directory: Path) -> Path:
    """Download a golden PDF and check its sha256.

    Raises:
        SourceError: If the URL fails, or the file is not the recorded one.
    """
    try:
        content = fetch.get(report.source_url).content
    except httpx.HTTPError as e:
        raise SourceError(
            f"SOURCE GONE: SitRep {report.sitrep_no} ({report.report_date}) "
            f"no longer downloads from {report.source_url}: {e}"
        ) from e
    path = directory / f"{report.report_date}.pdf"
    path.write_bytes(content)
    if (sha256 := sha256_file(path)) != report.sha256:
        raise SourceError(
            f"SOURCE CHANGED: SitRep {report.sitrep_no} ({report.report_date}) "
            f"at {report.source_url} has sha256 {sha256}, recorded {report.sha256}"
        )
    return path


def _local_pdfs(pdf_dirs: Iterable[Path]) -> dict[str, Path]:
    """PDFs in the directories by sha256."""
    return {sha256_file(p): p for d in pdf_dirs for p in sorted(d.glob("*.pdf"))}


# Commands


def record(
    extractors: Iterable[Extractor],
    *,
    limit: int | None = None,
    pdf_dirs: Iterable[Path] = (),
) -> None:
    """Run production clients on the golden PDFs and save their responses.

    Args:
        extractors: Which clients to run; their API keys must be set.
        limit: Record only the first `limit` reports.
        pdf_dirs: Directories of local PDFs, matched by sha256; a report
            with no local copy is downloaded.
    """
    local = _local_pdfs(pdf_dirs)
    with tempfile.TemporaryDirectory() as tmp:
        for report in load_reports()[:limit]:
            pdf = local.get(report.sha256) or fetch_pdf(report, Path(tmp))
            for extractor in extractors:
                result = EXTRACT[extractor](pdf)
                path = write_recorded(extractor, report, result)
                logger.info(
                    "Recorded %s for %s (%s) to %s",
                    result.model_id_returned,
                    report.report_date,
                    pdf.name,
                    path.relative_to(ROOT),
                )


def run(*, limit: int | None = None) -> int:
    """Run both clients live on the golden PDFs and compare with the truth.

    Returns:
        0 if every report passes; 1 on an accepted-wrong value, a changed
        model id or an extractor error; 2 if the only failures are sources.
    """
    gazetteer = load_gazetteer()
    truth = load_truth()
    failures: list[str] = []
    source_failures: list[str] = []
    flagged = total = 0

    with tempfile.TemporaryDirectory() as tmp:
        for report in load_reports()[:limit]:
            label = f"SitRep {report.sitrep_no} ({report.report_date})"
            try:
                pdf = fetch_pdf(report, Path(tmp))
            except SourceError as e:
                source_failures.append(str(e))
                continue

            results: dict[Extractor, ModelResult] = {}
            for extractor in EXTRACTORS:
                try:
                    results[extractor] = EXTRACT[extractor](pdf)
                except (httpx.HTTPError, RuntimeError, ValueError) as e:
                    failures.append(f"{label}: {extractor} failed: {e}")
            if len(results) < len(EXTRACTORS):
                continue

            for extractor, result in results.items():
                try:
                    pinned = read_recorded(extractor, report).model_id_returned
                except FileNotFoundError as e:
                    failures.append(str(e))
                    continue
                if result.model_id_returned != pinned:
                    failures.append(
                        f"{label}: {extractor} returned model "
                        f"{result.model_id_returned}, recorded {pinned}"
                    )

            outcome = score(
                Records(results["gemini"], results["mistral"], pdf),
                gazetteer=gazetteer,
            )
            report_truth = truth[report.sitrep_no]
            wrong = compare(outcome.accepted, report_truth)
            cells = outcome.flagged
            flagged += cells
            total += truth_cells(report_truth)
            print(
                f"{label}: {len(wrong)} accepted-wrong, {cells} flagged of "
                f"{truth_cells(report_truth)}; models "
                + ", ".join(f"{e}={r.model_id_returned}" for e, r in results.items())
            )
            failures.extend(
                f"{label}: accepted {w.accepted} for {w.key} {w.field}, "
                f"truth {w.expected}"
                for w in wrong
            )

    if total:
        print(f"Flag rate: {flagged} of {total} cells ({flagged / total:.1%})")
    for message in [*source_failures, *failures]:
        print(message, file=sys.stderr)
    if failures:
        return 1
    return 2 if source_failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `record` or `run` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="record golden fixtures")
    record_parser.add_argument(
        "--extractor", choices=EXTRACTORS, help="record one client (default: both)"
    )
    record_parser.add_argument("--limit", type=int)
    record_parser.add_argument(
        "--pdf-dir",
        type=Path,
        action="append",
        default=[],
        help="directory of local PDFs, matched by sha256 (repeatable)",
    )
    run_parser = commands.add_parser("run", help="live check against the truth")
    run_parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    if args.command == "record":
        extractors = [args.extractor] if args.extractor else EXTRACTORS
        record(extractors, limit=args.limit, pdf_dirs=args.pdf_dir)
        return 0
    return run(limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
