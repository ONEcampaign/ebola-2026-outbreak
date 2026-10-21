"""Tests for the WHO Wayback backfill."""

from pathlib import Path

import pytest

from ebola_2026_outbreak import drc_2026_who_wayback as wayback

FIXTURE = Path(__file__).parent / "fixtures" / "who_bundibugyo_2026-10-20.html"
GOOD, BEFORE_OUTBREAK = "20261020120000", "20260506075346"


@pytest.fixture(autouse=True)
def fake_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve captures from memory instead of archive.org."""
    pages = {GOOD: FIXTURE.read_text(), BEFORE_OUTBREAK: "<html><h2>Other</h2></html>"}
    monkeypatch.setattr(wayback, "load_snapshot", pages.__getitem__)


def test_parse_snapshots_skips_unusable_captures() -> None:
    df, skipped = wayback.parse_snapshots([BEFORE_OUTBREAK, GOOD])
    assert list(skipped) == [BEFORE_OUTBREAK]
    assert "Bundibugyo" in skipped[BEFORE_OUTBREAK]
    assert df.height == 35
    assert df["source_url"].str.contains(f"web.archive.org/web/{GOOD}id_/").all()
    assert df["retrieved_at"].dt.strftime("%Y%m%d%H%M%S").unique().to_list() == [GOOD]


def test_parse_snapshots_raises_when_nothing_parses() -> None:
    with pytest.raises(ValueError, match="No Wayback capture"):
        wayback.parse_snapshots([BEFORE_OUTBREAK])
