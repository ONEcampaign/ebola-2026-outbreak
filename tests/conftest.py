"""Fixtures shared by every test."""

import pytest


@pytest.fixture(autouse=True)
def outside_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test as if outside GitHub Actions, so `gh` is never called.

    A test of the `gh` path sets `GITHUB_ACTIONS` itself, with `gh` stubbed.
    """
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
