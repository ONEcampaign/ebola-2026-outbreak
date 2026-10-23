"""Tests for the shared HTTP helpers."""

import json
import logging

import httpx
import pytest

from ebola_2026_outbreak import fetch

URL = "https://api.example.org/v1/models/m:generateContent"
KEY = "secret-key-123"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)


def client_for(statuses: list[int], seen: list[httpx.Request]) -> httpx.Client:
    """A client whose server answers with `statuses` in turn."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(statuses[len(seen) - 1], json={"ok": True})

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_post_sends_body_and_headers_and_retries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[httpx.Request] = []
    with caplog.at_level(logging.WARNING):
        response = fetch.post(
            URL,
            json={"a": 1},
            headers={"x-goog-api-key": KEY},
            client=client_for([503, 200], seen),
        )

    assert response.json() == {"ok": True}
    assert len(seen) == 2
    request = seen[-1]
    assert request.method == "POST"
    assert json.loads(request.content) == {"a": 1}
    assert request.headers["x-goog-api-key"] == KEY
    assert request.headers["User-Agent"] == fetch.USER_AGENT
    assert KEY not in str(request.url)
    assert caplog.records, "the retry should log"
    assert KEY not in caplog.text


def test_get_uses_injected_client() -> None:
    seen: list[httpx.Request] = []
    fetch.get(URL, client=client_for([200], seen))
    assert [r.method for r in seen] == ["GET"]
    assert seen[0].headers["User-Agent"] == fetch.USER_AGENT


@pytest.mark.parametrize(
    ("statuses", "attempts"),
    [([400], 1), ([503] * fetch.MAX_ATTEMPTS, fetch.MAX_ATTEMPTS)],
    ids=["non-retryable", "retries-exhausted"],
)
def test_post_raises(statuses: list[int], attempts: int) -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(httpx.HTTPStatusError):
        fetch.post(URL, json={}, headers={}, client=client_for(statuses, seen))
    assert len(seen) == attempts


def client_raising(
    errors: list[httpx.TransportError | None], seen: list[httpx.Request]
) -> httpx.Client:
    """A client whose transport raises each of `errors` in turn, then answers 200."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        error = errors[len(seen) - 1] if len(seen) <= len(errors) else None
        if error is not None:
            raise error
        return httpx.Response(200, json={"ok": True})

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_post_retries_connect_error() -> None:
    seen: list[httpx.Request] = []
    response = fetch.post(
        URL,
        json={},
        headers={},
        client=client_raising([httpx.ConnectError("refused")], seen),
    )
    assert response.json() == {"ok": True}
    assert len(seen) == 2


def test_post_does_not_retry_read_timeout() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(httpx.ReadTimeout):
        fetch.post(
            URL,
            json={},
            headers={},
            client=client_raising([httpx.ReadTimeout("slow")], seen),
        )
    assert len(seen) == 1


def test_post_does_not_retry_server_error() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(httpx.HTTPStatusError):
        fetch.post(URL, json={}, headers={}, client=client_for([500, 200], seen))
    assert len(seen) == 1


def test_get_retries_read_timeout() -> None:
    seen: list[httpx.Request] = []
    response = fetch.get(URL, client=client_raising([httpx.ReadTimeout("slow")], seen))
    assert response.json() == {"ok": True}
    assert len(seen) == 2
