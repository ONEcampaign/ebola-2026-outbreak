"""HTTP fetching shared by the scrapers and the sitrep extractors.

`get` retries network errors and transient server statuses with exponential
backoff. `post` carries paid model calls, so it retries only where the server
cannot have processed the request: connect failures and the statuses in
`UNPROCESSED_STATUS`. A read timeout is raised at once. API keys travel in
request headers only. URLs appear in retry log lines and in httpx exception
messages, and headers appear in neither.
"""

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Some sources (WHO, sante.gouv.cd) reject the default httpx User-Agent.
USER_AGENT = "Mozilla/5.0 (compatible; ONE-data-ebola-scraper/0.1)"
DEFAULT_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Rejections issued before any model work starts, so a repeat is not billed twice.
UNPROCESSED_STATUS = frozenset({429, 502, 503, 504})


def _is_retryable(error: httpx.HTTPError, *, idempotent: bool) -> bool:
    """Retry transient failures; a non-idempotent request only where unprocessed."""
    if isinstance(error, httpx.HTTPStatusError):
        statuses = RETRY_STATUS if idempotent else UNPROCESSED_STATUS
        return error.response.status_code in statuses
    if idempotent:
        return isinstance(error, httpx.TransportError)
    return isinstance(error, httpx.ConnectError | httpx.ConnectTimeout)


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: Any,
    timeout: float,
    idempotent: bool,
) -> httpx.Response:
    """Send one request through `client`, retrying transient failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.request(
                method,
                url,
                headers={"User-Agent": USER_AGENT, **headers},
                json=json,
                timeout=timeout,
                follow_redirects=True,
            )
            return response.raise_for_status()
        except httpx.HTTPError as e:
            if not _is_retryable(e, idempotent=idempotent) or attempt == MAX_ATTEMPTS:
                raise
            wait_s = BACKOFF_BASE_S**attempt
            logger.warning(
                "%s %s failed (%s), retrying in %.0fs", method, url, e, wait_s
            )
            time.sleep(wait_s)
    raise AssertionError("unreachable: loop always returns or raises")


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: Any,
    client: httpx.Client | None,
    timeout: float,
    idempotent: bool,
) -> httpx.Response:
    """Send through `client`, or through a short-lived client if none."""
    if client is not None:
        return _send(
            client,
            method,
            url,
            headers=headers,
            json=json,
            timeout=timeout,
            idempotent=idempotent,
        )
    with httpx.Client() as own:
        return _send(
            own,
            method,
            url,
            headers=headers,
            json=json,
            timeout=timeout,
            idempotent=idempotent,
        )


def get(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> httpx.Response:
    """GET a URL, retrying transient failures with exponential backoff.

    Args:
        url: URL to fetch.
        client: Client to send through. Tests pass one built on
            `httpx.MockTransport`. A short-lived client is used if None.
        timeout: Per-request timeout in seconds.

    Returns:
        The successful response.

    Raises:
        httpx.HTTPError: If the request fails with a non-retryable error or
            still fails after all attempts.
    """
    return _request(
        "GET",
        url,
        headers={},
        json=None,
        client=client,
        timeout=timeout,
        idempotent=True,
    )


def post(
    url: str,
    *,
    json: Any,
    headers: dict[str, str],
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> httpx.Response:
    """POST a JSON body, retrying only failures that leave it unprocessed.

    Connect errors and 429, 502, 503 and 504 responses are retried with
    exponential backoff. A read timeout or a 500 is raised, since the server
    may have run, and billed, the call.

    Args:
        url: URL to post to. Must not carry an API key.
        json: Body, serialised as JSON.
        headers: Extra headers, including any API key, which is not logged.
        client: Client to send through. Tests pass one built on
            `httpx.MockTransport`. A short-lived client is used if None.
        timeout: Per-request timeout in seconds.

    Returns:
        The successful response.

    Raises:
        httpx.HTTPError: If the request fails with a non-retryable error or
            still fails after all attempts.
    """
    return _request(
        "POST",
        url,
        headers=headers,
        json=json,
        client=client,
        timeout=timeout,
        idempotent=False,
    )
