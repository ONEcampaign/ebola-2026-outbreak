"""HTTP fetching shared by the scrapers."""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Some sources (WHO, sante.gouv.cd) reject the default httpx User-Agent.
USER_AGENT = "Mozilla/5.0 (compatible; ONE-data-ebola-scraper/0.1)"
DEFAULT_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_retryable(error: httpx.HTTPError) -> bool:
    """Retry network errors and transient server statuses only."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRY_STATUS
    return isinstance(error, httpx.TransportError)


def get(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> httpx.Response:
    """GET a URL, retrying transient failures with exponential backoff.

    Args:
        url: URL to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        The successful response.

    Raises:
        httpx.HTTPError: If the request fails with a non-retryable error or
            still fails after all attempts.
    """
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.get(
                url, headers=headers, timeout=timeout, follow_redirects=True
            )
            return response.raise_for_status()
        except httpx.HTTPError as e:
            if not _is_retryable(e) or attempt == MAX_ATTEMPTS:
                raise
            wait_s = BACKOFF_BASE_S**attempt
            logger.warning("GET %s failed (%s), retrying in %.0fs", url, e, wait_s)
            time.sleep(wait_s)
    raise AssertionError("unreachable: loop always returns or raises")
