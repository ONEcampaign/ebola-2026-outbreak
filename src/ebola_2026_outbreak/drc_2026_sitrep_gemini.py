"""Extract a DRC sitrep with Gemini.

Every page of the PDF is rendered to a PNG at 110 dpi and sent in one
`generateContent` request with `PROMPT` and the Gemini response schema from
`schema_for("gemini")`. PNGs are saved unquantised, since pdfplumber's default
256-colour palette blurs small digits. Gemini returns the extraction as JSON
text in its first candidate.

Gemini 3.6 Flash and newer ignore temperature, so none is sent. Repeated
calls can differ, so the cache fixes the answer used and the Mistral
cross-check catches a bad one. The response's `modelVersion` is recorded so a silent model change shows up
against the pin.

The API key is read from `GEMINI_API_KEY` and travels only in the
`x-goog-api-key` header. No URL, log line or error message here carries it.
"""

import base64
import io
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pdfplumber

from ebola_2026_outbreak import fetch
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    PROMPT,
    ModelResult,
    UnusableAnswer,
    schema_for,
)

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.8-flash"
API_KEY_ENV = "GEMINI_API_KEY"
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
RENDER_DPI = 110
# One request reads every page and writes every table; allow for a slow answer.
TIMEOUT_S = 300.0


class GeminiError(RuntimeError):
    """Gemini gave no usable extraction, or could not be called."""


def render_pages(pdf_path: Path) -> list[bytes]:
    """Render every page of a PDF to PNG bytes at `RENDER_DPI`."""
    pages: list[bytes] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            buf = io.BytesIO()
            page.to_image(resolution=RENDER_DPI).save(buf, format="PNG", quantize=False)
            pages.append(buf.getvalue())
    return pages


def build_request(pages: list[bytes]) -> dict[str, Any]:
    """Request body: the prompt, one inline PNG per page, and the schema."""
    images = [
        {
            "inlineData": {
                "mimeType": "image/png",
                "data": base64.b64encode(png).decode("ascii"),
            }
        }
        for png in pages
    ]
    return {
        "contents": [{"role": "user", "parts": [{"text": PROMPT}, *images]}],
        "generationConfig": {
            "responseFormat": {
                "text": {"mimeType": "APPLICATION_JSON", "schema": schema_for("gemini")}
            }
        },
    }


def parse_response(body: dict[str, Any]) -> dict[str, Any]:
    """Decode the extraction from the first candidate's text.

    Thought parts, which Gemini may return alongside the answer, are skipped.

    Raises:
        GeminiError: If the prompt was blocked, the candidate stopped for any
            reason other than finishing, or its text is not a JSON object.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        reason = body.get("promptFeedback", {}).get("blockReason", "none given")
        raise GeminiError(f"Gemini returned no candidate (block reason: {reason})")

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish != "STOP":
        raise GeminiError(f"Gemini candidate did not finish (finishReason: {finish})")

    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise GeminiError("Gemini candidate has no text")
    try:
        extraction = json.loads(text)
    except json.JSONDecodeError as e:
        raise GeminiError(f"Gemini answer is not valid JSON: {e}") from e
    if not isinstance(extraction, dict):
        raise GeminiError(
            f"Gemini answer is a JSON {type(extraction).__name__}, not an object"
        )
    return extraction


def extract(
    pdf_path: Path, *, client: httpx.Client | None = None, model: str = GEMINI_MODEL
) -> ModelResult:
    """Extract one sitrep with Gemini.

    Args:
        pdf_path: The sitrep PDF.
        client: Client to send through. Tests pass one built on
            `httpx.MockTransport`. A short-lived client is used if None.
        model: Gemini model id to request.

    Returns:
        The extraction with the requested and returned model ids, the usage
        metadata and the full response body.

    Raises:
        GeminiError: If `GEMINI_API_KEY` is unset.
        UnusableAnswer: If Gemini answered, billing the call, but the answer
            holds no usable extraction (see `parse_response`). It carries
            the response body.
        httpx.HTTPError: If the request fails after retries.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise GeminiError(f"{API_KEY_ENV} is not set")

    pages = render_pages(pdf_path)
    logger.info("Sending %d pages of %s to %s", len(pages), pdf_path.name, model)
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    response = fetch.post(
        API_URL.format(model=model),
        json=build_request(pages),
        headers={"x-goog-api-key": key},
        client=client,
        timeout=TIMEOUT_S,
    )
    body = response.json()
    try:
        extraction = parse_response(body)
    except GeminiError as e:
        raise UnusableAnswer(
            str(e), model_id_requested=model, requested_at=requested_at, response=body
        ) from e
    return ModelResult(
        extraction_raw=extraction,
        model_id_requested=model,
        model_id_returned=body.get("modelVersion"),
        usage=body.get("usageMetadata", {}),
        requested_at=requested_at,
        response=body,
    )
