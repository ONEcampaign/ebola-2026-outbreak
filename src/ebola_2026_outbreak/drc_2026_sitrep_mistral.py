"""Extract a DRC sitrep with Mistral OCR document annotation.

The whole PDF goes to Mistral's OCR endpoint as a base64 data URL. The model
reads every page and returns, besides the per-page markdown, a
`document_annotation`, a JSON string shaped by `schema_for("mistral")` and
written under `PROMPT`. That string is parsed into the raw extraction, and
`drc_2026_sitrep_schema.normalise` keys it on canonical names.

The model is pinned to `mistral-ocr-4-1`. Mistral echoes the requested id in
the response's `model`, so a silent change behind the pin shows only in the
output and in `usage_info.pages_processed`, both kept with the response.
"""

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ebola_2026_outbreak import fetch
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    PROMPT,
    ModelResult,
    UnusableAnswer,
    schema_for,
)

MISTRAL_OCR_MODEL = "mistral-ocr-4-1"
OCR_URL = "https://api.mistral.ai/v1/ocr"
API_KEY_ENV = "MISTRAL_API_KEY"
# Annotating the 13 pages of SitRep 157 took 112 s.
TIMEOUT_S = 600.0


def request_body(pdf: bytes, *, model: str = MISTRAL_OCR_MODEL) -> dict[str, Any]:
    """The OCR request for one PDF, asking for the sitrep annotation."""
    data_url = "data:application/pdf;base64," + base64.b64encode(pdf).decode()
    return {
        "model": model,
        "document": {"type": "document_url", "document_url": data_url},
        "include_image_base64": False,
        "table_format": "markdown",
        "document_annotation_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sitrep",
                "strict": True,
                "schema": schema_for("mistral"),
            },
        },
        "document_annotation_prompt": PROMPT,
    }


def parse_annotation(response: dict[str, Any]) -> dict[str, Any]:
    """Parse the `document_annotation` JSON string of an OCR response.

    Raises:
        ValueError: If the annotation is missing, does not parse, or is not a
            JSON object.
    """
    annotation = response.get("document_annotation")
    if not isinstance(annotation, str) or not annotation.strip():
        raise ValueError("Mistral OCR response has no document_annotation")
    try:
        raw = json.loads(annotation)
    except json.JSONDecodeError as e:
        raise ValueError(f"Mistral document_annotation is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(
            f"Mistral document_annotation is not a JSON object: {type(raw).__name__}"
        )
    return raw


def extract(
    pdf_path: Path,
    *,
    client: httpx.Client | None = None,
    model: str = MISTRAL_OCR_MODEL,
) -> ModelResult:
    """Ask Mistral OCR for a sitrep's structured annotation.

    The key is read from the `MISTRAL_API_KEY` environment variable and sent
    in the `Authorization` header only.

    Args:
        pdf_path: The sitrep PDF.
        client: Client to send through. Tests pass one built on
            `httpx.MockTransport`. A short-lived client is used if None.
        model: Model id to request.

    Returns:
        The parsed annotation, the requested and echoed model ids, the
        `usage_info`, and the full response body (per-page markdown and
        tables included, about 120 KB a report; images are not requested).

    Raises:
        RuntimeError: If `MISTRAL_API_KEY` is not set.
        httpx.HTTPError: If the request fails after retries.
        UnusableAnswer: If Mistral answered, billing the call, but the
            response carries no annotation, or one that is not a JSON
            object. It carries the response body.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set")

    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    response = fetch.post(
        OCR_URL,
        json=request_body(pdf_path.read_bytes(), model=model),
        headers={"Authorization": f"Bearer {key}"},
        client=client,
        timeout=TIMEOUT_S,
    ).json()

    try:
        extraction = parse_annotation(response)
    except ValueError as e:
        raise UnusableAnswer(
            str(e),
            model_id_requested=model,
            requested_at=requested_at,
            response=response,
        ) from e
    return ModelResult(
        extraction_raw=extraction,
        model_id_requested=model,
        model_id_returned=response.get("model"),
        usage=response.get("usage_info") or {},
        requested_at=requested_at,
        response=response,
    )
