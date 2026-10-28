"""Tests for the Mistral OCR annotation client."""

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ebola_2026_outbreak import drc_2026_sitrep_mistral as mistral
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    PROMPT,
    UnusableAnswer,
    load_gazetteer,
    normalise,
    schema_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sitrep_models"
# Recorded from mistral-ocr-4-1 on SitRep 157, per-page layout blocks removed.
RESPONSE = json.loads((FIXTURE / "mistral_157_response.json").read_text())
KEY = "secret-mistral-key"
PDF = b"%PDF-1.7 not a real report"


@pytest.fixture
def pdf_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(mistral.API_KEY_ENV, KEY)
    path = tmp_path / "SitRep_157.pdf"
    path.write_bytes(PDF)
    return path


def client_for(body: dict[str, Any], seen: list[httpx.Request]) -> httpx.Client:
    """A client whose server answers every request with `body`."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_extract_sends_pinned_annotation_request(pdf_path: Path) -> None:
    seen: list[httpx.Request] = []
    mistral.extract(pdf_path, client=client_for(RESPONSE, seen))

    [request] = seen
    assert request.method == "POST"
    assert str(request.url) == "https://api.mistral.ai/v1/ocr"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url)

    body = json.loads(request.content)
    assert body["model"] == "mistral-ocr-4-1"
    document = body["document"]
    assert document["type"] == "document_url"
    prefix, b64 = document["document_url"].split(",", 1)
    assert prefix == "data:application/pdf;base64"
    assert base64.b64decode(b64) == PDF
    assert body["include_image_base64"] is False
    assert body["table_format"] == "markdown"
    assert body["document_annotation_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "sitrep",
            "strict": True,
            "schema": schema_for("mistral"),
        },
    }
    assert body["document_annotation_prompt"] == PROMPT


def test_extract_parses_recorded_response(pdf_path: Path) -> None:
    result = mistral.extract(pdf_path, client=client_for(RESPONSE, []))

    assert result.extraction_raw == json.loads(RESPONSE["document_annotation"])
    assert result.model_id_requested == "mistral-ocr-4-1"
    assert result.model_id_returned == "mistral-ocr-4-1"
    assert result.usage["pages_processed"] == 13
    assert result.response == RESPONSE

    normalised = normalise(result.extraction_raw, load_gazetteer())
    assert normalised.extraction.sitrep_no == 157
    assert normalised.extraction.report_date == "2026-10-18"
    assert sum(k.section == "zones" for k in normalised.extraction.rows) == 67
    assert normalised.name_flags == []


def test_extract_refuses_without_key(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(mistral.API_KEY_ENV)
    seen: list[httpx.Request] = []
    with pytest.raises(RuntimeError, match=mistral.API_KEY_ENV):
        mistral.extract(pdf_path, client=client_for(RESPONSE, seen))
    assert seen == []


@pytest.mark.parametrize(
    ("annotation", "match"),
    [
        (None, "no document_annotation"),
        ("", "no document_annotation"),
        ('{"sitrep_no": 157', "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
    ],
)
def test_extract_rejects_bad_annotation(
    pdf_path: Path, annotation: str | None, match: str
) -> None:
    body = {**RESPONSE, "document_annotation": annotation}
    with pytest.raises(UnusableAnswer, match=match) as error:
        mistral.extract(pdf_path, client=client_for(body, []))
    assert KEY not in str(error.value)
    # The call was billed: the error carries what extraction records.
    assert error.value.response == body
    assert error.value.model_id_requested == mistral.MISTRAL_OCR_MODEL


def test_extract_raises_on_rejected_request(pdf_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid schema"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with pytest.raises(httpx.HTTPStatusError) as error:
        mistral.extract(pdf_path, client=client)
    assert KEY not in str(error.value)
