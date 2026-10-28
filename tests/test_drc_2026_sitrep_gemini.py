"""Tests for the Gemini sitrep extractor. No test reaches the network."""

import base64
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pypdfium2
import pytest
from PIL import Image

from ebola_2026_outbreak import drc_2026_sitrep_gemini as gemini
from ebola_2026_outbreak.drc_2026_sitrep_schema import (
    PROMPT,
    UnusableAnswer,
    load_gazetteer,
    normalise,
    schema_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sitrep_models" / "gemini_response.json"
KEY = "test-gemini-key-123"
PAGE_SIZE_PT = (200, 100)


@pytest.fixture
def response_body() -> dict[str, Any]:
    """The recorded Gemini response for SitRep 96."""
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    """A blank two-page PDF."""
    doc = pypdfium2.PdfDocument.new()
    for _ in range(2):
        doc.new_page(*PAGE_SIZE_PT)
    path = tmp_path / "sitrep.pdf"
    doc.save(path)
    return path


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set the Gemini API key in the environment and return it."""
    monkeypatch.setenv(gemini.API_KEY_ENV, KEY)
    return KEY


def client_for(body: dict[str, Any], seen: list[httpx.Request]) -> httpx.Client:
    """A client whose server answers every request with `body`."""

    def handle(request: httpx.Request) -> httpx.Response:
        """Record the request and answer with `body`."""
        seen.append(request)
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_render_pages_gives_unquantised_png_per_page(pdf_path: Path) -> None:
    pages = gemini.render_pages(pdf_path)

    assert len(pages) == 2
    scale = gemini.RENDER_DPI / 72
    for png in pages:
        image = Image.open(io.BytesIO(png))
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (
            round(PAGE_SIZE_PT[0] * scale),
            round(PAGE_SIZE_PT[1] * scale),
        )


def test_extract_request_shape(
    pdf_path: Path, api_key: str, response_body: dict[str, Any]
) -> None:
    seen: list[httpx.Request] = []
    gemini.extract(pdf_path, client=client_for(response_body, seen))

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:generateContent"
    )
    assert request.headers["x-goog-api-key"] == api_key
    assert api_key not in str(request.url)

    body = json.loads(request.content)
    (content,) = body["contents"]
    assert content["role"] == "user"
    text, *images = content["parts"]
    assert text == {"text": PROMPT}
    assert [base64.b64decode(p["inlineData"]["data"]) for p in images] == (
        gemini.render_pages(pdf_path)
    )
    assert {p["inlineData"]["mimeType"] for p in images} == {"image/png"}
    assert body["generationConfig"]["responseFormat"]["text"] == {
        "mimeType": "APPLICATION_JSON",
        "schema": schema_for("gemini"),
    }


def test_extract_parses_answer_and_records_model_version(
    pdf_path: Path, api_key: str, response_body: dict[str, Any]
) -> None:
    response_body["modelVersion"] = "gemini-3.8-flash-001"
    result = gemini.extract(pdf_path, client=client_for(response_body, []))

    answer = response_body["candidates"][0]["content"]["parts"][0]["text"]
    assert result.extraction_raw == json.loads(answer)
    assert result.model_id_requested == "gemini-3.8-flash"
    assert result.model_id_returned == "gemini-3.8-flash-001"
    assert result.usage == response_body["usageMetadata"]
    assert result.response == response_body
    assert result.requested_at.endswith("+00:00")
    assert normalise(result.extraction_raw, load_gazetteer()).name_flags == []


def test_parse_response_skips_thought_parts(response_body: dict[str, Any]) -> None:
    parts = response_body["candidates"][0]["content"]["parts"]
    parts.insert(0, {"text": "Reading the zone table first.", "thought": True})

    assert gemini.parse_response(response_body) == json.loads(parts[1]["text"])


def test_extract_without_key_fails_before_sending(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)
    seen: list[httpx.Request] = []

    with pytest.raises(gemini.GeminiError, match=gemini.API_KEY_ENV):
        gemini.extract(pdf_path, client=client_for({}, seen))
    assert seen == []


def _blocked(body: dict[str, Any]) -> None:
    """Corrupt `body` into a response blocked by the safety filter."""
    body.pop("candidates")
    body["promptFeedback"] = {"blockReason": "SAFETY"}


def _stopped_early(body: dict[str, Any]) -> None:
    """Corrupt `body` into a candidate that stopped at the token limit."""
    body["candidates"][0]["finishReason"] = "MAX_TOKENS"


def _no_parts(body: dict[str, Any]) -> None:
    """Corrupt `body` into a candidate with no content parts."""
    body["candidates"][0]["content"]["parts"] = []


def _not_json(body: dict[str, Any]) -> None:
    """Corrupt `body` into a candidate whose text is truncated JSON."""
    body["candidates"][0]["content"]["parts"][0]["text"] = '{"sitrep_no": 96,'


def _json_array(body: dict[str, Any]) -> None:
    """Corrupt `body` into a candidate whose text is a JSON array."""
    body["candidates"][0]["content"]["parts"][0]["text"] = "[]"


@pytest.mark.parametrize(
    ("corrupt", "code"),
    [
        (_blocked, "SAFETY"),
        (_stopped_early, "MAX_TOKENS"),
        (_no_parts, None),
        (_not_json, None),
        (_json_array, None),
    ],
)
def test_extract_rejects_unusable_answer(
    pdf_path: Path,
    api_key: str,
    response_body: dict[str, Any],
    corrupt: Callable[[dict[str, Any]], None],
    code: str | None,
) -> None:
    corrupt(response_body)

    with pytest.raises(UnusableAnswer, match=code) as error:
        gemini.extract(pdf_path, client=client_for(response_body, []))
    assert api_key not in str(error.value)
    # The call was billed: the error carries what extraction records.
    assert isinstance(error.value.__cause__, gemini.GeminiError)
    assert error.value.response == response_body
    assert error.value.model_id_requested == gemini.GEMINI_MODEL
