from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from visionbench.models.base import FatalModelError, SampleModelError, TransientModelError
from visionbench.models.remote import GeminiImageBackend, OpenAIGPTImageBackend, VolcanoArkSeedreamBackend, combine_prompts


def image_b64() -> str:
    output = io.BytesIO()
    Image.new("RGB", (7, 5), "red").save(output, "PNG")
    return base64.b64encode(output.getvalue()).decode()


def fake_response(payload):
    def request(method, url, headers, body):
        return 200, json.dumps(payload).encode()
    return request


def test_prompt_combination_is_stable():
    assert combine_prompts("system", "sample") == "system\n\nsample"
    assert combine_prompts("", "sample") == "sample"


@pytest.mark.parametrize("backend,payload", [
    (OpenAIGPTImageBackend("gpt-image", "key"), lambda encoded: {"data": [{"b64_json": encoded}]}),
    (GeminiImageBackend("gemini-model", "key"), lambda encoded: {"candidates": [{"content": {"parts": [{"inlineData": {"data": encoded, "mimeType": "image/png"}}]}}]}),
    (VolcanoArkSeedreamBackend("seedream-model", "key"), lambda encoded: {"data": [{"b64_json": encoded}]}),
])
def test_adapters_return_image_and_do_not_change_input(backend, payload):
    captured = {}
    def request(method, url, headers, body):
        captured.update(method=method, url=url, headers=headers, body=body)
        return 200, json.dumps(payload(image_b64())).encode()
    backend.request = request
    source = Image.new("RGBA", (13, 9), "blue")
    result = backend.generate(source, "sample", "system", 42)
    assert source.size == (13, 9)
    assert result.image.size == (7, 5)
    assert result.metadata["input_size"] == [13, 9]
    assert result.metadata["output_size"] == [7, 5]
    assert captured["method"] == "POST"


@pytest.mark.parametrize("status,error", [(401, FatalModelError), (404, FatalModelError), (429, TransientModelError), (503, TransientModelError), (400, SampleModelError)])
def test_status_classification(status, error):
    backend = OpenAIGPTImageBackend("gpt-image", "key", request=lambda *args: (status, b'{"error":{"message":"bad"}}'))
    with pytest.raises(error):
        backend.generate(Image.new("RGB", (2, 2)), "p", "", None)


def test_gemini_request_contains_image_and_prompt():
    body = {}
    def request(method, url, headers, raw):
        body.update(json.loads(raw))
        return 200, json.dumps({"candidates": [{"content": {"parts": [{"inlineData": {"data": image_b64()}}]}}]}).encode()
    GeminiImageBackend("gemini-x", "secret", request=request).generate(Image.new("RGB", (3, 4)), "p", "s", 5)
    assert body["contents"][0]["parts"][0]["text"] == "s\n\np"
    assert "seed" not in body["generationConfig"]


def test_openai_gpt_image_omits_unsupported_response_format_and_seed():
    body = b""
    def request(method, url, headers, raw):
        nonlocal body
        body = raw
        return 200, json.dumps({"data": [{"b64_json": image_b64()}]}).encode()
    result = OpenAIGPTImageBackend("gpt-image-1", "key", request=request).generate(
        Image.new("RGB", (3, 4)), "p", "s", 7
    )
    assert b'response_format' not in body
    assert b'name="seed"' not in body
    assert result.metadata["seed"] == 7
    assert result.metadata["seed_applied"] is False


def test_openai_uses_input_dimensions_by_default_and_explicit_size_when_requested():
    requests = []
    def request(method, url, headers, raw):
        requests.append(raw)
        return 200, json.dumps({"data": [{"b64_json": image_b64()}]}).encode()
    source = Image.new("RGB", (321, 123))
    default = OpenAIGPTImageBackend("gpt-image-1", "key", request=request).generate(source, "p", "", None)
    explicit = OpenAIGPTImageBackend("gpt-image-1", "key", size="640x480", request=request).generate(source, "p", "", None)
    assert b"321x123" in requests[0]
    assert b"640x480" in requests[1]
    assert default.metadata["request"]["size"] == "321x123"
    assert explicit.metadata["request"]["size"] == "640x480"
    assert default.metadata["requested_output_size"] == "321x123"
    assert default.metadata["output_size"] == [7, 5]


def test_ark_does_not_send_unsupported_seed_and_keeps_usage_metadata():
    request_payload = {}
    def request(method, url, headers, raw):
        request_payload.update(json.loads(raw))
        return 200, json.dumps({"data": [{"b64_json": image_b64(), "size": "7x5"}], "usage": {"total_tokens": 12}}).encode()
    result = VolcanoArkSeedreamBackend("dola-seedream-5-0-pro-260628", "key", request=request).generate(
        Image.new("RGB", (3, 4)), "p", "s", 7
    )
    assert "seed" not in request_payload
    assert result.metadata["seed"] == 7
    assert result.metadata["seed_applied"] is False
    assert result.metadata["provider_usage"] == {"total_tokens": 12}
    assert result.metadata["provider_output_size"] == "7x5"


def test_ark_uses_input_dimensions_unless_size_is_overridden():
    requests = []
    def request(method, url, headers, raw):
        requests.append(json.loads(raw))
        return 200, json.dumps({"data": [{"b64_json": image_b64()}]}).encode()
    source = Image.new("RGB", (321, 123))
    default = VolcanoArkSeedreamBackend("dola-seedream-5-0-pro-260628", "key", request=request).generate(source, "p", "", None)
    explicit = VolcanoArkSeedreamBackend("dola-seedream-5-0-pro-260628", "key", size="640x480", request=request).generate(source, "p", "", None)
    assert requests[0]["size"] == "321x123"
    assert requests[1]["size"] == "640x480"
    assert default.metadata["request"]["size"] == "321x123"
    assert explicit.metadata["request"]["size"] == "640x480"
    assert default.metadata["requested_output_size"] == "321x123"
    assert default.metadata["output_size"] == [7, 5]
