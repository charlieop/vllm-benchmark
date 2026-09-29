"""Small, dependency-free adapters for the hosted image editing APIs.

The adapters deliberately use their providers' HTTPS APIs directly.  A callable
``request`` can be supplied by tests (or an application with its own HTTP
stack); it receives ``(method, url, headers, body)`` and returns
``(status_code, response_bytes)``.  No request is made while constructing a
backend.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .base import FatalModelError, GenerationOutput, SampleModelError, TransientModelError

RequestFn = Callable[[str, str, Mapping[str, str], bytes], tuple[int, bytes]]


def combine_prompts(system_prompt: str, prompt: str) -> str:
    """Use one stable representation on APIs without a system role."""
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


def _default_request(method: str, url: str, headers: Mapping[str, str], body: bytes) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=60) as response:  # nosec B310 - provider endpoint supplied by caller
            return int(response.status), response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise TransientModelError(f"network error contacting provider: {exc}") from exc


def _raise_for_status(provider: str, status: int, payload: bytes) -> None:
    if 200 <= status < 300:
        return
    try:
        message = json.loads(payload.decode("utf-8")).get("error", {}).get("message")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        message = payload.decode("utf-8", "replace")[:500]
    detail = f"{provider} returned HTTP {status}: {message or 'no error detail'}"
    if status in {401, 403, 404}:
        raise FatalModelError(detail)
    if status in {408, 409, 425, 429} or 500 <= status <= 599:
        raise TransientModelError(detail)
    raise SampleModelError(detail)


def _decode_json(provider: str, payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SampleModelError(f"{provider} returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise SampleModelError(f"{provider} returned a non-object JSON response")
    return decoded


def _provider_response_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Keep compact, useful provider fields without persisting image payloads."""
    metadata: dict[str, Any] = {}
    for source, target in (
        ("usage", "provider_usage"),
        ("usageMetadata", "provider_usage"),
        ("model", "provider_model"),
        ("id", "provider_request_id"),
        ("created", "provider_created"),
    ):
        value = data.get(source)
        if value is not None:
            metadata[target] = value
    return metadata


def _image_bytes(image: Image.Image) -> tuple[bytes, str]:
    # Saving to a BytesIO does not mutate the loader-owned image.
    copied = image.copy()
    output = io.BytesIO()
    if copied.mode in {"RGBA", "LA", "P"}:
        copied.save(output, format="PNG")
        return output.getvalue(), "image/png"
    copied.convert("RGB").save(output, format="PNG")
    return output.getvalue(), "image/png"


def _image_from_b64(provider: str, value: str) -> Image.Image:
    try:
        raw = base64.b64decode(value, validate=True)
        result = Image.open(io.BytesIO(raw))
        result.load()
        return result
    except Exception as exc:  # Pillow's decoding exceptions vary by format.
        raise SampleModelError(f"{provider} returned an invalid image payload") from exc


def _image_from_url(provider: str, value: str, request: RequestFn) -> Image.Image:
    try:
        status, payload = request("GET", value, {}, b"")
        _raise_for_status(provider, status, payload)
        result = Image.open(io.BytesIO(payload))
        result.load()
        return result
    except (TransientModelError, FatalModelError, SampleModelError):
        raise
    except Exception as exc:
        raise SampleModelError(f"{provider} returned an invalid image URL") from exc


def _multipart(fields: Mapping[str, str], image: Image.Image) -> tuple[bytes, str]:
    image_data, mime_type = _image_bytes(image)
    boundary = f"----visionbench-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"), b"\r\n",
        ))
    chunks.extend((
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="input.png"\r\n',
        f"Content-Type: {mime_type}\r\n\r\n".encode(), image_data, b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    return b"".join(chunks), boundary


@dataclass
class OpenAIGPTImageBackend:
    """OpenAI Images API image-edit adapter.

    ``size`` optionally overrides the requested output dimensions.  By default
    the input's exact ``WIDTHxHEIGHT`` dimensions are sent to GPT Image, whose
    API supports arbitrary dimensions within its documented limits.  The image
    itself is never resized before upload. ``quality`` and ``output_format``
    are passed only when explicitly set.
    """

    model: str
    api_key: str
    size: str | None = None
    quality: str | None = None
    output_format: str | None = "png"
    endpoint: str = "https://api.openai.com/v1/images/edits"
    request: RequestFn = _default_request

    def generate(self, image: Image.Image, prompt: str, system_prompt: str, seed: int | None) -> GenerationOutput:
        # GPT Image models return base64 image data by default.  Their Images
        # API rejects the legacy ``response_format`` parameter.
        requested_size = self.size or f"{image.width}x{image.height}"
        fields = {"model": self.model, "prompt": combine_prompts(system_prompt, prompt), "size": requested_size}
        if self.quality is not None:
            fields["quality"] = self.quality
        if self.output_format is not None:
            fields["output_format"] = self.output_format
        # GPT Image currently does not expose a seed parameter.  Record it so a
        # run remains auditable without implying provider-side determinism.
        body, boundary = _multipart(fields, image)
        status, response = self.request("POST", self.endpoint, {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }, body)
        _raise_for_status("OpenAI", status, response)
        data = _decode_json("OpenAI", response)
        try:
            item = data["data"][0]
            encoded = item["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SampleModelError("OpenAI response did not contain b64_json image data") from exc
        output = _image_from_b64("OpenAI", encoded)
        metadata = {
            "provider": "openai", "model": self.model, "seed": seed,
            "seed_applied": False,
            "input_size": list(image.size), "requested_output_size": requested_size,
            "output_size": list(output.size),
            "revised_prompt": item.get("revised_prompt"),
            "request": {"size": requested_size, "quality": self.quality, "output_format": self.output_format},
        }
        metadata.update(_provider_response_metadata(data))
        return GenerationOutput(output, metadata)

    def close(self) -> None:
        return None


@dataclass
class GeminiImageBackend:
    """Gemini Developer API image-generation adapter (REST generateContent)."""

    model: str
    api_key: str
    generation_config: Mapping[str, Any] | None = None
    endpoint_base: str = "https://generativelanguage.googleapis.com/v1beta"
    request: RequestFn = _default_request

    def generate(self, image: Image.Image, prompt: str, system_prompt: str, seed: int | None) -> GenerationOutput:
        raw_image, mime_type = _image_bytes(image)
        config: dict[str, Any] = {"responseModalities": ["IMAGE"]}
        if self.generation_config:
            config.update(self.generation_config)
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": combine_prompts(system_prompt, prompt)},
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(raw_image).decode("ascii")}},
            ]}],
            "generationConfig": config,
        }
        endpoint = f"{self.endpoint_base.rstrip('/')}/models/{self.model}:generateContent"
        status, response = self.request("POST", endpoint, {
            "Content-Type": "application/json", "x-goog-api-key": self.api_key,
        }, json.dumps(payload).encode())
        _raise_for_status("Gemini", status, response)
        data = _decode_json("Gemini", response)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            inline = next(
                part.get("inlineData", part.get("inline_data"))
                for part in parts if "inlineData" in part or "inline_data" in part
            )
            encoded = inline["data"]
        except (KeyError, IndexError, StopIteration, TypeError) as exc:
            raise SampleModelError("Gemini response did not contain inline image data") from exc
        output = _image_from_b64("Gemini", encoded)
        metadata = {
            "provider": "gemini", "model": self.model, "seed": seed,
            "seed_applied": False,
            "input_size": list(image.size), "requested_output_size": "provider_default",
            "output_size": list(output.size),
            "response_mime_type": inline.get("mimeType"),
            "request": {"generation_config": config, "size": "provider_default"},
        }
        metadata.update(_provider_response_metadata(data))
        return GenerationOutput(output, metadata)

    def close(self) -> None:
        return None


@dataclass
class VolcanoArkSeedreamBackend:
    """Volcano Ark Seedream image-to-image adapter.

    Ark's OpenAI-compatible image endpoint accepts a base64 data URL in the
    ``image`` field.  ``endpoint`` is configurable for regions and gateways.
    """

    model: str
    api_key: str
    endpoint: str = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    response_format: str = "url"
    size: str | None = None
    extra_options: Mapping[str, Any] | None = None
    request: RequestFn = _default_request

    def generate(self, image: Image.Image, prompt: str, system_prompt: str, seed: int | None) -> GenerationOutput:
        raw_image, mime_type = _image_bytes(image)
        requested_size = self.size or f"{image.width}x{image.height}"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": combine_prompts(system_prompt, prompt),
            "image": f"data:{mime_type};base64,{base64.b64encode(raw_image).decode('ascii')}",
            "response_format": self.response_format,
            "size": requested_size,
        }
        # The documented Seedream 5 Pro image-generation parameters do not
        # include ``seed``.  Do not transmit an unsupported option; retain the
        # requested seed in output metadata for reproducibility reporting.
        if self.extra_options:
            payload.update(self.extra_options)
        status, response = self.request("POST", self.endpoint, {
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
        }, json.dumps(payload).encode())
        _raise_for_status("Volcano Ark", status, response)
        data = _decode_json("Volcano Ark", response)
        try:
            item = data["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise SampleModelError("Volcano Ark response did not contain image data") from exc
        if isinstance(item.get("b64_json"), str):
            output = _image_from_b64("Volcano Ark", item["b64_json"])
        elif isinstance(item.get("url"), str):
            output = _image_from_url("Volcano Ark", item["url"], self.request)
        else:
            raise SampleModelError("Volcano Ark response did not contain b64_json or URL image data")
        metadata = {
            "provider": "volcano_ark", "model": self.model, "seed": seed,
            "seed_applied": False,
            "input_size": list(image.size), "requested_output_size": requested_size,
            "output_size": list(output.size),
            "url": item.get("url"),
            "request": {
                "response_format": self.response_format,
                "size": requested_size,
                "extra_options": dict(self.extra_options or {}),
            },
        }
        metadata.update(_provider_response_metadata(data))
        if item.get("size") is not None:
            metadata["provider_output_size"] = item["size"]
        return GenerationOutput(output, metadata)

    def close(self) -> None:
        return None
