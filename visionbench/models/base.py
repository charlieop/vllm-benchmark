from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass
class GenerationOutput:
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelError(RuntimeError):
    """Base backend error."""


class TransientModelError(ModelError):
    """Retryable backend error."""


class FatalModelError(ModelError):
    """A run-wide problem, such as credentials, configuration, or GPU OOM."""


class SampleModelError(ModelError):
    """A permanent error affecting one sample."""


class ModelBackend(Protocol):
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str,
        seed: int | None,
    ) -> GenerationOutput: ...

    def close(self) -> None: ...
