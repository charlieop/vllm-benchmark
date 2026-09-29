"""Deterministic infrastructure smoke-test backend."""

from __future__ import annotations

from PIL import Image

from .base import GenerationOutput


class DummyBackend:
    def __init__(self, model: str = "white-background-dummy", **_: object) -> None:
        self.model = model

    def generate(self, image: Image.Image, prompt: str, system_prompt: str, seed: int | None) -> GenerationOutput:
        # Copy first; the benchmark must never mutate the loader's image.
        output = image.convert("RGB").copy()
        pixels = output.load()
        # The toy data use a saturated foreground against a non-white background.
        for y in range(output.height):
            for x in range(output.width):
                red, green, blue = pixels[x, y]
                if max(red, green, blue) - min(red, green, blue) < 12:
                    pixels[x, y] = (255, 255, 255)
        return GenerationOutput(output, {"model": self.model, "seed": seed, "dummy": True})

    def close(self) -> None:
        return None
