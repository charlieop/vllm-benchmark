"""Fraction of output pixels that are exactly white."""

from PIL import Image

from visionbench.types import Sample


METRICS = {
    "white_fraction": {"unit": "fraction", "higher_is_better": True},
}


def score(generated_image: Image.Image, sample: Sample, config: dict) -> dict[str, float]:
    rgb = generated_image.convert("RGB")
    white = sum(rgb.getpixel((x, y)) == (255, 255, 255)
                for y in range(rgb.height) for x in range(rgb.width))
    return {"white_fraction": white / (rgb.width * rgb.height)}
