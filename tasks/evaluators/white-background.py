"""Fraction of output pixels that are exactly white."""

from PIL import Image

from visionbench.types import Sample


METRICS = {
    "white_fraction": {"unit": "fraction", "higher_is_better": True},
}


def score(generated_image: Image.Image, sample: Sample, config: dict) -> dict[str, float]:
    rgb = generated_image.convert("RGB")
    minimum_channel = config.get("white_min_channel", 240)
    if not isinstance(minimum_channel, int) or not 0 <= minimum_channel <= 255:
        raise ValueError("white_min_channel must be an integer from 0 through 255")

    white = sum(
        all(channel >= minimum_channel for channel in rgb.getpixel((x, y)))
        for y in range(rgb.height)
        for x in range(rgb.width)
    )
    return {"white_fraction": white / (rgb.width * rgb.height)}
