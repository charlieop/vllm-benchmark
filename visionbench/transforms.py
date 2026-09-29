"""Opt-in image preprocessing and annotation-aware coordinate transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Sequence

from PIL import Image

SizeLimit = int | tuple[int, int] | None


def _resampling(name: str) -> Image.Resampling:
    return getattr(Image.Resampling, name)


@dataclass(frozen=True)
class CoordinateTransform:
    """An affine mapping from original image coordinates to processed coordinates."""

    original_size: tuple[int, int]
    source_crop: tuple[int, int, int, int]
    scaled_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: tuple[float, float]
    final_crop: tuple[int, int, int, int]

    def __call__(self, x: float | Sequence[float], y: float | None = None) -> tuple[float, float]:
        """Map either ``transform(x, y)`` or ``transform((x, y))``."""
        point = x if y is None else (x, y)
        return self.forward(point)

    def forward(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = point
        left, top, _, _ = self.source_crop
        offset_x, offset_y, _, _ = self.final_crop
        sx, sy = self.scale
        return ((x - left) * sx - offset_x, (y - top) * sy - offset_y)

    forward_point = forward

    def inverse(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = point
        left, top, _, _ = self.source_crop
        offset_x, offset_y, _, _ = self.final_crop
        sx, sy = self.scale
        return ((x + offset_x) / sx + left, (y + offset_y) / sy + top)

    inverse_point = inverse

    def visible(self, point: Sequence[float]) -> bool:
        x, y = self.forward(point)
        width, height = self.output_size
        return 0 <= x < width and 0 <= y < height

    def forward_points(self, points: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
        return [self.forward(point) for point in points]

    def inverse_points(self, points: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
        return [self.inverse(point) for point in points]

    def forward_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        a, b = self.forward((x0, y0))
        c, d = self.forward((x1, y1))
        return (a, b, c, d)

    def inverse_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        a, b = self.inverse((x0, y0))
        c, d = self.inverse((x1, y1))
        return (a, b, c, d)

    def forward_boxes(self, boxes: Iterable[Sequence[float]]) -> list[tuple[float, float, float, float]]:
        return [self.forward_box(box) for box in boxes]

    def inverse_boxes(self, boxes: Iterable[Sequence[float]]) -> list[tuple[float, float, float, float]]:
        return [self.inverse_box(box) for box in boxes]

    def crop_box_to_visible(self, box: Sequence[float]) -> tuple[float, float, float, float] | None:
        x0, y0, x1, y1 = self.forward_box(box)
        width, height = self.output_size
        x0, x1 = max(0.0, x0), min(float(width), x1)
        y0, y1 = max(0.0, y0), min(float(height), y1)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def transform_image(self, image: Image.Image, *, resample: Image.Resampling) -> Image.Image:
        """Apply this transform to a raster aligned with the original image."""
        if image.size != self.original_size:
            raise ValueError(f"Annotation size {image.size} does not match original size {self.original_size}")
        image = image.crop(self.source_crop)
        image = image.resize(self.scaled_size, resample=resample)
        return image.crop(self.final_crop)

    def transform_mask(self, mask: Image.Image) -> Image.Image:
        return self.transform_image(mask, resample=_resampling("NEAREST"))

    def transform_depth(self, depth: Image.Image) -> Image.Image:
        return self.transform_image(depth, resample=_resampling("BILINEAR"))

    def transform_normals(self, normals: Image.Image) -> Image.Image:
        # Cropping/scaling do not rotate normal vectors; interpolation is sufficient.
        return self.transform_image(normals, resample=_resampling("BILINEAR"))

    forward_mask = transform_mask
    forward_depth = transform_depth
    forward_normals = transform_normals

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoordinateTransform":
        return cls(**{key: tuple(value) for key, value in data.items()})


def _limit(max_size: SizeLimit) -> tuple[int, int] | None:
    if max_size is None:
        return None
    if isinstance(max_size, int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        return max_size, max_size
    if len(max_size) != 2 or max_size[0] <= 0 or max_size[1] <= 0:
        raise ValueError("max_size must be a positive integer or (max_width, max_height)")
    return int(max_size[0]), int(max_size[1])


def preprocess_image(
    image: Image.Image,
    max_size: SizeLimit = None,
    shrink_mode: Literal["crop", "scale"] = "scale",
    size_multiple: int = 1,
) -> tuple[Image.Image, CoordinateTransform, dict[str, Any]]:
    """Shrink an image, then center crop it down to a required size multiple.

    This is deliberately opt-in; benchmark runners should preserve loader images unless
    the task calls this helper itself.
    """
    if shrink_mode not in {"crop", "scale"}:
        raise ValueError("shrink_mode must be 'crop' or 'scale'")
    if not isinstance(size_multiple, int) or size_multiple <= 0:
        raise ValueError("size_multiple must be a positive integer")
    original_size = image.size
    width, height = original_size
    limit = _limit(max_size)
    source_crop = (0, 0, width, height)
    if limit and (width > limit[0] or height > limit[1]) and shrink_mode == "crop":
        crop_width, crop_height = min(width, limit[0]), min(height, limit[1])
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        source_crop = (left, top, left + crop_width, top + crop_height)
    crop_width, crop_height = source_crop[2] - source_crop[0], source_crop[3] - source_crop[1]
    scale = (1.0, 1.0)
    if limit and shrink_mode == "scale" and (width > limit[0] or height > limit[1]):
        factor = min(limit[0] / width, limit[1] / height)
        scaled_size = (max(1, round(width * factor)), max(1, round(height * factor)))
        scale = (scaled_size[0] / width, scaled_size[1] / height)
    else:
        scaled_size = (crop_width, crop_height)
    output_size = (scaled_size[0] // size_multiple * size_multiple, scaled_size[1] // size_multiple * size_multiple)
    if 0 in output_size:
        raise ValueError(f"Image {scaled_size} is smaller than size_multiple={size_multiple}")
    crop_left = (scaled_size[0] - output_size[0]) // 2
    crop_top = (scaled_size[1] - output_size[1]) // 2
    final_crop = (crop_left, crop_top, crop_left + output_size[0], crop_top + output_size[1])
    transform = CoordinateTransform(original_size, source_crop, scaled_size, output_size, scale, final_crop)
    processed = transform.transform_image(image, resample=_resampling("LANCZOS"))
    params = {
        "max_size": limit, "shrink_mode": shrink_mode, "size_multiple": size_multiple,
        "original_size": original_size, "source_crop": source_crop, "scaled_size": scaled_size,
        "final_crop": final_crop, "output_size": output_size, "scale": scale,
    }
    return processed, transform, params


# Short alias useful in task loaders.
preprocess = preprocess_image
