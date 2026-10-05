"""Shared loader for the surface normal benchmarks (NYUv2, iBims-1, DIODE).

Data layout follows the Marigold normals evaluation release, which is also what
RINO compares against (DSINE / Marigold / StableNormal numbers):

    raw/surface_normal/nyuv2/test/000000_img.png, 000000_normal.npy, ...
    raw/surface_normal/ibims/ibims/corridor_01_img.png, corridor_01_normal.npy, ...
    raw/surface_normal/diode/val/indoors/scene_xxxxx/scan_xxxxx/*.png, *_normal.npy

Run ``scripts/download_surface_normal.sh`` to fetch the data. It also places each Marigold
split file (e.g. ``nyuv2_test.txt``) inside its dataset folder; every line lists
``<rgb relative path> <normal .npy relative path>``, relative to that folder.

GT ``*_normal.npy`` is an H x W x 3 float array. Pixels whose GT vector is all zeros
are invalid and are excluded by the evaluator (the Marigold rule).

The thin per-dataset loaders (surface_normal_nyuv2.py, ...) import this file by path,
because the runner loads task files by path rather than as a package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from visionbench.transforms import preprocess_image
from visionbench.types import Sample

DEFAULT_PROMPT = (
    "Convert this photo into a surface normal map of exactly the same scene. "
    "Keep the camera viewpoint, layout, object boundaries and image size unchanged. "
    "Color every pixel by the 3D direction its surface faces, in the standard RGB "
    "normal-map encoding (R = x, G = y, B = z, each mapped from [-1, 1] to [0, 255]). "
    "Output only the normal map: no text, no shading, no original photo colors."
)


def _read_split(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_number}: expected '<rgb> <normal.npy>'")
        rows.append((parts[0], parts[1]))
    return rows


def _sample_id(rgb_rel: str, dataset_name: str) -> str:
    stem = rgb_rel[:-len(Path(rgb_rel).suffix)] if Path(rgb_rel).suffix else rgb_rel
    stem = stem.removesuffix("_img")
    return f"{dataset_name}/{stem}"


class SurfaceNormalDataset:
    """Random-access dataset; images and GT are read lazily per item."""

    def __init__(self, *, dataset_name: str, root: Path, split_file: Path, prompt: str,
                 max_size: Any, shrink_mode: str, size_multiple: int,
                 subset: str | None = None, sample_stride: int = 1, max_samples: int | None = None):
        if not root.exists():
            raise FileNotFoundError(
                f"{dataset_name}: data not found at {root}. Run scripts/download_surface_normal.sh first.")
        if not split_file.is_file():
            raise FileNotFoundError(
                f"{dataset_name}: split file not found: {split_file}. Run scripts/download_surface_normal.sh first.")
        rows = _read_split(split_file)
        if subset:
            rows = [row for row in rows if row[0].split("/", 1)[0] == subset]
        if sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")
        rows = rows[::sample_stride]
        if max_samples is not None:
            rows = rows[:int(max_samples)]
        if not rows:
            raise ValueError(f"{dataset_name}: split {split_file} selected no samples (subset={subset!r})")
        self.dataset_name = dataset_name
        self.root = root
        self.rows = rows
        self.prompt = prompt
        self.max_size = tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size
        self.shrink_mode = shrink_mode
        self.size_multiple = size_multiple

    def __len__(self) -> int:
        return len(self.rows)

    def _open_rgb(self, rel: str) -> Image.Image:
        with Image.open(self.root / rel) as source:
            return source.convert("RGB")

    def __getitem__(self, index: int) -> Sample:
        rgb_rel, normal_rel = self.rows[index]
        original = self._open_rgb(rgb_rel)
        image, transform, parameters = preprocess_image(
            original, max_size=self.max_size, shrink_mode=self.shrink_mode, size_multiple=self.size_multiple)
        return Sample(
            sample_id=_sample_id(rgb_rel, self.dataset_name),
            image=image,
            prompt=self.prompt,
            # GT stays on disk; the evaluator loads it so aggregation stays memory bounded.
            ground_truth={
                "normal_root": str(self.root),
                "normal_rel": normal_rel,
                "gt_size": list(original.size),
            },
            metadata={
                "dataset": self.dataset_name,
                "rgb_rel": rgb_rel,
                "transform": transform.to_dict(),
                "preprocessing": parameters,
            },
        )


def build(config: dict, *, dataset_name: str, default_split: str, default_subset: str | None = None) -> SurfaceNormalDataset:
    raw_dir = Path(config["raw_dir"])
    return SurfaceNormalDataset(
        dataset_name=dataset_name,
        root=raw_dir,
        split_file=raw_dir / config.get("split_file", default_split),
        prompt=config.get("prompt", DEFAULT_PROMPT),
        max_size=config.get("max_size", (1024, 1024)),
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=config.get("size_multiple", 32),
        subset=config.get("subset", default_subset),
        sample_stride=int(config.get("sample_stride", 1)),
        max_samples=config.get("max_samples"),
    )
