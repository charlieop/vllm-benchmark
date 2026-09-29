"""Four tiny, lazily loaded images for a complete benchmark smoke test."""

from pathlib import Path

from PIL import Image

from visionbench.transforms import preprocess_image
from visionbench.types import Sample


class WhiteBackgroundDataset:
    def __init__(self, root: Path, *, max_size: tuple[int, int] | None, shrink_mode: str, size_multiple: int):
        self.paths = sorted(root.glob("*.png"))
        self.max_size = max_size
        self.shrink_mode = shrink_mode
        self.size_multiple = size_multiple
        if not self.paths:
            raise ValueError(f"No PNG images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Sample:
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        image, transform, parameters = preprocess_image(
            image,
            max_size=self.max_size,
            shrink_mode=self.shrink_mode,
            size_multiple=self.size_multiple,
        )
        return Sample(
            sample_id=path.stem,
            image=image,
            prompt="Change the background to pure white #FFFFFF. Keep the central object unchanged.",
            ground_truth={"target_background": "#FFFFFF"},
            metadata={"source_path": str(path), "transform": transform.to_dict(), "preprocessing": parameters},
        )


def load_dataset(config: dict) -> WhiteBackgroundDataset:
    root = Path(config["raw_dir"])
    max_size = config.get("max_size", (1024, 1024))
    return WhiteBackgroundDataset(
        root,
        max_size=tuple(max_size) if isinstance(max_size, (list, tuple)) else max_size,
        shrink_mode=config.get("shrink_mode", "scale"),
        size_multiple=config.get("size_multiple", 16),
    )
