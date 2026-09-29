from PIL import Image

from visionbench.config import load_config
from visionbench.plugins import load_dataset
from visionbench.transforms import CoordinateTransform


def test_example_loader_scales_and_aligns_to_16(tmp_path):
    Image.new("RGB", (101, 77), "red").save(tmp_path / "odd.png")
    config = load_config("config_dummy")
    loader = config.scoped_path("dataset", "white_background.py")
    dataset = load_dataset(loader, {
        "raw_dir": str(tmp_path), "max_size": [80, 80],
        "shrink_mode": "scale", "size_multiple": 16,
    })
    sample = dataset[0]
    assert sample.image.size == (80, 48)
    assert sample.metadata["preprocessing"]["size_multiple"] == 16
    transform = CoordinateTransform.from_dict(sample.metadata["transform"])
    assert transform.original_size == (101, 77)
    assert transform.output_size == sample.image.size
