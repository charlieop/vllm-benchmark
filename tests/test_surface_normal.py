import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionbench.plugins import load_dataset
from visionbench.types import Sample

ROOT = Path(__file__).resolve().parents[1]


def _module(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _module("tasks/evaluators/surface_normal.py")
agg = _module("tasks/aggregators/surface_normal.py")


def _smooth_normals(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = np.stack([np.sin(xx / w * 2.0) * 0.6, np.cos(yy / h * 2.0) * 0.5, np.ones_like(xx)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _encode(normals, order=(0, 1, 2), signs=(1, 1, 1)):
    """Make an RGB image whose channel c holds sign * GT axis -- i.e. inverse of rgb_to_gt."""
    rgb = np.zeros_like(normals)
    for gt_axis, (channel, sign) in enumerate(zip(order, signs)):
        rgb[..., channel] = sign * normals[..., gt_axis]
    return Image.fromarray(np.clip(np.round((rgb + 1) * 127.5), 0, 255).astype(np.uint8))


def _sample(tmp_path, gt, transform=None):
    np.save(tmp_path / "x_normal.npy", gt)
    meta = {} if transform is None else {"transform": transform}
    return Sample("s", Image.new("RGB", (gt.shape[1], gt.shape[0])), "p",
                  {"normal_root": str(tmp_path), "normal_rel": "x_normal.npy", "is_tar": False}, meta)


def test_perfect_prediction_is_near_zero(tmp_path):
    gt = _smooth_normals(48, 64)
    result = ev.score(_encode(gt), _sample(tmp_path, gt), {})
    assert result["mean_angular_error"] < 0.6  # 8-bit quantization only
    assert result["pct_within_11_25"] == 100.0
    assert result["degenerate_pixel_pct"] == 0.0


def test_convention_is_applied(tmp_path):
    gt = _smooth_normals(32, 32)
    image = _encode(gt, signs=(1, -1, -1))  # model encodes y,z flipped
    sample = _sample(tmp_path, gt)
    assert ev.score(image, sample, {})["mean_angular_error"] > 90
    assert ev.score(image, sample, {"rgb_to_gt": ["+r", "-g", "-b"]})["mean_angular_error"] < 0.6
    swapped = _encode(gt, order=(1, 0, 2))  # R holds y, G holds x
    assert ev.score(swapped, sample, {"rgb_to_gt": "+g +r +b"})["mean_angular_error"] < 0.6
    assert len(ev.all_conventions()) == 48


def test_invalid_gt_excluded_and_degenerate_kept(tmp_path):
    gt = _smooth_normals(20, 20)
    gt[:10] = 0.0  # invalid GT rows
    rgb = np.asarray(_encode(gt)).copy()
    rgb[10:15] = 128  # mid-gray: no direction, inside valid GT
    result = ev.score(Image.fromarray(rgb), _sample(tmp_path, gt), {})
    assert result["degenerate_pixel_pct"] == pytest.approx(50.0, abs=1e-6)
    assert result["mean_angular_error"] == pytest.approx(45.0, abs=0.5)
    assert result["median_angular_error"] > 0  # between the two halves


def test_output_size_mismatch_is_resized(tmp_path):
    gt = _smooth_normals(48, 64)
    big = _encode(_smooth_normals(96, 128))
    assert ev.score(big, _sample(tmp_path, gt), {})["mean_angular_error"] < 1.5


def test_loader_crop_marks_uncovered_pixels(tmp_path):
    from visionbench.transforms import preprocess_image
    gt = _smooth_normals(40, 70)
    _, transform, _ = preprocess_image(Image.new("RGB", (70, 40)), size_multiple=32)  # -> 64x32 center crop
    pred = _encode(gt).crop((3, 4, 67, 36))
    result = ev.score(pred, _sample(tmp_path, gt, transform.to_dict()), {})
    expected = 100 * (1 - 64 * 32 / (70 * 40))
    assert result["degenerate_pixel_pct"] == pytest.approx(expected, abs=1e-6)


def test_loader_reads_split_and_subset(tmp_path):
    raw = tmp_path / "raw"
    for sub in ("indoor/a", "outdoor/b"):
        (raw / sub).mkdir(parents=True)
        Image.new("RGB", (100, 70), "red").save(raw / sub / "im.png")
        np.save(raw / sub / "im_normal.npy", _smooth_normals(70, 100))
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "diode_test.txt").write_text(
        "indoor/a/im.png indoor/a/im_normal.npy\noutdoor/b/im.png outdoor/b/im_normal.npy\n")
    loader = ROOT / "tasks/dataloaders/surface_normal_diode.py"
    dataset = load_dataset(loader, {"raw_dir": str(raw), "assets_dir": str(assets), "size_multiple": 32})
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.sample_id == "diode/indoor/a/im"
    assert sample.image.size == (96, 64)
    assert sample.ground_truth["gt_size"] == [100, 70]
    outdoor = load_dataset(loader, {"raw_dir": str(raw), "assets_dir": str(assets), "subset": "outdoor"})
    assert outdoor[0].sample_id == "diode/outdoor/b/im"


def test_aggregator_policies():
    good = {name: 10.0 for name in agg.METRIC_NAMES}
    records = [{"metrics": good}, {"metrics": {}}]
    penal = agg.aggregate(records, {})
    assert penal["mean_angular_error"] == pytest.approx(50.0)
    assert penal["pct_within_30"] == pytest.approx(5.0)
    assert penal["samples_failed"] == 1.0
    excl = agg.aggregate(records, {"failed_sample_policy": "exclude"})
    assert excl["mean_angular_error"] == 10.0
