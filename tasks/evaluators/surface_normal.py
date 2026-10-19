"""Surface normal evaluator (NYUv2 / iBims-1 / DIODE-indoor), RINO / Marigold protocol.

Pipeline per generated image:
  1. decode RGB -> vector:  v = 2 * rgb / 255 - 1          (linear encoding)
  2. bilinear-resize the vector map back to the GT frame (undoing the loader's
     preprocessing), then re-normalize each vector to unit length
  3. map the model's axes/signs to the GT convention with a FIXED signed permutation
     (``rgb_to_gt``), chosen once on separate calibration data (see ``calibrate`` below)
  4. per valid pixel: angle = degrees(arccos(clip(pred . gt, -1, 1)))
  5. per image: mean, median, % < 11.25 / 22.5 / 30 deg; the runner/aggregator then
     averages per-image values across images (same as Marigold's eval.py).

Invalid pixels: GT vectors with zero length (or non-finite) are excluded, exactly as in
Marigold ``compute_cosine_error(masked=True)``.

Prediction failures inside valid GT (decided before evaluation, reported in results):
  * a decoded vector shorter than ``degenerate_norm_eps`` (e.g. mid-gray 128,128,128)
    has no direction, and a valid GT pixel not covered by the prediction (only possible
    when the loader cropped) has no prediction;
  * both are kept and scored as ``degenerate_error_deg`` (default 90 deg, i.e. what
    torch.cosine_similarity gives for a zero vector in Marigold's evaluator);
  * their share is reported as ``degenerate_pixel_pct``.
Whole-image generation failures are handled by tasks/aggregators/surface_normal.py.

Calibrate the coordinate convention (run from the repo root, after ``generate`` on the
calibration config):

    uv run python tasks/evaluators/surface_normal.py calibrate --config qwen_normal_calib
"""

from __future__ import annotations

import io
import itertools
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

METRICS = {
    "mean_angular_error": {"unit": "degrees", "higher_is_better": False},
    "median_angular_error": {"unit": "degrees", "higher_is_better": False},
    "pct_within_11_25": {"unit": "percent", "higher_is_better": True},
    "pct_within_22_5": {"unit": "percent", "higher_is_better": True},
    "pct_within_30": {"unit": "percent", "higher_is_better": True},
    "degenerate_pixel_pct": {"unit": "percent", "higher_is_better": False},
}

THRESHOLDS = {"pct_within_11_25": 11.25, "pct_within_22_5": 22.5, "pct_within_30": 30.0}
DEFAULT_RGB_TO_GT = ["+r", "+g", "+b"]


# --------------------------------------------------------------------------- helpers

def parse_convention(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    """``["+r", "-g", "-b"]`` -> (channel order, signs). Entry i gives GT axis i."""
    if spec is None:
        spec = DEFAULT_RGB_TO_GT
    if isinstance(spec, str):
        spec = spec.replace(",", " ").split()
    if len(spec) != 3:
        raise ValueError(f"rgb_to_gt needs 3 entries, got {spec!r}")
    order, signs = [], []
    for item in spec:
        item = str(item).strip().lower()
        sign = -1.0 if item.startswith("-") else 1.0
        channel = item.lstrip("+-")
        if channel not in {"r", "g", "b"}:
            raise ValueError(f"Bad rgb_to_gt entry {item!r}; use e.g. '+r', '-g'")
        order.append("rgb".index(channel))
        signs.append(sign)
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"rgb_to_gt must use each of r, g, b once: {spec!r}")
    return np.array(order), np.array(signs, dtype=np.float32)


def format_convention(order: np.ndarray, signs: np.ndarray) -> list[str]:
    return [("+" if s > 0 else "-") + "rgb"[c] for c, s in zip(order, signs)]


def all_conventions() -> list[list[str]]:
    """The 48 signed axis permutations."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            out.append(format_convention(np.array(perm), np.array(signs)))
    return out


def load_gt(ground_truth: dict) -> np.ndarray:
    """H x W x 3 float32 normals from the .npy file (folder or tar)."""
    root = Path(ground_truth["normal_root"])
    rel = ground_truth["normal_rel"]
    if ground_truth.get("is_tar"):
        with tarfile.open(root) as archive:
            name = rel if rel in archive.getnames() else "./" + rel
            data = io.BytesIO(archive.extractfile(name).read())
        gt = np.load(data)
    else:
        gt = np.load(root / rel)
    gt = np.asarray(gt, dtype=np.float32)
    if gt.ndim == 3 and gt.shape[0] == 3 and gt.shape[-1] != 3:
        gt = np.transpose(gt, (1, 2, 0))
    if gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError(f"GT normals must be HxWx3, got {gt.shape} for {rel}")
    return gt


def _resize_channels(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of an H x W x C float array to (width, height)."""
    if (array.shape[1], array.shape[0]) == tuple(size):
        return array
    channels = [np.asarray(Image.fromarray(array[..., c], mode="F").resize(size, Image.Resampling.BILINEAR))
                for c in range(array.shape[-1])]
    return np.stack(channels, axis=-1)


def decode_to_gt_frame(image: Image.Image, sample_metadata: dict, gt_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode RGB -> raw vectors placed in the GT pixel frame.

    Returns (vectors HxWx3 in RGB-channel order, covered mask HxW). Vectors are NOT yet
    normalized; pixels outside the region the model saw have covered=False.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    vectors = 2.0 * rgb / 255.0 - 1.0
    gt_w, gt_h = gt_size
    covered = np.zeros((gt_h, gt_w), dtype=bool)
    out = np.zeros((gt_h, gt_w, 3), dtype=np.float32)

    transform = sample_metadata.get("transform")
    if transform is None:
        x0, y0, x1, y1 = 0.0, 0.0, float(gt_w), float(gt_h)
    else:
        from visionbench.transforms import CoordinateTransform
        t = CoordinateTransform.from_dict(transform)
        if tuple(t.original_size) != (gt_w, gt_h):
            raise ValueError(f"GT size {gt_size} != loader original size {t.original_size}")
        out_w, out_h = t.output_size
        x0, y0 = t.inverse((0.0, 0.0))
        x1, y1 = t.inverse((float(out_w), float(out_h)))
    left, top = int(round(x0)), int(round(y0))
    right, bottom = min(gt_w, int(round(x1))), min(gt_h, int(round(y1)))
    left, top = max(0, left), max(0, top)
    if right <= left or bottom <= top:
        return out, covered
    # The generated image (whatever size the model returned) spans exactly this region.
    out[top:bottom, left:right] = _resize_channels(vectors, (right - left, bottom - top))
    covered[top:bottom, left:right] = True
    return out, covered


def angular_errors(pred_rgb_vectors: np.ndarray, covered: np.ndarray, gt: np.ndarray,
                   rgb_to_gt: Any = None, degenerate_norm_eps: float = 0.05,
                   degenerate_error_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-valid-pixel angular error in degrees, plus a degenerate flag per valid pixel."""
    order, signs = parse_convention(rgb_to_gt)
    pred = pred_rgb_vectors[..., order] * signs
    gt_norm = np.linalg.norm(gt, axis=-1)
    valid = np.isfinite(gt).all(axis=-1) & (gt_norm > 1e-6)
    pred_v = pred[valid].astype(np.float64)
    gt_v = gt[valid].astype(np.float64) / gt_norm[valid, None]
    pred_norm = np.linalg.norm(pred_v, axis=-1)
    degenerate = (~covered[valid]) | (pred_norm < degenerate_norm_eps) | ~np.isfinite(pred_norm)
    safe = np.where(degenerate, 1.0, pred_norm)
    pred_v = pred_v / safe[:, None]
    cosine = np.clip(np.sum(pred_v * gt_v, axis=-1), -1.0, 1.0)
    errors = np.degrees(np.arccos(cosine))
    errors[degenerate] = degenerate_error_deg
    return errors, degenerate


def summarize(errors: np.ndarray, degenerate: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        raise ValueError("No valid GT pixels in this sample")
    result = {
        "mean_angular_error": float(np.mean(errors)),
        "median_angular_error": float(np.median(errors)),
        "degenerate_pixel_pct": float(100.0 * np.mean(degenerate)),
    }
    for name, threshold in THRESHOLDS.items():
        result[name] = float(100.0 * np.mean(errors < threshold))
    return result


# --------------------------------------------------------------------------- plugin API

def prepare_output(image, sample, config):
    return image.convert("RGB")


def score(generated_image, sample, config) -> dict[str, float]:
    gt = load_gt(sample.ground_truth)
    gt_size = (gt.shape[1], gt.shape[0])
    vectors, covered = decode_to_gt_frame(generated_image, sample.metadata, gt_size)
    errors, degenerate = angular_errors(
        vectors, covered, gt,
        rgb_to_gt=config.get("rgb_to_gt", DEFAULT_RGB_TO_GT),
        degenerate_norm_eps=float(config.get("degenerate_norm_eps", 0.05)),
        degenerate_error_deg=float(config.get("degenerate_error_deg", 90.0)),
    )
    return summarize(errors, degenerate)


# --------------------------------------------------------------------------- calibration

def calibrate(config_name: str, out_path: str | None = None) -> dict:
    """Score every signed axis permutation on a calibration run; pick the best one.

    Selection criterion: lowest mean (over images) of per-image mean angular error.
    Uses only the generated images of the given (calibration) config, never test data.
    """
    import json

    from visionbench.config import load_config
    from visionbench.runner import _dataset, _image_path, _read, _trial_path

    config = load_config(config_name)
    params = dict(config.data["evaluator"].get("params", {}))
    eps = float(params.get("degenerate_norm_eps", 0.05))
    deg_err = float(params.get("degenerate_error_deg", 90.0))
    dataset = _dataset(config)
    candidates = all_conventions()
    per_image: dict[str, list[float]] = {",".join(c): [] for c in candidates}
    used = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        gt = None
        for trial in range(config.data["trials"]):
            checkpoint = _trial_path(config.run_dir, index, trial)
            if not checkpoint.exists() or _read(checkpoint).get("status") != "success":
                continue
            if gt is None:
                gt = load_gt(sample.ground_truth)
            with Image.open(_image_path(config.run_dir, index, trial)) as image:
                vectors, covered = decode_to_gt_frame(image.convert("RGB"), sample.metadata, (gt.shape[1], gt.shape[0]))
            for candidate in candidates:
                errors, _ = angular_errors(vectors, covered, gt, candidate, eps, deg_err)
                if errors.size:
                    per_image[",".join(candidate)].append(float(np.mean(errors)))
            used += 1
    if used == 0:
        raise RuntimeError(f"No successful generations under {config.run_dir}; run `visionbench generate` first")
    ranking = sorted(((float(np.mean(v)), k) for k, v in per_image.items() if v))
    best = ranking[0][1].split(",")
    report = {"calibration_config": str(config.source), "images_used": used,
              "best_rgb_to_gt": best, "ranking": [{"rgb_to_gt": k.split(","), "mean_angular_error": m}
                                                 for m, k in ranking]}
    target = Path(out_path) if out_path else config.run_dir / "convention_calibration.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibrated on {used} generated images -> {target}")
    for m, k in ranking[:5]:
        print(f"  {k:<14} mean angular error {m:6.2f} deg")
    print(f"\nPut this in every surface-normal config's evaluator params:\n  rgb_to_gt: {best}")
    return report


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="choose the RGB->GT axis convention on a calibration run")
    cal.add_argument("--config", required=True, help="config name in configs/ or a YAML path")
    cal.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args()
    calibrate(args.config, args.out)
