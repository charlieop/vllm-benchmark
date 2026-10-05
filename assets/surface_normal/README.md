# Surface normal estimation (NYUv2 / iBims-1 / DIODE-indoor)

Same test sets, GT normals and valid masks as the Marigold normals evaluation release,
which is the protocol behind the DSINE / Marigold / StableNormal numbers in RINO Table 3.

| Config | Data | Images | Role |
|---|---|---|---|
| `qwen_normal_calib` | DIODE val **outdoor** (every 9th) | ~50 | pick the RGB→GT axis convention once |
| `qwen_normal_nyuv2` | NYUv2 test | 654 | test |
| `qwen_normal_ibims` | iBims-1 | 100 | test |
| `qwen_normal_diode` | DIODE val **indoor** | 325 | test |

Split files here (`nyuv2_test.txt`, `ibims_test.txt`, `diode_test.txt`) are copied unchanged from
[prs-eth/Marigold `data_split/`](https://github.com/prs-eth/Marigold/tree/main/data_split) (Apache-2.0).

## 1. Download (≈ a few GB)

```bash
mkdir -p raw/surface_normal && cd raw/surface_normal
wget https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals/evaluation_dataset.zip
unzip evaluation_dataset.zip && rm evaluation_dataset.zip
cd ../..
```

The configs expect these folders (move them if the zip adds an extra top-level folder):

```text
raw/surface_normal/nyuv2/test/000000_img.png   000000_normal.npy ...
raw/surface_normal/ibims/ibims/corridor_01_img.png   corridor_01_normal.npy ...
raw/surface_normal/diode/val/indoors/scene_*/scan_*/*.png   *_normal.npy
raw/surface_normal/diode/val/outdoor/...
```

If a folder ships as a `.tar` instead (e.g. `nyuv2/test.tar`), point `dataset.raw_path` at the
tar file; the loader reads tars directly.

## 2. Calibrate the coordinate convention (once)

Generated normal maps use an unknown axis/sign convention. We fix it on calibration data
that is disjoint from every test set, then freeze it:

```bash
uv run visionbench generate --config qwen_normal_calib
uv run python tasks/evaluators/surface_normal.py calibrate --config qwen_normal_calib
# -> prints e.g.  rgb_to_gt: ['+r', '-g', '-b']
```

Copy that `rgb_to_gt` into the evaluator params of `qwen_normal_nyuv2/ibims/diode.yaml`
(and of any other model's configs — calibrate each model separately, with its own calib run).

## 3. Run the test sets

```bash
uv run visionbench run --config qwen_normal_nyuv2
uv run visionbench run --config qwen_normal_ibims
uv run visionbench run --config qwen_normal_diode
```

Changing only `rgb_to_gt` later does not require regenerating: `visionbench evaluate` reuses images.

## Protocol details

* Decode: `v = 2·RGB/255 − 1`, bilinear resize to GT resolution (undoing the loader's
  preprocessing), re-normalize, apply `rgb_to_gt`.
* Error: `degrees(arccos(clip(pred·gt, −1, 1)))` per valid pixel.
* Valid pixels: GT vector non-zero and finite (Marigold rule). Nothing else is masked.
* Degenerate predictions inside valid GT (|v| < 0.05, e.g. mid-gray, or not covered by the
  output) are **kept** and scored as 90° (what Marigold's cosine-similarity gives a zero
  vector); their share is reported as `degenerate_pixel_pct`.
* Failed generations: aggregator `failed_sample_policy: penalize` scores the whole image as
  90° (mean = median = 90, thresholds = 0 %); `samples_failed` is reported.
* Metrics are computed per image, then averaged over images (Marigold `eval.py`):
  `mean_angular_error` (main), `median_angular_error`, `pct_within_11_25 / 22_5 / 30`.
