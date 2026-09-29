# VisionBench

Stage 1 image-to-image benchmarking for vision tasks. A run first saves every generated image, then evaluates successful outputs. Teammates only write a dataset loader, evaluator, and optional aggregator. [SPEC.md](SPEC.md) records the agreed behavior and acceptance criteria.

## Quick start

On Linux with an NVIDIA GPU, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --frozen
cp .env.example .env
uv run visionbench run --config config_dummy
```

The dummy run needs no GPU or API key. It uses four small images and scores the fraction of exact white pixels. The default real-model configuration is [`configs/qwen.yaml`](configs/qwen.yaml), which uses a 16 GB-oriented NF4/offload profile and a pinned Qwen-Image 2.1 revision. Run it after confirming a CUDA driver and sufficient model-cache storage:

```bash
uv run visionbench run --config qwen
```

The first Qwen run downloads a large checkpoint. Set `model.cache_dir` to persistent storage. Image dimensions affect peak memory: the 16 GB profile is a starting configuration, not a guarantee for arbitrary input sizes. The runner never resizes input images or silently changes quantization. If the GPU runs out of memory, it stops; preprocess in the dataset loader and use a new `generation_version` when changing the generated input.

To test one image without running the dataset:

```bash
uv run visionbench infer --config config_dummy --image raw/white_background/sample_01.png --prompt 'Change the background to white #FFFFFF'
```

Available commands are `run`, `generate`, `evaluate`, `infer`, and `archive`. `--config` accepts a name from `configs/` (`qwen`, `ark`, `config_dummy`, and so on) or an explicit YAML path. `run` executes generation, then evaluation, then optional S3 archival. `generate` resumes completed outputs by default. `evaluate` starts a new numbered evaluation every time. `infer` writes a debug PNG and JSON outside benchmark runs. Open [`notebooks/01_single_sample_debug.ipynb`](notebooks/01_single_sample_debug.ipynb) or [`notebooks/02_full_pipeline.ipynb`](notebooks/02_full_pipeline.ipynb) from the repository root for interactive work.

## Task code

Each task uses three separate files under [`tasks/`](tasks/): [`tasks/dataloaders/`](tasks/dataloaders/), [`tasks/evaluators/`](tasks/evaluators/), and [`tasks/aggregators/`](tasks/aggregators/). Put unmodified source dataset files under [`raw/`](raw/) and optional task support files under [`assets/`](assets/). The toy input images are in [`raw/white_background/`](raw/white_background/). Copy the examples in [`tasks/dataloaders/white_background.py`](tasks/dataloaders/white_background.py), [`tasks/evaluators/white-background.py`](tasks/evaluators/white-background.py), and [`tasks/aggregators/white_background.py`](tasks/aggregators/white_background.py).

YAML paths are relative to their dedicated folder. All supplied YAML files, including [`configs/config_dummy.yaml`](configs/config_dummy.yaml), live in `configs/`. For example:

```yaml
dataset:
  path: white_background.py   # tasks/dataloaders/white_background.py
  raw_path: white_background  # raw/white_background/
  assets_path: .              # assets/ (optional; default)
  params:
    max_size: [1024, 1024]
    shrink_mode: scale
    size_multiple: 16
evaluator:
  path: white-background.py   # tasks/evaluators/white-background.py
aggregator:
  path: white_background.py  # tasks/aggregators/white_background.py
```

The runner passes resolved `raw_dir` and `assets_dir` strings to `load_dataset(config)`, alongside `dataset.params`. Absolute paths, `..` escapes, and symlinks that leave a dedicated folder are rejected for these YAML paths. Keep source inputs in `raw/`; the runner's `outputs/.../inputs/` folder stores the processed images actually sent to a model.

```python
# tasks/dataloaders/my_task.py
from visionbench.types import Sample

class MyDataset:
    def __len__(self): ...
    def __getitem__(self, index) -> Sample: ...  # stable ID, RGB/RGBA Pillow image, prompt, arbitrary GT

def load_dataset(config: dict) -> MyDataset: ...
```

```python
# tasks/evaluators/my_task.py
METRICS = {"score_name": {"unit": "fraction", "higher_is_better": True}}

def prepare_output(image, sample, config):  # optional
    return image

def score(generated_image, sample, config) -> dict[str, float]: ...
```

```python
# tasks/aggregators/my_task.py; omit this file/path to use a mean across samples
def aggregate(records: list[dict], config: dict) -> dict[str, float]: ...
```

`records` contains a lazy `sample` reference (`.load()` retrieves the full `Sample`; fields such as `.ground_truth` can be accessed directly), its trial-mean `metrics`, trial records, output paths, and failure count. Keeping samples lazy prevents thousands of large images from accumulating in RAM. Every trial is scored independently; its metrics are averaged within the sample before aggregation. Invalid or missing metric values are warned about and excluded with per-metric coverage reported. Scoring exceptions affect only the relevant trial. Dataset loading errors and duplicate IDs stop the run during full preflight, before model loading or paid calls.

The sample loader calls `visionbench.transforms.preprocess_image(image, max_size=(1024, 1024), shrink_mode="scale", size_multiple=16)` and stores the serializable transform and parameters in sample metadata. The helper first shrinks if needed, then center-crops down to the multiple. It supports forward/inverse points and boxes, visibility checks, and mask/depth/normal raster transforms. The runner itself never changes image geometry. The Qwen backend currently requires dimensions divisible by **32**, so `configs/qwen.yaml` sets `size_multiple: 32`; set the same preprocessing parameters in every model config when comparing them on a new dataset.

## YAML and credentials

The YAML files are [`configs/config_dummy.yaml`](configs/config_dummy.yaml), [`configs/qwen.yaml`](configs/qwen.yaml), [`configs/openai.yaml`](configs/openai.yaml), [`configs/gemini.yaml`](configs/gemini.yaml), and [`configs/ark.yaml`](configs/ark.yaml). They select explicit model IDs, a multiline system prompt, task files/parameters, trial count, remote concurrency (default 3), retry policy, output root, and S3 archive settings. Local Qwen always runs one generation at a time. Model-specific controls belong under `model.options`; invalid Qwen combinations fail before inference. Model-specific prompt tuning should use a separately labeled run; all main comparisons should use the same task and sample prompts.

Copy `.env.example` to `.env` for API keys. YAML uses `${OPENAI_API_KEY}`, `${GEMINI_API_KEY}`, or `${ARK_API_KEY}`; literal API keys are rejected. `.env` is ignored by Git, and saved resolved configurations redact secrets. OpenAI uses the Images edit API, Google uses the Gemini Developer API, and Seedream uses ByteDance Volcano Ark. The adapters accept provider-native options, and API responses may return a different output size. Task `prepare_output` handles alignment before scoring. Provider API availability, access, and billing must be checked in each account.

The original model list said Seedance; `configs/ark.yaml` selects Seedream 5.0 Pro for this image task. Confirm that substitution before using its scores in a comparison. The four tiny example images are for the dummy smoke test; for real providers, use a dataset whose sizes satisfy each provider's current input and output limits.

For S3, set `s3.enabled: true`, `s3.bucket`, and optional `s3.prefix` in YAML. Use an EC2 IAM role or standard `AWS_*` environment variables. The archive command uploads a tarball of the local run. S3 is only an archive; local checkpoints drive resumption. Archive failure is logged without deleting local results.

## Run layout and resume

```text
outputs/{task_name}/{generation_version}/
  manifest.json
  config.resolved.json
  run.log
  inputs/                         # processed images received from the loader
  images/                         # raw generated images, one per trial
  checkpoints/                    # one atomic record per generation trial
  generation_summary.json
  evaluations/0001/results.json
  evaluations/0002/results.json
```

The manifest fingerprints every processed input image and prompt. A rerun with unchanged generation settings skips completed outputs and retries failed ones. Changing a model, generation setting, prompt, or input image under the same `generation_version` raises an error; choose a new version. Changing evaluator or aggregator code can reuse images and creates the next evaluation number. Evaluation restarts from the beginning when rerun. Each `results.json` contains scores, full per-sample/per-trial records, failures, timing, model metadata, and coverage.

## Docker on AWS or another Linux GPU host

Install the NVIDIA driver and NVIDIA Container Toolkit on the host, then mount this repository and a persistent model cache:

```bash
docker build -t visionbench .
docker run --rm --gpus all --env-file .env \
  -v "$PWD/outputs:/app/outputs" -v "$PWD/model-cache:/app/model-cache" \
  -v "$PWD/raw:/app/raw:ro" -v "$PWD/assets:/app/assets:ro" \
  visionbench run --config qwen
```

The image installs all model and API dependencies. It includes only the four toy raw images; mount `raw/` and `assets/` to use larger datasets without baking them into the image. For API-only calls, the same image can run without `--gpus all`. On EC2, ensure the instance has enough EBS space for model weights, datasets, and saved inputs/outputs; an instance profile is the easiest way to grant S3 archive permissions. Model download and real GPU/API generation were not available in the development environment, so use the dummy run and notebooks first, then manually smoke-test one image per real provider and Qwen profile.

## Tests

```bash
uv run python -m pytest -q
```

The test suite covers plugin contracts, transforms, mocked provider requests, Qwen configuration and mocked generation, archive handling, and run lifecycle. Real provider and CUDA calls are deliberately excluded from automated tests.
