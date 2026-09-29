"""Preflight, durable generation, evaluation, and run orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

from .config import RunConfig
from .models.base import FatalModelError, SampleModelError, TransientModelError
from .plugins import load_aggregator, load_dataset, load_evaluator
from .types import Sample


@dataclass(frozen=True)
class SampleReference:
    """Reload sample data on demand so large global aggregations stay memory bounded."""

    dataset: Any
    index: int
    sample_id: str
    fingerprint: str

    def load(self) -> Sample:
        sample = _sample(self.dataset, self.index)
        if sample.sample_id != self.sample_id or _fingerprint(sample) != self.fingerprint:
            raise ValueError(f"Dataset changed at index {self.index}")
        return sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self.load(), name)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"visionbench.{run_dir}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        for handler in (file_handler, stream_handler):
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
    return logger


def _fingerprint(sample: Sample) -> str:
    digest = hashlib.sha256()
    digest.update(sample.image.mode.encode())
    digest.update(str(sample.image.size).encode())
    digest.update(sample.image.tobytes())
    digest.update(sample.prompt.encode("utf-8"))
    return digest.hexdigest()


def _sample(dataset: Any, index: int) -> Sample:
    try:
        item = dataset[index]
    except Exception as exc:
        raise RuntimeError(f"Dataset failed to load index {index}: {exc}") from exc
    if not isinstance(item, Sample):
        raise TypeError(f"Dataset index {index} must return visionbench.types.Sample")
    if not isinstance(item.sample_id, str) or not item.sample_id:
        raise ValueError(f"Dataset index {index} has no stable sample_id")
    if not isinstance(item.prompt, str) or not item.prompt.strip():
        raise ValueError(f"Dataset index {index} has an empty prompt")
    if not isinstance(item.image, Image.Image) or item.image.mode not in {"RGB", "RGBA"}:
        raise ValueError(f"Dataset index {index} must return an RGB/RGBA Pillow image")
    return item


def _dataset(config: RunConfig) -> Any:
    entry = config.data["dataset"]
    params = dict(entry.get("params", {}))
    params["raw_dir"] = str(config.scoped_path("raw", entry.get("raw_path", ".")))
    params["assets_dir"] = str(config.scoped_path("assets", entry.get("assets_path", ".")))
    return load_dataset(config.scoped_path("dataset", entry["path"]), params)


def preflight(config: RunConfig, *, save_inputs: bool = True) -> tuple[Any, list[dict[str, Any]]]:
    """Validate all samples before model loading or paid calls."""
    run_dir = config.run_dir
    log = _logger(run_dir)
    dataset = _dataset(config)
    entries: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index in range(len(dataset)):
        sample = _sample(dataset, index)
        if sample.sample_id in ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        ids.add(sample.sample_id)
        entries.append({"index": index, "sample_id": sample.sample_id, "prompt": sample.prompt,
                        "image_mode": sample.image.mode, "image_size": list(sample.image.size),
                        "fingerprint": _fingerprint(sample)})
    settings = config.generation_settings()
    manifest = {"generation_settings": settings, "samples": entries}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = _read(manifest_path)
        if old != manifest:
            raise ValueError("Generation version already exists with changed model, prompts, images, or settings; choose a new generation_version")
    else:
        _json(manifest_path, manifest)
        _json(run_dir / "config.resolved.json", config.snapshot())
    if save_inputs:
        for entry in entries:
            target = run_dir / "inputs" / f"{entry['index']:08d}.png"
            if not target.exists():
                sample = _sample(dataset, entry["index"])
                if _fingerprint(sample) != entry["fingerprint"]:
                    raise ValueError(f"Dataset changed during preflight at index {entry['index']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                sample.image.save(target, format="PNG")
    log.info("Preflight passed: %d samples", len(entries))
    return dataset, entries


def _backend(config: RunConfig) -> Any:
    model = config.data["model"]
    kind = model["backend"].lower()
    options = model.get("options", {})
    if kind == "dummy":
        from .models.dummy import DummyBackend
        return DummyBackend(model["id"])
    if kind == "qwen":
        from .models.qwen import QwenBackend, QwenConfig
        profile = model.get("profile", "custom")
        cache = config.resolve(model.get("cache_dir", "model-cache"))
        if profile == "vram_16gb":
            settings = QwenConfig.vram_16gb(model_id=model["id"], revision=model.get("revision", ""), cache_dir=cache, **options)
        else:
            settings = QwenConfig(model_id=model["id"], revision=model.get("revision", ""), cache_dir=cache, **options)
        return QwenBackend(settings)
    from .models.remote import GeminiImageBackend, OpenAIGPTImageBackend, VolcanoArkSeedreamBackend
    key = model.get("api_key", "")
    if not key:
        raise ValueError(f"model.api_key is required for {kind}")
    if kind == "openai":
        return OpenAIGPTImageBackend(model=model["id"], api_key=key, **options)
    if kind == "gemini":
        return GeminiImageBackend(model=model["id"], api_key=key, **options)
    if kind == "ark":
        return VolcanoArkSeedreamBackend(model=model["id"], api_key=key, **options)
    raise ValueError(f"Unknown model.backend: {kind}")


def _seed(base_seed: int, sample_id: str, trial: int) -> int:
    raw = f"{base_seed}:{sample_id}:{trial}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _trial_path(run_dir: Path, index: int, trial: int) -> Path:
    return run_dir / "checkpoints" / f"{index:08d}_trial_{trial:03d}.json"


def _image_path(run_dir: Path, index: int, trial: int) -> Path:
    return run_dir / "images" / f"{index:08d}_trial_{trial:03d}.png"


def _generate_one(config: RunConfig, backend: Any, entry: dict[str, Any], trial: int,
                  input_image: Image.Image) -> dict[str, Any]:
    retry = config.data["retries"]
    max_attempts = retry["max_attempts"]
    seed = _seed(config.data["base_seed"], entry["sample_id"], trial)
    started = time.monotonic()
    last_error = "transient generation error"
    for attempt in range(1, max_attempts + 1):
        try:
            result = backend.generate(input_image.copy(), entry["prompt"], config.data["system_prompt"], seed)
            if not isinstance(result.image, Image.Image):
                raise SampleModelError("Backend returned no Pillow image")
            output = _image_path(config.run_dir, entry["index"], trial)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.stem + ".tmp.png")
            result.image.save(temporary, format="PNG")
            temporary.replace(output)
            return {"status": "success", "sample_id": entry["sample_id"], "index": entry["index"],
                    "trial": trial, "seed": seed, "attempts": attempt,
                    "latency_s": round(time.monotonic() - started, 4), "image": str(output.relative_to(config.run_dir)),
                    "width": result.image.width, "height": result.image.height, "metadata": result.metadata}
        except TransientModelError as exc:
            last_error = str(exc)
            if attempt == max_attempts:
                break
            delay = min(float(retry["max_delay_s"]), float(retry["base_delay_s"]) * 2 ** (attempt - 1))
            time.sleep(delay * random.uniform(0.8, 1.2))
        except SampleModelError as exc:
            return {"status": "failed", "sample_id": entry["sample_id"], "index": entry["index"],
                    "trial": trial, "seed": seed, "attempts": attempt, "error": str(exc)}
    return {"status": "failed", "sample_id": entry["sample_id"], "index": entry["index"],
            "trial": trial, "seed": seed, "attempts": max_attempts, "error": last_error}


def generate(config: RunConfig) -> dict[str, Any]:
    log = _logger(config.run_dir)
    dataset, entries = preflight(config, save_inputs=bool(config.data["save_inputs"]))
    pending: list[tuple[dict[str, Any], int]] = []
    for entry in entries:
        for trial in range(config.data["trials"]):
            checkpoint = _trial_path(config.run_dir, entry["index"], trial)
            if checkpoint.exists():
                previous = _read(checkpoint)
                if previous.get("status") == "success" and (config.run_dir / previous["image"]).exists():
                    continue
            pending.append((entry, trial))
    if pending:
        backend = _backend(config)
        workers = 1 if config.data["model"]["backend"].lower() in {"qwen", "dummy"} else config.data["concurrency"]
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                tasks: dict[Any, tuple[dict[str, Any], int]] = {}
                iterator = iter(pending)

                def submit_next() -> bool:
                    try:
                        entry, trial = next(iterator)
                    except StopIteration:
                        return False
                    input_path = config.run_dir / "inputs" / f"{entry['index']:08d}.png"
                    if input_path.exists():
                        with Image.open(input_path) as stored:
                            input_image = stored.copy()
                    else:
                        sample = _sample(dataset, entry["index"])
                        if _fingerprint(sample) != entry["fingerprint"]:
                            raise ValueError(f"Dataset changed at index {entry['index']}")
                        input_image = sample.image.copy()
                    future = pool.submit(_generate_one, config, backend, entry, trial, input_image)
                    tasks[future] = (entry, trial)
                    return True

                for _ in range(workers * 2):
                    if not submit_next():
                        break
                while tasks:
                    done, _ = wait(tasks, return_when=FIRST_COMPLETED)
                    for future in done:
                        entry, trial = tasks.pop(future)
                        record = future.result()  # fatal/configuration errors stop the run
                        _json(_trial_path(config.run_dir, entry["index"], trial), record)
                        if record["status"] == "failed":
                            log.warning("Generation failed sample=%s trial=%s: %s", entry["sample_id"], trial, record["error"])
                        else:
                            log.info("Generated sample=%s trial=%s", entry["sample_id"], trial)
                        submit_next()
        finally:
            backend.close()
    records = [_read(_trial_path(config.run_dir, entry["index"], trial))
               for entry in entries for trial in range(config.data["trials"])
               if _trial_path(config.run_dir, entry["index"], trial).exists()]
    summary = {"samples": len(entries), "trials_requested": len(entries) * config.data["trials"],
               "trials_succeeded": sum(row["status"] == "success" for row in records),
               "trials_failed": sum(row["status"] == "failed" for row in records)}
    _json(config.run_dir / "generation_summary.json", summary)
    return summary


def _next_evaluation_dir(run_dir: Path) -> Path:
    root = run_dir / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        target = root / f"{number:04d}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            number += 1


def evaluate(config: RunConfig) -> Path:
    log = _logger(config.run_dir)
    manifest_path = config.run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No generation manifest; run generate first")
    dataset, entries = preflight(config, save_inputs=False)
    evaluator_entry = config.data["evaluator"]
    evaluator = load_evaluator(config.scoped_path("evaluator", evaluator_entry["path"]), evaluator_entry.get("params", {}))
    aggregator_entry = config.data.get("aggregator")
    aggregator = (load_aggregator(config.scoped_path("aggregator", aggregator_entry["path"]), aggregator_entry.get("params", {}))
                  if aggregator_entry and aggregator_entry.get("path") else None)
    evaluation_dir = _next_evaluation_dir(config.run_dir)
    full_records: list[dict[str, Any]] = []
    public_records: list[dict[str, Any]] = []
    scoring_failures = 0
    invalid_metrics: dict[str, int] = {name: 0 for name in evaluator.metrics}
    for entry in entries:
        sample = _sample(dataset, entry["index"])
        if _fingerprint(sample) != entry["fingerprint"]:
            raise ValueError(f"Dataset changed at index {entry['index']}")
        trial_rows: list[dict[str, Any]] = []
        output_paths: list[Path] = []
        for trial in range(config.data["trials"]):
            checkpoint = _trial_path(config.run_dir, entry["index"], trial)
            if not checkpoint.exists():
                trial_rows.append({"trial": trial, "status": "generation_missing"})
                continue
            generation = _read(checkpoint)
            if generation["status"] != "success":
                trial_rows.append({"trial": trial, "status": "generation_failed", "error": generation.get("error")})
                continue
            output_path = config.run_dir / generation["image"]
            output_paths.append(output_path)
            try:
                with Image.open(output_path) as saved:
                    prepared = evaluator.prepare_output(saved.copy(), sample)
                scores = evaluator.score(prepared, sample)
                if not isinstance(scores, dict):
                    raise TypeError("score() must return a dictionary")
                valid: dict[str, float] = {}
                for name in evaluator.metrics:
                    value = scores.get(name)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        valid[name] = float(value)
                    else:
                        invalid_metrics[name] += 1
                        log.warning("Invalid or missing metric %s sample=%s trial=%s", name, sample.sample_id, trial)
                extra = set(scores) - set(evaluator.metrics)
                if extra:
                    log.warning("Undeclared metrics ignored sample=%s: %s", sample.sample_id, sorted(extra))
                trial_rows.append({"trial": trial, "status": "scored", "metrics": valid,
                                   "image": generation["image"], "generation": generation})
            except Exception as exc:
                scoring_failures += 1
                log.warning("Scoring failed sample=%s trial=%s: %s", sample.sample_id, trial, exc, exc_info=True)
                trial_rows.append({"trial": trial, "status": "scoring_failed", "error": str(exc),
                                   "image": generation["image"]})
        trial_means = {name: mean([row["metrics"][name] for row in trial_rows
                                   if row["status"] == "scored" and name in row["metrics"]])
                       for name in evaluator.metrics
                       if any(row["status"] == "scored" and name in row["metrics"] for row in trial_rows)}
        full_records.append({"sample_id": sample.sample_id,
                             "sample": SampleReference(dataset, entry["index"], sample.sample_id, entry["fingerprint"]),
                             "metrics": trial_means,
                             "trials": trial_rows, "output_paths": output_paths,
                             "generation_failures": sum(row["status"].startswith("generation_") for row in trial_rows)})
        public_records.append({"sample_id": sample.sample_id, "index": entry["index"],
                               "metrics": trial_means, "trials": trial_rows})
    if aggregator:
        overall = aggregator.aggregate(full_records)
        if not isinstance(overall, dict):
            raise TypeError("aggregate() must return a dictionary")
    else:
        overall = {name: mean([row["metrics"][name] for row in full_records if name in row["metrics"]])
                   for name in evaluator.metrics
                   if any(name in row["metrics"] for row in full_records)}
    for name, value in overall.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Aggregator returned invalid metric {name}={value}")
    metric_metadata = {name: {"unit": definition.unit, "higher_is_better": definition.higher_is_better}
                       for name, definition in evaluator.metrics.items()}
    summary = {"task_name": config.data["task_name"], "generation_version": config.data["generation_version"],
               "evaluation_version": int(evaluation_dir.name), "created_at": datetime.now(timezone.utc).isoformat(),
               "model": config.snapshot()["model"], "metrics": overall, "metric_definitions": metric_metadata,
               "coverage": {"samples_total": len(entries),
                            "generation_trials_success": sum(row["status"] == "scored" or row["status"] == "scoring_failed"
                                                             for sample_row in public_records for row in sample_row["trials"]),
                            "scoring_failures": scoring_failures, "invalid_metric_trials": invalid_metrics,
                            "metric_samples": {name: sum(name in row["metrics"] for row in full_records)
                                               for name in evaluator.metrics}},
               "samples": public_records}
    path = evaluation_dir / "results.json"
    _json(path, summary)
    log.info("Evaluation %s complete: %s", evaluation_dir.name, path)
    return path


def archive(config: RunConfig) -> None:
    s3 = config.data.get("s3", {})
    if not s3.get("enabled"):
        return
    log = _logger(config.run_dir)
    try:
        from .storage import S3Archive
        store = S3Archive(bucket=s3["bucket"], prefix=s3.get("prefix", ""),
                          region_name=s3.get("region_name"), endpoint_url=s3.get("endpoint_url"))
        result = store.archive_run(config.run_dir)
        if result.uploaded:
            log.info("Archived run to s3://%s/%s", result.bucket, result.key)
        else:
            log.warning("S3 archive failed; local results are intact: %s", result.error)
    except Exception as exc:
        log.warning("S3 archive failed; local results are intact: %s", exc)


def run(config: RunConfig) -> Path:
    generate(config)
    result = evaluate(config)
    archive(config)
    return result


def infer(config: RunConfig, image_path: str | Path, prompt: str, system_prompt: str | None = None) -> Path:
    with Image.open(image_path) as source:
        image = source.copy()
    if image.mode not in {"RGB", "RGBA"}:
        raise ValueError("Debug input must be RGB or RGBA; convert it explicitly first")
    backend = _backend(config)
    try:
        output = backend.generate(image, prompt, system_prompt if system_prompt is not None else config.data["system_prompt"],
                                  config.data["base_seed"])
    finally:
        backend.close()
    target = config.root / "debug" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    output.image.save(target, format="PNG")
    _json(target.with_suffix(".json"), {"prompt": prompt, "model": config.snapshot()["model"],
                                       "metadata": output.metadata, "image": str(target)})
    return target
