import json
import shutil
import threading
from pathlib import Path

import pytest
import yaml
from PIL import Image

from visionbench.config import RunConfig, load_config
from visionbench.models.base import FatalModelError, GenerationOutput, SampleModelError, TransientModelError
from visionbench.runner import evaluate, generate, run


PROJECT = Path(__file__).resolve().parents[1]


def make_config(tmp_path, *, trials=1, aggregator=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for folder in ("dataloaders", "evaluators", "aggregators"):
        shutil.copytree(PROJECT / "tasks" / folder, tmp_path / "tasks" / folder, dirs_exist_ok=True)
    shutil.copytree(PROJECT / "raw" / "white_background", tmp_path / "raw" / "white_background", dirs_exist_ok=True)
    (tmp_path / "assets").mkdir(exist_ok=True)
    config = {
        "task_name": "white_background",
        "generation_version": "test-v1",
        "output_root": str(tmp_path / "outputs"),
        "system_prompt": "Return one image.",
        "dataset": {"path": "white_background.py", "raw_path": "white_background", "params": {}},
        "evaluator": {"path": "white-background.py", "params": {}},
        "model": {"backend": "dummy", "id": "dummy-v1"},
        "trials": trials,
        "base_seed": 42,
        "retries": {"max_attempts": 2, "base_delay_s": 0, "max_delay_s": 0},
    }
    if aggregator:
        config["aggregator"] = {"path": "white_background.py", "params": {}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


def test_run_resume_and_numbered_reevaluation(tmp_path):
    config = make_config(tmp_path)
    first = run(config)
    result = json.loads(first.read_text())
    assert result["evaluation_version"] == 1
    assert result["coverage"]["samples_total"] == 4
    assert result["coverage"]["metric_samples"]["white_fraction"] == 4
    images = sorted((config.run_dir / "images").glob("*.png"))
    assert len(images) == 4
    mtime = [path.stat().st_mtime_ns for path in images]
    summary = generate(config)
    assert summary["trials_succeeded"] == 4
    assert [path.stat().st_mtime_ns for path in images] == mtime
    second = evaluate(config)
    assert second.parent.name == "0002"
    assert json.loads(second.read_text())["metrics"] == result["metrics"]


def test_changed_generation_setting_is_rejected(tmp_path):
    config = make_config(tmp_path)
    generate(config)
    raw = yaml.safe_load(config.source.read_text())
    raw["system_prompt"] = "A different prompt"
    config.source.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Generation version already exists"):
        generate(load_config(config.source))


def test_trial_mean_before_default_global_mean(tmp_path, monkeypatch):
    config = make_config(tmp_path, trials=2, aggregator=False)

    class AlternatingBackend:
        calls = 0

        def generate(self, image, prompt, system_prompt, seed):
            self.calls += 1
            color = "white" if self.calls % 2 else "black"
            return GenerationOutput(Image.new("RGB", image.size, color), {})

        def close(self):
            pass

    monkeypatch.setattr("visionbench.runner._backend", lambda _: AlternatingBackend())
    generate(config)
    result = json.loads(evaluate(config).read_text())
    assert result["metrics"]["white_fraction"] == pytest.approx(0.5)
    assert all(sample["metrics"]["white_fraction"] == 0.5 for sample in result["samples"])
    assert len(list((config.run_dir / "images").glob("*.png"))) == 8


def test_transient_retry_and_fatal_error(tmp_path, monkeypatch):
    config = make_config(tmp_path)

    class FlakyBackend:
        calls = 0

        def generate(self, image, prompt, system_prompt, seed):
            self.calls += 1
            if self.calls == 1:
                raise TransientModelError("temporary")
            return GenerationOutput(image, {})

        def close(self):
            pass

    monkeypatch.setattr("visionbench.runner._backend", lambda _: FlakyBackend())
    generate(config)
    first = json.loads((config.run_dir / "checkpoints" / "00000000_trial_000.json").read_text())
    assert first["attempts"] == 2

    changed = make_config(tmp_path / "other")

    class FatalBackend:
        def generate(self, image, prompt, system_prompt, seed):
            raise FatalModelError("bad credentials")

        def close(self):
            pass

    monkeypatch.setattr("visionbench.runner._backend", lambda _: FatalBackend())
    with pytest.raises(FatalModelError):
        generate(changed)


def test_sample_generation_failure_and_scoring_coverage(tmp_path, monkeypatch):
    config = make_config(tmp_path, aggregator=False)

    class PartialBackend:
        calls = 0

        def generate(self, image, prompt, system_prompt, seed):
            self.calls += 1
            if self.calls == 1:
                raise SampleModelError("one refused sample")
            return GenerationOutput(image, {})

        def close(self):
            pass

    monkeypatch.setattr("visionbench.runner._backend", lambda _: PartialBackend())
    summary = generate(config)
    assert summary["trials_failed"] == 1
    result = json.loads(evaluate(config).read_text())
    assert result["coverage"]["generation_trials_success"] == 3
    assert result["coverage"]["metric_samples"]["white_fraction"] == 3


def test_scoring_exception_and_invalid_metric_continue(tmp_path):
    config = make_config(tmp_path, aggregator=False)
    generate(config)
    evaluator = tmp_path / "tasks" / "evaluators" / "custom-evaluator.py"
    evaluator.write_text("""
METRICS = {'value': {'unit': 'fraction', 'higher_is_better': True}}
def score(image, sample, config):
    if sample.sample_id == 'sample_01':
        raise RuntimeError('bad score')
    if sample.sample_id == 'sample_02':
        return {'value': float('nan')}
    return {'value': 0.75}
""")
    raw = yaml.safe_load(config.source.read_text())
    raw["evaluator"] = {"path": evaluator.name, "params": {}}
    config.source.write_text(yaml.safe_dump(raw))
    result = json.loads(evaluate(load_config(config.source)).read_text())
    assert result["metrics"]["value"] == 0.75
    assert result["coverage"]["scoring_failures"] == 1
    assert result["coverage"]["invalid_metric_trials"]["value"] == 1
    assert result["coverage"]["metric_samples"]["value"] == 2


def test_dataset_error_stops_during_preflight(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    loader = tmp_path / "tasks" / "dataloaders" / "broken-loader.py"
    loader.write_text("""
from PIL import Image
from visionbench.types import Sample
class Dataset:
    def __len__(self): return 2
    def __getitem__(self, index):
        if index == 1: raise FileNotFoundError('missing.png')
        return Sample('first', Image.new('RGB', (16, 16)), 'edit')
def load_dataset(config): return Dataset()
""")
    raw = yaml.safe_load(config.source.read_text())
    raw["dataset"] = {"path": loader.name, "params": {}}
    config.source.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr("visionbench.runner._backend", lambda _: pytest.fail("backend loaded before preflight"))
    with pytest.raises(RuntimeError, match="Dataset failed to load index 1"):
        generate(load_config(config.source))


def test_custom_aggregator_gets_lazy_sample_and_output_paths(tmp_path):
    config = make_config(tmp_path)
    generate(config)
    aggregator = tmp_path / "tasks" / "aggregators" / "full-records.py"
    aggregator.write_text("""
def aggregate(records, config):
    assert len(records) == 4
    assert all(row['sample'].ground_truth['target_background'] == '#FFFFFF' for row in records)
    assert all(len(row['output_paths']) == 1 and row['output_paths'][0].is_file() for row in records)
    return {'white_fraction': sum(row['metrics']['white_fraction'] for row in records) / len(records)}
""")
    raw = yaml.safe_load(config.source.read_text())
    raw["aggregator"] = {"path": aggregator.name, "params": {}}
    config.source.write_text(yaml.safe_dump(raw))
    result = json.loads(evaluate(load_config(config.source)).read_text())
    assert 0 < result["metrics"]["white_fraction"] < 1


def test_remote_backend_uses_configured_concurrency(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    data = dict(base.data)
    data["model"] = {"backend": "openai", "id": "mock-remote"}
    data["concurrency"] = 3
    config = RunConfig(base.source, data)
    barrier = threading.Barrier(3, timeout=2)

    class ConcurrentBackend:
        def generate(self, image, prompt, system_prompt, seed):
            barrier.wait()
            return GenerationOutput(image, {})

        def close(self):
            pass

    monkeypatch.setattr("visionbench.runner._backend", lambda _: ConcurrentBackend())
    # The fourth request need not wait on the first three-party barrier.
    calls = {"count": 0}
    original = ConcurrentBackend.generate

    def first_three(self, image, prompt, system_prompt, seed):
        calls["count"] += 1
        if calls["count"] <= 3:
            return original(self, image, prompt, system_prompt, seed)
        return GenerationOutput(image, {})

    monkeypatch.setattr(ConcurrentBackend, "generate", first_three)
    summary = generate(config)
    assert summary["trials_succeeded"] == 4
