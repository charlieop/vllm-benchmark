from pathlib import Path

import pytest
from PIL import Image

from visionbench.plugins import PluginError, invoke_plugin, load_dataset, load_dataset_plugin, load_evaluator, load_evaluator_plugin
from visionbench.types import MetricDefinition, Sample


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_loads_hyphenated_dataset_and_evaluator_plugins(tmp_path):
    dataset_path = tmp_path / "toy-dataset.py"
    _write(dataset_path, """
def load_dataset(config):
    return [config['name']]
""")
    evaluator_path = tmp_path / "white-score.py"
    _write(evaluator_path, """
METRICS = {'white_fraction': {'unit': 'fraction', 'higher_is_better': True}}
def score(image, sample, config):
    return {'white_fraction': config['value']}
def prepare_output(image, sample, config):
    return image
""")

    dataset = load_dataset_plugin(dataset_path, {"name": "example"})
    assert invoke_plugin(dataset.load_dataset, dataset.parameters) == ["example"]
    assert load_dataset(dataset_path, {"name": "example"}) == ["example"]

    evaluator = load_evaluator_plugin(evaluator_path, {"value": 0.75})
    assert evaluator.metrics == {"white_fraction": MetricDefinition("white_fraction", "fraction", True)}
    sample = Sample("id", Image.new("RGB", (1, 1)), "prompt")
    assert evaluator.score(sample.image, sample) == {"white_fraction": 0.75}
    assert evaluator.prepare_output(sample.image, sample) is sample.image
    assert load_evaluator(evaluator_path, {"value": 0.75}).score(sample.image, sample) == {"white_fraction": 0.75}


def test_metrics_and_fixed_function_contract_are_validated(tmp_path):
    path = tmp_path / "bad.py"
    _write(path, "def score(image, sample, config): return {}\n")
    with pytest.raises(PluginError, match="METRICS"):
        load_evaluator_plugin(path)

    path = tmp_path / "signature.py"
    _write(path, "METRICS = {'x': {'unit': 'n', 'higher_is_better': False}}\ndef score(one): return {}\n")
    evaluator = load_evaluator_plugin(path)
    with pytest.raises(PluginError, match="incompatible signature"):
        evaluator.score(None, None)
