"""Loading and validating task plugins stored in arbitrary Python files."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from PIL import Image

from .types import Dataset, MetricDefinition, Sample


class PluginError(ValueError):
    """Raised when a task plugin does not meet the public plugin contract."""


DatasetLoader = Callable[[dict[str, Any]], Dataset]
ScoreFunction = Callable[[Image.Image, Sample, dict[str, Any]], dict[str, float]]
PrepareOutputFunction = Callable[[Image.Image, Sample, dict[str, Any]], Image.Image]
AggregateFunction = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, float]]


@dataclass(frozen=True)
class DatasetPlugin:
    load_dataset: DatasetLoader
    parameters: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class EvaluatorPlugin:
    _score: ScoreFunction
    metrics: dict[str, MetricDefinition]
    parameters: dict[str, Any]
    path: Path
    _prepare_output: PrepareOutputFunction | None = None

    def score(self, image: Image.Image, sample: Sample) -> dict[str, float]:
        return invoke_plugin(self._score, image, sample, self.parameters)

    def prepare_output(self, image: Image.Image, sample: Sample) -> Image.Image:
        if self._prepare_output is None:
            return image
        return invoke_plugin(self._prepare_output, image, sample, self.parameters)


@dataclass(frozen=True)
class AggregatorPlugin:
    _aggregate: AggregateFunction
    parameters: dict[str, Any]
    path: Path

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, float]:
        return invoke_plugin(self._aggregate, records, self.parameters)


def load_module(path: str | Path) -> ModuleType:
    """Import a plugin file without requiring its filename to be an identifier."""
    plugin_path = Path(path).expanduser().resolve()
    if not plugin_path.is_file():
        raise PluginError(f"Plugin file does not exist: {plugin_path}")
    digest = hashlib.sha256(str(plugin_path).encode()).hexdigest()[:16]
    module_name = f"visionbench_plugin_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise PluginError(f"Cannot import plugin: {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


# A descriptive alias for callers that previously used this name.
load_module_from_path = load_module


def _function(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise PluginError(f"{module.__file__} must define callable {name}(...)")
    return value


def _parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise PluginError("Plugin parameters must be a mapping")
    return dict(parameters)


def _metric_definitions(module: ModuleType) -> dict[str, MetricDefinition]:
    raw = getattr(module, "METRICS", getattr(module, "metric_definitions", None))
    if raw is None:
        raise PluginError(f"{module.__file__} must define METRICS with metric unit and direction")
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, (list, tuple)):
        items = ((item.name if isinstance(item, MetricDefinition) else item.get("name"), item) for item in raw)
    else:
        raise PluginError("METRICS must be a mapping or a list of metric definitions")

    definitions: dict[str, MetricDefinition] = {}
    for name, value in items:
        if isinstance(value, MetricDefinition):
            definition = value
            if name is not None and name != definition.name:
                raise PluginError("Metric mapping key must match MetricDefinition.name")
        elif isinstance(value, Mapping):
            metric_name = value.get("name", name)
            try:
                definition = MetricDefinition(
                    name=str(metric_name), unit=str(value["unit"]), higher_is_better=bool(value["higher_is_better"])
                )
            except KeyError as exc:
                raise PluginError(f"Metric {name!r} requires unit and higher_is_better") from exc
        else:
            raise PluginError(f"Metric {name!r} is not a MetricDefinition or mapping")
        if not definition.name or definition.name in definitions:
            raise PluginError(f"Metric names must be unique and non-empty: {definition.name!r}")
        definitions[definition.name] = definition
    if not definitions:
        raise PluginError("METRICS cannot be empty")
    return definitions


def load_dataset_plugin(path: str | Path, parameters: Mapping[str, Any] | None = None) -> DatasetPlugin:
    module = load_module(path)
    return DatasetPlugin(_function(module, "load_dataset"), _parameters(parameters), Path(path).resolve())


def load_evaluator_plugin(path: str | Path, parameters: Mapping[str, Any] | None = None) -> EvaluatorPlugin:
    module = load_module(path)
    prepare = getattr(module, "prepare_output", None)
    if prepare is not None and not callable(prepare):
        raise PluginError(f"{module.__file__}.prepare_output must be callable when present")
    return EvaluatorPlugin(
        _score=_function(module, "score"), metrics=_metric_definitions(module),
        parameters=_parameters(parameters), path=Path(path).resolve(), _prepare_output=prepare,
    )


def load_aggregator_plugin(path: str | Path, parameters: Mapping[str, Any] | None = None) -> AggregatorPlugin:
    module = load_module(path)
    return AggregatorPlugin(_function(module, "aggregate"), _parameters(parameters), Path(path).resolve())


def load_dataset(path: str | Path, params: Mapping[str, Any] | None = None) -> Dataset:
    """Load a dataset immediately using its task-specific parameter dictionary."""
    plugin = load_dataset_plugin(path, params)
    dataset = invoke_plugin(plugin.load_dataset, plugin.parameters)
    if not isinstance(dataset, Dataset):
        raise PluginError(f"load_dataset in {plugin.path} must return a __len__/__getitem__ dataset")
    return dataset


def load_evaluator(path: str | Path, params: Mapping[str, Any] | None = None) -> EvaluatorPlugin:
    """Load an evaluator with parameters bound to score and prepare_output."""
    return load_evaluator_plugin(path, params)


def load_aggregator(path: str | Path, params: Mapping[str, Any] | None = None) -> AggregatorPlugin:
    """Load an aggregator with its parameter dictionary bound to aggregate."""
    return load_aggregator_plugin(path, params)


def invoke_plugin(function: Callable[..., Any], *args: Any) -> Any:
    """Call a fixed plugin function, rejecting signatures that cannot accept it."""
    try:
        inspect.signature(function).bind(*args)
    except TypeError as exc:
        raise PluginError(f"Plugin function {function.__name__} has an incompatible signature: {exc}") from exc
    return function(*args)
