"""Shared task-facing types for datasets and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass
class Sample:
    """One deterministic dataset item supplied by a task loader."""

    sample_id: str
    image: Image.Image
    prompt: str
    ground_truth: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Dataset(Protocol):
    """The minimal random-access dataset interface used by VisionBench."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Sample: ...


@dataclass(frozen=True)
class MetricDefinition:
    """Describes a metric returned by an evaluator's ``score`` function."""

    name: str
    unit: str
    higher_is_better: bool

