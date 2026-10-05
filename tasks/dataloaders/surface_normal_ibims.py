"""Surface normal loader: iBims-1 (100 images, Marigold split)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "visionbench_task_surface_normal_common", Path(__file__).with_name("_surface_normal_common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def load_dataset(config: dict):
    return _common.build(config, dataset_name="ibims", default_split="ibims_test.txt", default_subset=None)
