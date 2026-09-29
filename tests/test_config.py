import pytest
import yaml
from pathlib import Path

from visionbench.config import load_config


def _config(tmp_path):
    return {
        "task_name": "task",
        "generation_version": "v1",
        "output_root": str(tmp_path / "outputs"),
        "system_prompt": "line one\nline two",
        "model": {"backend": "openai", "id": "model-v1", "api_key": "${TEST_IMAGE_API_KEY}"},
        "dataset": {"path": "loader.py", "params": {}},
        "evaluator": {"path": "evaluator.py", "params": {}},
    }


def test_env_reference_is_expanded_and_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_IMAGE_API_KEY", "secret-value")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path)))
    config = load_config(path)
    assert config.data["model"]["api_key"] == "secret-value"
    assert config.snapshot()["model"]["api_key"] == "[REDACTED]"
    assert "secret-value" not in str(config.generation_settings())


def test_literal_key_is_rejected(tmp_path):
    raw = _config(tmp_path)
    raw["model"]["api_key"] = "secret-value"
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="environment-variable reference"):
        load_config(path)


@pytest.mark.parametrize("section,path_value", [
    ("dataset", "../outside.py"),
    ("evaluator", "/tmp/outside.py"),
    ("aggregator", "../outside.py"),
    ("raw_path", "../outside"),
    ("assets_path", "/tmp/outside"),
])
def test_scoped_yaml_paths_reject_escapes(tmp_path, monkeypatch, section, path_value):
    monkeypatch.setenv("TEST_IMAGE_API_KEY", "secret-value")
    raw = _config(tmp_path)
    if section in {"raw_path", "assets_path"}:
        raw["dataset"][section] = path_value
    else:
        raw.setdefault(section, {})["path"] = path_value
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="relative to"):
        load_config(path)


def test_relative_paths_resolve_inside_dedicated_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_IMAGE_API_KEY", "secret-value")
    raw = _config(tmp_path)
    raw["dataset"]["raw_path"] = "my_task"
    raw["dataset"]["assets_path"] = "my_task"
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.scoped_path("dataset", raw["dataset"]["path"]) == tmp_path / "tasks" / "dataloaders" / "loader.py"
    assert config.scoped_path("evaluator", raw["evaluator"]["path"]) == tmp_path / "tasks" / "evaluators" / "evaluator.py"
    assert config.scoped_path("raw", raw["dataset"]["raw_path"]) == tmp_path / "raw" / "my_task"
    assert config.scoped_path("assets", raw["dataset"]["assets_path"]) == tmp_path / "assets" / "my_task"


def test_config_name_resolves_from_configs_outside_project_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("config_dummy")
    assert config.source == Path(__file__).resolve().parents[1] / "configs" / "config_dummy.yaml"
    assert config.scoped_path("dataset", config.data["dataset"]["path"]).is_file()


def test_unknown_config_name_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="configs/"):
        load_config("missing_name")
