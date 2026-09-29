"""Validated YAML configuration and environment references."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SECRET_KEYS = {"api_key", "aws_access_key_id", "aws_secret_access_key", "session_token"}
_SCOPED_FOLDERS = {
    "dataset": "tasks/dataloaders",
    "evaluator": "tasks/evaluators",
    "aggregator": "tasks/aggregators",
    "assets": "assets",
    "raw": "raw",
}


def _config_source(value: str | Path) -> Path:
    """Accept a path or a config name from the project's configs/ folder."""
    requested = Path(value).expanduser()
    if requested.is_file():
        return requested.resolve()
    if requested.parent == Path(".") and requested.suffix in {"", ".yaml", ".yml"}:
        name = requested.stem if requested.suffix else requested.name
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
                candidate = root / "configs" / f"{name}.yaml"
                if candidate.is_file():
                    return candidate.resolve()
    raise FileNotFoundError(f"Configuration not found: {value}. Use a name from configs/ or a YAML path.")


def _load_env_file(config_path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Missing environment variable {name}")
            return os.environ[name]

        return _ENV.sub(replace, value)
    return value


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if key.lower() in _SECRET_KEYS and item else _redact(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


@dataclass(frozen=True)
class RunConfig:
    source: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.resolve(self.data["output_root"])

    @property
    def run_dir(self) -> Path:
        return self.root / self.data["task_name"] / self.data["generation_version"]

    @property
    def workspace_root(self) -> Path:
        """Find the project containing this YAML, regardless of its subfolder."""
        for parent in (self.source.parent, *self.source.parents):
            if (parent / "pyproject.toml").is_file() and (parent / "tasks" / "dataloaders").is_dir():
                return parent
        return self.source.parent

    def scoped_path(self, section: str, path: str | Path) -> Path:
        """Resolve a YAML path inside its dedicated project folder."""
        if section not in _SCOPED_FOLDERS:
            raise ValueError(f"Unknown path section: {section}")
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError(f"{section} path must be a nonempty relative path")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{section} path must be relative to {_SCOPED_FOLDERS[section]}/")
        base = (self.workspace_root / _SCOPED_FOLDERS[section]).resolve()
        resolved = (base / relative).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"{section} path escapes {_SCOPED_FOLDERS[section]}/")
        return resolved

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (self.source.parent / candidate).resolve()

    def snapshot(self) -> dict[str, Any]:
        return _redact(copy.deepcopy(self.data))

    def generation_settings(self) -> dict[str, Any]:
        """Settings that may not change while reusing a generation version."""
        return _redact({key: copy.deepcopy(self.data[key]) for key in
                        ("task_name", "system_prompt", "dataset", "model", "trials", "base_seed")})


def load_config(path: str | Path) -> RunConfig:
    source = _config_source(path)
    _load_env_file(source)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install with uv sync") from exc
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for field in ("task_name", "generation_version", "output_root", "system_prompt", "model", "dataset", "evaluator"):
        if field not in raw:
            raise ValueError(f"Missing configuration field: {field}")
    for field in ("task_name", "generation_version"):
        value = raw[field]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
            raise ValueError(f"{field} must be a safe nonempty directory name")
    for field in ("model", "dataset", "evaluator"):
        if not isinstance(raw[field], dict):
            raise ValueError(f"{field} must be a mapping")
    if raw.get("aggregator") is not None and not isinstance(raw["aggregator"], dict):
        raise ValueError("aggregator must be a mapping")
    for field in ("dataset", "evaluator"):
        if not raw[field].get("path"):
            raise ValueError(f"{field}.path is required")
    config = RunConfig(source=source, data=raw)
    for section in ("dataset", "evaluator", "aggregator"):
        entry = raw.get(section)
        if entry and entry.get("path"):
            config.scoped_path(section, entry["path"])
    for section, key in (("raw", "raw_path"), ("assets", "assets_path")):
        if key in raw["dataset"]:
            config.scoped_path(section, raw["dataset"][key])
    if not raw["model"].get("backend") or not raw["model"].get("id"):
        raise ValueError("model.backend and model.id are required")
    if "api_key" in raw["model"] and not re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", str(raw["model"]["api_key"])):
        raise ValueError("model.api_key must be an environment-variable reference")
    data = _expand(raw)
    data.setdefault("trials", 1)
    data.setdefault("base_seed", 42)
    data.setdefault("concurrency", 3)
    data.setdefault("retries", {})
    data["retries"].setdefault("max_attempts", 3)
    data["retries"].setdefault("base_delay_s", 1.0)
    data["retries"].setdefault("max_delay_s", 30.0)
    data.setdefault("save_inputs", True)
    data.setdefault("s3", {"enabled": False})
    if not isinstance(data["trials"], int) or data["trials"] < 1:
        raise ValueError("trials must be a positive integer")
    if not isinstance(data["concurrency"], int) or data["concurrency"] < 1:
        raise ValueError("concurrency must be a positive integer")
    if not isinstance(data["base_seed"], int):
        raise ValueError("base_seed must be an integer")
    if not isinstance(data["system_prompt"], str):
        raise ValueError("system_prompt must be multiline YAML text")
    if not isinstance(data["retries"]["max_attempts"], int) or data["retries"]["max_attempts"] < 1:
        raise ValueError("retries.max_attempts must be positive")
    return RunConfig(source=source, data=data)
