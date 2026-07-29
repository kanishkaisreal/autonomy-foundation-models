from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the project configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ProjectSection:
    name: str
    version: str
    seed: int


@dataclass(frozen=True, slots=True)
class PathsSection:
    data_root: Path
    output_root: Path
    cache_root: Path


@dataclass(frozen=True, slots=True)
class RuntimeSection:
    device: str
    precision: str
    num_workers: int


@dataclass(frozen=True, slots=True)
class ModelsSection:
    vlm_checkpoint: str
    text_checkpoint: str
    radar_checkpoint: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    project: ProjectSection
    paths: PathsSection
    runtime: RuntimeSection
    models: ModelsSection


_ALLOWED_FIELDS = {
    "project": {"name", "version", "seed"},
    "paths": {"data_root", "output_root", "cache_root"},
    "runtime": {"device", "precision", "num_workers"},
    "models": {"vlm_checkpoint", "text_checkpoint", "radar_checkpoint"},
}

_REQUIRED_FIELDS = {
    "project": {"name", "version", "seed"},
    "paths": {"data_root", "output_root", "cache_root"},
    "runtime": {"device", "precision", "num_workers"},
    "models": {"vlm_checkpoint", "text_checkpoint"},
}


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name!r} must be a YAML mapping.")
    return dict(value)


def _check_fields(section_name: str, section: dict[str, Any]) -> None:
    unknown = set(section) - _ALLOWED_FIELDS[section_name]
    missing = _REQUIRED_FIELDS[section_name] - set(section)

    if unknown:
        raise ConfigError(f"Unknown field(s) in {section_name!r}: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"Missing field(s) in {section_name!r}: {sorted(missing)}")


def _apply_environment_overrides(
    raw_config: dict[str, Any],
    environ: Mapping[str, str],
    prefix: str,
) -> dict[str, Any]:
    config = copy.deepcopy(raw_config)

    for variable_name, raw_value in environ.items():
        if not variable_name.startswith(prefix):
            continue

        path = variable_name[len(prefix) :].lower().split("__")
        if len(path) != 2:
            raise ConfigError(
                f"Invalid override {variable_name!r}; expected {prefix}SECTION__FIELD."
            )

        section_name, field_name = path
        if section_name not in _ALLOWED_FIELDS:
            raise ConfigError(f"Unknown override section: {section_name!r}")
        if field_name not in _ALLOWED_FIELDS[section_name]:
            raise ConfigError(f"Unknown override field: {section_name}.{field_name}")

        parsed_value = yaml.safe_load(raw_value)
        if isinstance(parsed_value, (dict, list)):
            raise ConfigError("Environment overrides must be scalar values.")

        section = _as_mapping(config[section_name], section_name)
        section[field_name] = parsed_value
        config[section_name] = section

    return config


def _expect(value: Any, expected_type: type, field_name: str) -> Any:
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{field_name!r} must be an integer.")
        return value

    if not isinstance(value, expected_type):
        raise ConfigError(f"{field_name!r} must be a {expected_type.__name__}.")
    return value


def load_project_config(
    config_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    env_prefix: str = "AFM__",
) -> ProjectConfig:
    """Load a strict typed project configuration from YAML."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    root = _as_mapping(raw, "config")
    unknown_sections = set(root) - set(_ALLOWED_FIELDS)
    missing_sections = set(_ALLOWED_FIELDS) - set(root)

    if unknown_sections:
        raise ConfigError(f"Unknown section(s): {sorted(unknown_sections)}")
    if missing_sections:
        raise ConfigError(f"Missing section(s): {sorted(missing_sections)}")

    resolved = _apply_environment_overrides(
        root,
        os.environ if environ is None else environ,
        env_prefix,
    )

    project = _as_mapping(resolved["project"], "project")
    paths = _as_mapping(resolved["paths"], "paths")
    runtime = _as_mapping(resolved["runtime"], "runtime")
    models = _as_mapping(resolved["models"], "models")

    for name, section in (
        ("project", project),
        ("paths", paths),
        ("runtime", runtime),
        ("models", models),
    ):
        _check_fields(name, section)

    radar_checkpoint = models.get("radar_checkpoint")
    if radar_checkpoint is not None and not isinstance(radar_checkpoint, str):
        raise ConfigError("'models.radar_checkpoint' must be a string or null.")

    return ProjectConfig(
        project=ProjectSection(
            name=_expect(project["name"], str, "project.name"),
            version=_expect(project["version"], str, "project.version"),
            seed=_expect(project["seed"], int, "project.seed"),
        ),
        paths=PathsSection(
            data_root=Path(_expect(paths["data_root"], str, "paths.data_root")),
            output_root=Path(_expect(paths["output_root"], str, "paths.output_root")),
            cache_root=Path(_expect(paths["cache_root"], str, "paths.cache_root")),
        ),
        runtime=RuntimeSection(
            device=_expect(runtime["device"], str, "runtime.device"),
            precision=_expect(runtime["precision"], str, "runtime.precision"),
            num_workers=_expect(runtime["num_workers"], int, "runtime.num_workers"),
        ),
        models=ModelsSection(
            vlm_checkpoint=_expect(
                models["vlm_checkpoint"], str, "models.vlm_checkpoint"
            ),
            text_checkpoint=_expect(
                models["text_checkpoint"], str, "models.text_checkpoint"
            ),
            radar_checkpoint=radar_checkpoint,
        ),
    )


def main() -> None:
    """Run function 001 manually with F5."""
    root = Path(__file__).resolve().parents[2]
    config = load_project_config(root / "configs" / "base.yaml")

    print(config)
    print(f"Project: {config.project.name}")
    print(f"Seed: {config.project.seed}")
    print(f"Requested device: {config.runtime.device}")
    print(f"Data root: {config.paths.data_root}")
    print(f"VLM checkpoint: {config.models.vlm_checkpoint}")


if __name__ == "__main__":
    main()
