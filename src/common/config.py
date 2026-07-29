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

class ConfigValidationError(ValueError):
    """Raised when loaded configuration values are operationally invalid."""
    
    
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

ALLOWED_DEVICES = {"auto", "cpu", "mps", "cuda"}

ALLOWED_PRECISIONS = {"fp32", "fp16", "bf16"}

DEVICE_PRECISION_POLICY = {
    "auto": {"fp32", "fp16", "bf16"},
    "cpu": {"fp32"},
    "mps": {"fp32", "fp16"},
    "cuda": {"fp32", "fp16", "bf16"},
}



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


def validate_project_config(
    config: ProjectConfig,
    *,
    project_root: str | Path | None = None,
) -> None:
    """
    Validate that a loaded ProjectConfig is operationally coherent.

    This function validates declared configuration values but does not inspect
    actual accelerator availability. CPU, MPS, and CUDA capability detection
    belongs to ``resolve_compute_device``.

    Args:
        config:
            Typed configuration returned by ``load_project_config``.
        project_root:
            Repository root used to resolve relative paths. When omitted, the
            root is inferred from this module's location.

    Raises:
        ConfigValidationError:
            If one or more configuration values are invalid.
    """
    root = (
        Path(project_root).expanduser()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Project metadata
    # ------------------------------------------------------------------
    if not config.project.name.strip():
        errors.append("project.name must not be empty.")

    if not config.project.version.strip():
        errors.append("project.version must not be empty.")

    if config.project.seed < 0:
        errors.append(
            f"project.seed must be nonnegative; received {config.project.seed}."
        )

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------
    device = config.runtime.device

    if device not in ALLOWED_DEVICES:
        errors.append(
            f"runtime.device must be one of {sorted(ALLOWED_DEVICES)}; "
            f"received {device!r}."
        )

    precision = config.runtime.precision

    if precision not in ALLOWED_PRECISIONS:
        errors.append(
            f"runtime.precision must be one of {sorted(ALLOWED_PRECISIONS)}; "
            f"received {precision!r}."
        )

    if device in DEVICE_PRECISION_POLICY and precision in ALLOWED_PRECISIONS:
        supported_precisions = DEVICE_PRECISION_POLICY[device]

        if precision not in supported_precisions:
            errors.append(
                f"runtime.precision={precision!r} is not supported by this "
                f"project when runtime.device={device!r}. "
                f"Allowed values: {sorted(supported_precisions)}."
            )

    if config.runtime.num_workers < 0:
        errors.append(
            "runtime.num_workers must be nonnegative; "
            f"received {config.runtime.num_workers}."
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def resolve_path(path: Path) -> Path:
        expanded = path.expanduser()
        return expanded if expanded.is_absolute() else root / expanded

    data_root = resolve_path(config.paths.data_root)
    output_root = resolve_path(config.paths.output_root)
    cache_root = resolve_path(config.paths.cache_root)

    if not data_root.exists():
        errors.append(
            f"paths.data_root does not exist: {data_root}. "
            "Create the directory or update configs/base.yaml."
        )
    elif not data_root.is_dir():
        errors.append(
            f"paths.data_root must be a directory: {data_root}."
        )

    for field_name, path in (
        ("paths.output_root", output_root),
        ("paths.cache_root", cache_root),
    ):
        if path.exists() and not path.is_dir():
            errors.append(
                f"{field_name} must be a directory when it exists: {path}."
            )

        if not path.exists() and not path.parent.exists():
            errors.append(
                f"The parent directory for {field_name} does not exist: "
                f"{path.parent}."
            )

    if data_root == output_root:
        errors.append(
            "paths.data_root and paths.output_root must be different directories."
        )

    if data_root == cache_root:
        errors.append(
            "paths.data_root and paths.cache_root must be different directories."
        )

    # ------------------------------------------------------------------
    # Pretrained checkpoints
    # ------------------------------------------------------------------
    def validate_checkpoint(
        field_name: str,
        checkpoint: str | None,
        *,
        optional: bool,
    ) -> None:
        if checkpoint is None:
            if optional:
                return

            errors.append(f"{field_name} must not be null.")
            return

        if not checkpoint.strip():
            errors.append(f"{field_name} must not be empty.")
            return

        if "\n" in checkpoint or "\r" in checkpoint:
            errors.append(
                f"{field_name} must be a single-line checkpoint identifier."
            )

    validate_checkpoint(
        "models.vlm_checkpoint",
        config.models.vlm_checkpoint,
        optional=False,
    )
    validate_checkpoint(
        "models.text_checkpoint",
        config.models.text_checkpoint,
        optional=False,
    )
    validate_checkpoint(
        "models.radar_checkpoint",
        config.models.radar_checkpoint,
        optional=True,
    )

    if errors:
        formatted_errors = "\n".join(
            f"  {index}. {message}"
            for index, message in enumerate(errors, start=1)
        )

        raise ConfigValidationError(
            "Project configuration validation failed:\n"
            f"{formatted_errors}"
        )
        
        

def main() -> None:
    """Load and validate the base configuration for manual F5 execution."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / "configs" / "base.yaml"

    config = load_project_config(config_path)
    validate_project_config(
        config,
        project_root=repository_root,
    )

    print("Configuration loaded and validated successfully.")
    print()
    print(config)
    print()
    print(f"Project name: {config.project.name}")
    print(f"Seed: {config.project.seed}")
    print(f"Requested device: {config.runtime.device}")
    print(f"Precision: {config.runtime.precision}")
    print(f"Data root: {repository_root / config.paths.data_root}")
    print(f"Output root: {repository_root / config.paths.output_root}")
    print(f"Pretrained VLM checkpoint: {config.models.vlm_checkpoint}")


if __name__ == "__main__":
    main()