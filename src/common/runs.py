from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml


from common.config import (
    ProjectConfig,
    load_project_config,
    validate_project_config,
)
from common.environment import collect_environment_report

def _slugify(value: str) -> str:
    """
    Convert text into a filesystem-safe identifier component.

    Example:
        "VLM SFT Experiment" -> "vlm-sft-experiment"
    """
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")

def _make_serializable(value: Any) -> Any:
    """
    Convert configuration values into JSON/YAML-compatible Python values.

    In particular, pathlib.Path objects are converted into strings.
    """
    if is_dataclass(value):
        return _make_serializable(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _make_serializable(child_value)
            for key, child_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_make_serializable(item) for item in value]

    return value


def _get_git_status_short(project_root: Path) -> list[str] | None:
    """
    Return concise Git working-tree changes.

    None means Git was unavailable or project_root was not a Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return [
        line
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def create_experiment_id(
    project_name: str,
    experiment_name: str | None = None,
) -> str:
    """
    Create a readable and practically unique experiment identifier.

    Format:
        timestamp__project-name__experiment-name__unique-suffix

    Args:
        project_name:
            Name of the project that owns the experiment.
        experiment_name:
            Optional human-readable description such as ``vlm-sft`` or
            ``radar-baseline``.

    Returns:
        A filesystem-safe experiment identifier.

    Raises:
        ValueError:
            If the project or experiment name cannot produce a valid slug.
    """
    project_slug = _slugify(project_name)

    if not project_slug:
        raise ValueError(
            "project_name must contain at least one letter or number."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unique_suffix = uuid.uuid4().hex[:8]

    parts = [
        timestamp,
        project_slug,
    ]

    if experiment_name is not None:
        experiment_slug = _slugify(experiment_name)

        if not experiment_slug:
            raise ValueError(
                "experiment_name must contain at least one letter or number."
            )

        parts.append(experiment_slug)

    parts.append(unique_suffix)

    return "__".join(parts)

def create_run_directory(
    output_root: str | Path,
    experiment_id: str,
) -> Path:
    """
    Create an isolated directory for one experiment run.

    Args:
        output_root:
            Parent directory under which experiment runs are stored.
        experiment_id:
            Unique identifier returned by ``create_experiment_id``.

    Returns:
        The absolute path to the newly created run directory.

    Raises:
        ValueError:
            If the experiment ID is empty or contains path separators.
        FileExistsError:
            If a run directory with the same ID already exists.
    """
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty.")

    experiment_path = Path(experiment_id)

    if (
        experiment_path.is_absolute()
        or experiment_path.name != experiment_id
        or experiment_id in {".", ".."}
    ):
        raise ValueError(
            "experiment_id must be a single directory name, "
            "not a path."
        )

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    run_directory = root / experiment_id

    run_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    return run_directory


def snapshot_run_config(
    run_directory: str | Path,
    config: ProjectConfig,
    *,
    project_root: str | Path,
    command: Sequence[str] | None = None,
    dataset_versions: Mapping[str, str | None] | None = None,
) -> dict[str, Path]:
    """
    Save the resolved configuration and execution context before a run.

    Args:
        run_directory:
            Existing directory created by ``create_run_directory``.
        config:
            Fully resolved typed configuration returned by
            ``load_project_config``.
        project_root:
            Root of the Git repository.
        command:
            Command used to launch the run. Defaults to ``sys.argv``.
        dataset_versions:
            Optional dataset names, releases, or manifest checksums.

    Returns:
        Paths to the three snapshot files that were created.
    """
    run_path = Path(run_directory).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()

    if not run_path.is_dir():
        raise ValueError(
            f"run_directory must already exist and be a directory: {run_path}"
        )

    if not root.is_dir():
        raise ValueError(
            f"project_root must be an existing directory: {root}"
        )

    # This is the resolved typed configuration, not the original YAML text.
    resolved_config = _make_serializable(config)

    config_path = run_path / "resolved_config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            resolved_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    environment_path = run_path / "environment.json"

    environment_report = collect_environment_report(
        project_root=root,
        requested_device=config.runtime.device,
        model_versions={
            "vlm_checkpoint": config.models.vlm_checkpoint,
            "text_checkpoint": config.models.text_checkpoint,
            "radar_checkpoint": config.models.radar_checkpoint,
        },
        dataset_versions=dataset_versions,
        output_path=environment_path,
    )

    command_parts = [
        str(part)
        for part in (sys.argv if command is None else command)
    ]

    snapshot = {
        "created_at_utc": environment_report["created_at_utc"],
        "command": command_parts,
        "command_display": shlex.join(command_parts),
        "git": {
            **environment_report["git"],
            "status_short": _get_git_status_short(root),
        },
        "files": {
            "resolved_config": config_path.name,
            "environment": environment_path.name,
        },
    }

    snapshot_path = run_path / "run_snapshot.json"

    snapshot_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "resolved_config": config_path,
        "environment": environment_path,
        "run_snapshot": snapshot_path,
    }


def main() -> None:
    """Verify experiment creation and snapshotting using temporary files."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / "configs" / "base.yaml"

    config = load_project_config(config_path)
    validate_project_config(
        config,
        project_root=repository_root,
    )

    experiment_id = create_experiment_id(
        project_name=config.project.name,
        experiment_name="foundation-setup",
    )

    with TemporaryDirectory(prefix="afm-run-check-") as temporary_root:
        run_directory = create_run_directory(
            output_root=temporary_root,
            experiment_id=experiment_id,
        )

        snapshot_paths = snapshot_run_config(
            run_directory=run_directory,
            config=config,
            project_root=repository_root,
            command=[
                "python",
                "src/common/runs.py",
            ],
            dataset_versions={
                "DriveLM": None,
                "nuScenes": None,
            },
        )

        print("Temporary run snapshot created successfully.")
        print()
        print(f"Experiment ID: {experiment_id}")
        print(f"Run directory: {run_directory}")
        print()

        for artifact_name, artifact_path in snapshot_paths.items():
            print(f"{artifact_name}: {artifact_path.name}")

        print()
        print(
            "All snapshot files exist:",
            all(path.exists() for path in snapshot_paths.values()),
        )

    print()
    print("Temporary verification files cleaned up automatically.")


if __name__ == "__main__":
    main()
