from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from common.config import load_project_config, validate_project_config
from common.device import ComputeDevice, resolve_compute_device


DEFAULT_PACKAGES = (
    "torch",
    "numpy",
    "PyYAML",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
)


def _get_package_versions(
    package_names: Sequence[str],
) -> dict[str, str | None]:
    """
    Return installed versions for the requested Python packages.

    A missing optional package is recorded as None rather than causing the
    environment-report generation to fail.
    """
    versions: dict[str, str | None] = {}

    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None

    return versions


def _run_git_command(
    project_root: Path,
    arguments: Sequence[str],
) -> str | None:
    """
    Run one read-only Git command.

    None is returned when Git is unavailable or the directory is not part of
    a Git repository.
    """
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def _get_git_report(project_root: Path) -> dict[str, Any]:
    """Collect the current commit, branch, and working-tree state."""
    commit = _run_git_command(project_root, ["rev-parse", "HEAD"])
    branch = _run_git_command(
        project_root,
        ["rev-parse", "--abbrev-ref", "HEAD"],
    )
    status = _run_git_command(project_root, ["status", "--porcelain"])

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status) if status is not None else None,
    }


def _get_total_memory_bytes() -> int | None:
    """
    Return total system memory without requiring an additional dependency.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        return None


def _get_accelerator_report(
    device: ComputeDevice,
) -> dict[str, Any]:
    """Collect PyTorch accelerator and backend information."""
    cuda_report: dict[str, Any] = {
        "available": device.cuda_available,
        "pytorch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count()
        if device.cuda_available
        else 0,
        "devices": [],
    }

    if device.cuda_available:
        cuda_report["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    index
                ).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]

    return {
        "requested_device": device.requested,
        "resolved_device": device.resolved,
        "device_name": device.device_name,
        "mps": {
            "built": device.mps_built,
            "available": device.mps_available,
        },
        "cuda": cuda_report,
    }


def collect_environment_report(
    *,
    project_root: str | Path,
    requested_device: str = "auto",
    model_versions: Mapping[str, str | None] | None = None,
    dataset_versions: Mapping[str, str | None] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGES,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Capture the software, hardware, Git, model, and dataset environment.

    Args:
        project_root:
            Root of the Git repository.
        requested_device:
            Device requested by the configuration: auto, cpu, mps, or cuda.
        model_versions:
            Model checkpoint identifiers or versions used by the run.
        dataset_versions:
            Dataset names, releases, checksums, or versions used by the run.
        package_names:
            Python packages whose versions should be recorded.
        output_path:
            Optional JSON file path. When provided, the report is written
            there atomically.

    Returns:
        A JSON-serializable environment-report dictionary.
    """
    root = Path(project_root).expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"project_root must be a directory: {root}")

    device = resolve_compute_device(requested_device)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "system": {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "operating_system_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "total_memory_bytes": _get_total_memory_bytes(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "packages": _get_package_versions(package_names),
        "pytorch": {
            "version": torch.__version__,
            "debug_build": torch.version.debug,
        },
        "accelerator": _get_accelerator_report(device),
        "git": _get_git_report(root),
        "models": dict(model_versions or {}),
        "datasets": dict(dataset_versions or {}),
    }

    if output_path is not None:
        destination = Path(output_path).expanduser()

        if not destination.is_absolute():
            destination = root / destination

        destination.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = destination.with_suffix(
            destination.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(destination)

    return report


def main() -> None:
    """Collect a non-writing environment summary using F5."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / "configs" / "base.yaml"

    config = load_project_config(config_path)
    validate_project_config(config, project_root=repository_root)

    report = collect_environment_report(
        project_root=repository_root,
        requested_device=config.runtime.device,
    )
    print(
        f"Environment: Python={report['python']['version']}, "
        f"PyTorch={report['pytorch']['version']}, "
        f"device={report['accelerator']['resolved_device']}, "
        f"git_dirty={report['git']['is_dirty']}"
    )


if __name__ == "__main__":
    main()
