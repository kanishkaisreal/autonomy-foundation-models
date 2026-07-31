from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from common.checksums import compute_file_checksum


class ArtifactManifestError(ValueError):
    """Raised when an artifact manifest cannot be created safely."""


def _infer_artifact_type(path: Path) -> str:
    """Infer a simple artifact category from the file extension."""
    suffix = path.suffix.lower()

    type_by_suffix = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".yaml": "configuration",
        ".yml": "configuration",
        ".log": "log",
        ".txt": "text",
        ".csv": "table",
        ".png": "figure",
        ".jpg": "figure",
        ".jpeg": "figure",
        ".pdf": "report",
        ".pt": "checkpoint",
        ".pth": "checkpoint",
        ".safetensors": "checkpoint",
    }

    return type_by_suffix.get(suffix, "file")


def build_artifact_manifest(
    run_directory: str | Path,
    artifacts: Mapping[str, str | Path],
    *,
    provenance: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    require_all: bool = True,
) -> dict[str, Any]:
    """
    Build and save metadata for the files produced by one experiment run.

    Args:
        run_directory:
            Existing directory containing the run artifacts.
        artifacts:
            Mapping from logical artifact names to paths. Relative paths are
            interpreted relative to the run directory.
        provenance:
            Optional information identifying the run, stage, dataset, or model
            that produced the artifacts.
        output_path:
            Optional manifest destination. Defaults to
            ``run_directory/artifact_manifest.json``.
        require_all:
            Raise an error when one or more declared artifacts are missing.

    Returns:
        The complete artifact-manifest dictionary.

    Raises:
        ArtifactManifestError:
            If paths are invalid, declared files are missing, or the manifest
            cannot be written.
    """
    run_path = Path(run_directory).expanduser().resolve()

    if not run_path.is_dir():
        raise ArtifactManifestError(
            f"run_directory must be an existing directory: {run_path}"
        )

    if not artifacts:
        raise ArtifactManifestError(
            "At least one expected artifact must be declared."
        )

    artifact_entries: list[dict[str, Any]] = []
    missing_artifacts: list[dict[str, str]] = []

    for artifact_name, artifact_path in sorted(artifacts.items()):
        if not artifact_name.strip():
            raise ArtifactManifestError(
                "Artifact names must not be empty."
            )

        resolved_path = Path(artifact_path).expanduser()

        if not resolved_path.is_absolute():
            resolved_path = run_path / resolved_path

        resolved_path = resolved_path.resolve()

        try:
            relative_path = resolved_path.relative_to(run_path)
        except ValueError as exc:
            raise ArtifactManifestError(
                f"Artifact {artifact_name!r} is outside the run directory: "
                f"{resolved_path}"
            ) from exc

        if not resolved_path.exists():
            missing_artifacts.append(
                {
                    "name": artifact_name,
                    "path": relative_path.as_posix(),
                }
            )
            continue

        if not resolved_path.is_file():
            raise ArtifactManifestError(
                f"Artifact {artifact_name!r} must be a file: "
                f"{resolved_path}"
            )

        artifact_entries.append(
            {
                "name": artifact_name,
                "type": _infer_artifact_type(resolved_path),
                "path": relative_path.as_posix(),
                "size_bytes": resolved_path.stat().st_size,
                "sha256": compute_file_checksum(resolved_path),
            }
        )

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": run_path.name,
        "provenance": dict(provenance or {}),
        "artifacts": artifact_entries,
        "missing_artifacts": missing_artifacts,
    }

    destination = (
        run_path / "artifact_manifest.json"
        if output_path is None
        else Path(output_path).expanduser()
    )

    if not destination.is_absolute():
        destination = run_path / destination

    destination = destination.resolve()

    try:
        destination.relative_to(run_path)
    except ValueError as exc:
        raise ArtifactManifestError(
            f"Manifest output must remain inside the run directory: "
            f"{destination}"
        ) from exc

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(destination)

    except (OSError, TypeError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)

        raise ArtifactManifestError(
            f"Could not write artifact manifest {destination}: {exc}"
        ) from exc

    if missing_artifacts and require_all:
        missing_description = ", ".join(
            f"{item['name']} ({item['path']})"
            for item in missing_artifacts
        )

        raise ArtifactManifestError(
            "The following expected artifacts are missing: "
            f"{missing_description}. "
            f"The incomplete manifest was written to {destination}."
        )

    return manifest

def main() -> None:
    """Build a temporary artifact manifest for manual F5 verification."""
    with TemporaryDirectory(
        prefix="afm-artifact-check-"
    ) as temporary_directory:
        run_directory = Path(temporary_directory) / "example-run"
        run_directory.mkdir()

        config_path = run_directory / "resolved_config.yaml"
        metrics_path = run_directory / "metrics.json"
        predictions_path = run_directory / "predictions.jsonl"

        config_path.write_text(
            "project:\n  name: autonomy-foundation-models\n",
            encoding="utf-8",
        )

        metrics_path.write_text(
            '{"accuracy": 0.90}\n',
            encoding="utf-8",
        )

        predictions_path.write_text(
            '{"scene_id":"scene-001","prediction":"stop"}\n',
            encoding="utf-8",
        )

        manifest = build_artifact_manifest(
            run_directory=run_directory,
            artifacts={
                "resolved_config": "resolved_config.yaml",
                "metrics": "metrics.json",
                "predictions": "predictions.jsonl",
            },
            provenance={
                "experiment_id": "example-run",
                "stage": "part-01-verification",
            },
        )

        manifest_path = run_directory / "artifact_manifest.json"

        print("Artifact manifest created successfully.")
        print()
        print(f"Manifest path: {manifest_path}")
        print(
            f"Artifacts indexed: {len(manifest['artifacts'])}"
        )
        print(
            f"Missing artifacts: {len(manifest['missing_artifacts'])}"
        )
        print()

        for artifact in manifest["artifacts"]:
            print(
                f"{artifact['name']}: "
                f"type={artifact['type']}, "
                f"size={artifact['size_bytes']} bytes, "
                f"sha256={artifact['sha256'][:12]}..."
            )

    print()
    print("Temporary artifact files cleaned up automatically.")


if __name__ == "__main__":
    main()
