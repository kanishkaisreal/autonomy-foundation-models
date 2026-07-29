from __future__ import annotations

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from common.artifacts import build_artifact_manifest
from common.checksums import compute_file_checksum
from common.config import load_project_config, validate_project_config
from common.device import resolve_compute_device
from common.io import load_jsonl_records, save_jsonl_records
from common.reproducibility import seed_everything
from common.runs import (
    create_experiment_id,
    create_run_directory,
    snapshot_run_config,
)


def main() -> None:
    """
    Run the complete Part 1 foundation workflow.

    All generated files are placed in a temporary directory and removed after
    the verification finishes.
    """
    repository_root = Path(__file__).resolve().parents[1]
    config_path = repository_root / "configs" / "base.yaml"

    print("PART 1 INTEGRATION CHECK")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 001–002: Load and validate configuration
    # ---------------------------------------------------------------
    config = load_project_config(config_path)

    validate_project_config(
        config,
        project_root=repository_root,
    )

    print("[PASS] Configuration loaded and validated")

    # ---------------------------------------------------------------
    # 003: Resolve the current compute device
    # ---------------------------------------------------------------
    device = resolve_compute_device(config.runtime.device)

    print(
        f"[PASS] Device resolved: "
        f"{device.requested} -> {device.resolved} "
        f"({device.device_name})"
    )

    # ---------------------------------------------------------------
    # 004: Confirm reproducible random sequences
    # ---------------------------------------------------------------
    seed_everything(config.project.seed)

    first_random_values = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(1).item()),
    }

    seed_everything(config.project.seed)

    second_random_values = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(1).item()),
    }

    assert first_random_values == second_random_values

    print("[PASS] Random-number generation is reproducible")

    # ---------------------------------------------------------------
    # 005–012: Build one complete temporary experiment run
    # ---------------------------------------------------------------
    with TemporaryDirectory(
        prefix="afm-part01-integration-"
    ) as temporary_output_root:
        experiment_id = create_experiment_id(
            project_name=config.project.name,
            experiment_name="part-01-integration",
        )

        run_directory = create_run_directory(
            output_root=temporary_output_root,
            experiment_id=experiment_id,
        )

        print(f"[PASS] Experiment ID created: {experiment_id}")
        print("[PASS] Isolated run directory created")

        # Function 008 calls function 005 internally to create
        # environment.json.
        snapshot_paths = snapshot_run_config(
            run_directory=run_directory,
            config=config,
            project_root=repository_root,
            command=[
                "python",
                "scripts/verify_part01.py",
            ],
            dataset_versions={
                "DriveLM": "not-downloaded-yet",
                "nuScenes": "not-downloaded-yet",
            },
        )

        assert all(
            path.is_file()
            for path in snapshot_paths.values()
        )

        print("[PASS] Configuration and environment snapshot created")

        # -----------------------------------------------------------
        # 009–010: Save and reload JSONL records
        # -----------------------------------------------------------
        sample_records = [
            {
                "scene_id": "scene-001",
                "question": "What is ahead of the ego vehicle?",
                "objects": ["car-1", "pedestrian-1"],
            },
            {
                "scene_id": "scene-002",
                "question": "Is the lead vehicle moving?",
                "objects": ["car-2"],
            },
        ]

        records_path = run_directory / "sample_records.jsonl"

        records_saved = save_jsonl_records(
            records=sample_records,
            output_path=records_path,
        )

        loaded_records = list(
            load_jsonl_records(records_path)
        )

        assert records_saved == len(sample_records)
        assert loaded_records == sample_records

        print(
            f"[PASS] JSONL round trip completed: "
            f"{records_saved} records"
        )

        # -----------------------------------------------------------
        # 011: Confirm checksum stability
        # -----------------------------------------------------------
        first_checksum = compute_file_checksum(records_path)
        second_checksum = compute_file_checksum(records_path)

        assert first_checksum == second_checksum

        print(
            f"[PASS] Stable SHA-256 checksum: "
            f"{first_checksum[:12]}..."
        )

        # Create one simple metrics artifact.
        metrics_path = run_directory / "metrics.json"

        metrics = {
            "records_saved": records_saved,
            "jsonl_round_trip_matches": loaded_records == sample_records,
            "reproducibility_matches": (
                first_random_values == second_random_values
            ),
            "resolved_device": device.resolved,
        }

        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # -----------------------------------------------------------
        # 012: Build the final artifact manifest
        # -----------------------------------------------------------
        manifest = build_artifact_manifest(
            run_directory=run_directory,
            artifacts={
                "resolved_config": snapshot_paths["resolved_config"],
                "environment": snapshot_paths["environment"],
                "run_snapshot": snapshot_paths["run_snapshot"],
                "sample_records": records_path,
                "metrics": metrics_path,
            },
            provenance={
                "project": config.project.name,
                "experiment_id": experiment_id,
                "stage": "part-01-integration",
            },
        )

        manifest_path = (
            run_directory
            / "artifact_manifest.json"
        )

        assert manifest_path.is_file()
        assert len(manifest["artifacts"]) == 5
        assert not manifest["missing_artifacts"]

        print(
            f"[PASS] Artifact manifest created: "
            f"{len(manifest['artifacts'])} artifacts"
        )

        print()
        print("Temporary run contents:")

        for path in sorted(run_directory.iterdir()):
            print(f"  - {path.name}")

    print()
    print("=" * 60)
    print("PART 1 INTEGRATION CHECK PASSED")
    print("Temporary integration artifacts cleaned up automatically.")


if __name__ == "__main__":
    main()