from __future__ import annotations

import math
import random
from collections.abc import Iterable

from drivelm_align.data.grouping import (
    DriveLMSceneGrouping,
    DriveLMSceneGroupingError,
    DriveLMSceneRecordGroup,
    group_records_by_scene,
)
from drivelm_align.data.split_types import (
    DriveLMRecordSplitAssignment,
    DriveLMSceneTokenSplit,
    DriveLMSplitPartition,
)

import json
from pathlib import Path

from common.checksums import compute_file_checksum


__all__ = [
    "DriveLMRecordSplitAssignment",
    "DriveLMSceneGrouping",
    "DriveLMSceneGroupingError",
    "DriveLMSceneRecordGroup",
    "DriveLMSceneTokenSplit",
    "DriveLMSplitPartition",
    "assign_records_to_split",
    "group_records_by_scene",
    "split_scene_tokens",
    'load_split_manifest',
]


def split_scene_tokens(
    scene_tokens: Iterable[str],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DriveLMSceneTokenSplit:
    """
    Partition unique scene tokens into deterministic dataset splits.

    Inputs are scene tokens, positive ratios summing to one, and an
    integer seed. The output covers every input exactly once and follows
    sort → seeded shuffle → partition → canonical partition sorting.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        for ratio in ratios
    ):
        raise ValueError("All split ratios must be numeric values.")

    if any(not math.isfinite(float(ratio)) for ratio in ratios):
        raise ValueError("All split ratios must be finite.")

    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    ratio_sum = sum(ratios)
    if not math.isclose(
        ratio_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Split ratios must sum to 1.0; "
            f"received {ratio_sum:.12f}."
        )

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    normalized_scene_tokens: list[str] = []
    for scene_token in scene_tokens:
        if (
            not isinstance(scene_token, str)
            or not scene_token.strip()
        ):
            raise ValueError(
                "Every scene token must be a non-empty string; "
                f"received {scene_token!r}."
            )
        normalized_scene_tokens.append(scene_token)

    if not normalized_scene_tokens:
        raise ValueError(
            "scene_tokens must contain at least one scene."
        )

    if len(set(normalized_scene_tokens)) != len(
        normalized_scene_tokens
    ):
        raise ValueError(
            "scene_tokens contains duplicate scene tokens."
        )

    if len(normalized_scene_tokens) < 3:
        raise ValueError(
            "At least three scenes are required to create "
            "train, validation, and test splits."
        )

    # Keep the core algorithm visible and independent of source ordering.
    shuffled_scene_tokens = sorted(normalized_scene_tokens)
    random.Random(seed).shuffle(shuffled_scene_tokens)

    total_scene_count = len(shuffled_scene_tokens)
    train_count = int(total_scene_count * train_ratio)
    validation_count = int(total_scene_count * validation_ratio)
    test_count = total_scene_count - train_count - validation_count

    if min(train_count, validation_count, test_count) == 0:
        raise ValueError(
            "The requested ratios produce an empty split: "
            f"train={train_count}, validation={validation_count}, "
            f"test={test_count}."
        )

    train_end = train_count
    validation_end = train_count + validation_count
    split = DriveLMSceneTokenSplit(
        train_scene_tokens=tuple(
            sorted(shuffled_scene_tokens[:train_end])
        ),
        validation_scene_tokens=tuple(
            sorted(
                shuffled_scene_tokens[train_end:validation_end]
            )
        ),
        test_scene_tokens=tuple(
            sorted(shuffled_scene_tokens[validation_end:])
        ),
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    if split.total_count != total_scene_count:
        raise RuntimeError(
            "Split counts do not reconcile with the input: "
            f"split={split.total_count}, input={total_scene_count}."
        )

    return split


def assign_records_to_split(
    *,
    grouping: DriveLMSceneGrouping,
    scene_split: DriveLMSceneTokenSplit,
) -> DriveLMRecordSplitAssignment:
    """
    Assign complete scene groups to their scene-token partitions.

    Inputs come from Functions 019 and 020. The returned record
    partitions preserve all grouped totals, and every record inherits
    exactly the split assigned to its parent scene.
    """
    grouped_scene_tokens = set(grouping.groups)
    assigned_scene_tokens = (
        set(scene_split.train_scene_tokens)
        | set(scene_split.validation_scene_tokens)
        | set(scene_split.test_scene_tokens)
    )

    unknown_scene_tokens = assigned_scene_tokens - grouped_scene_tokens
    if unknown_scene_tokens:
        raise DriveLMSceneGroupingError(
            "The split references scenes that are absent from "
            "Function 019: "
            f"{sorted(unknown_scene_tokens)!r}"
        )

    missing_scene_tokens = grouped_scene_tokens - assigned_scene_tokens
    if missing_scene_tokens:
        raise DriveLMSceneGroupingError(
            "Some grouped scenes were not assigned to any split: "
            f"{sorted(missing_scene_tokens)!r}"
        )

    def build_partition(
        split_name: str,
        scene_tokens: tuple[str, ...],
    ) -> DriveLMSplitPartition:
        return DriveLMSplitPartition(
            split_name=split_name,
            scene_groups=tuple(
                grouping.groups[scene_token]
                for scene_token in scene_tokens
            ),
        )

    assignment = DriveLMRecordSplitAssignment(
        train=build_partition(
            "train",
            scene_split.train_scene_tokens,
        ),
        validation=build_partition(
            "validation",
            scene_split.validation_scene_tokens,
        ),
        test=build_partition(
            "test",
            scene_split.test_scene_tokens,
        ),
        scene_to_split=scene_split.scene_to_split,
    )

    assigned_counts = {
        "scene": assignment.total_scene_count,
        "frame": assignment.total_frame_count,
        "resolved image": assignment.total_resolved_image_count,
        "unresolved image": assignment.total_unresolved_image_count,
        "QA": assignment.total_qa_count,
        "parsed object": assignment.total_parsed_object_count,
        "rejected object": assignment.total_rejected_object_count,
    }
    grouped_counts = {
        "scene": grouping.scene_count,
        "frame": grouping.frame_count,
        "resolved image": grouping.resolved_image_count,
        "unresolved image": grouping.unresolved_image_count,
        "QA": grouping.qa_count,
        "parsed object": grouping.parsed_object_count,
        "rejected object": grouping.rejected_object_count,
    }

    for record_type, assigned_count in assigned_counts.items():
        grouped_count = grouped_counts[record_type]
        if assigned_count != grouped_count:
            raise DriveLMSceneGroupingError(
                f"Assigned {record_type} count does not match "
                f"Function 019: assigned={assigned_count}, "
                f"grouped={grouped_count}."
            )

    return assignment


def write_split_manifests(
    *,
    assignment: DriveLMRecordSplitAssignment,
    scene_split: DriveLMSceneTokenSplit,
    source_path: Path,
    output_path: Path,
) -> tuple[Path, str]:
    """Write a deterministic DriveLM split manifest and return its checksum."""
    if assignment.scene_to_split != scene_split.scene_to_split:
        raise ValueError(
            "The record assignment does not match the scene split."
        )

    source_path = Path(source_path)
    output_path = Path(output_path)

    def partition_data(
        partition: DriveLMSplitPartition,
        scene_tokens: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "scene_tokens": list(scene_tokens),
            "counts": {
                "scenes": partition.scene_count,
                "frames": partition.frame_count,
                "resolved_images": partition.resolved_image_count,
                "unresolved_images": partition.unresolved_image_count,
                "qa_records": partition.qa_count,
                "parsed_objects": partition.parsed_object_count,
                "rejected_objects": partition.rejected_object_count,
            },
            "records": [
                {
                    "scene_token": group.scene_token,
                    "frame_tokens": list(group.frame_tokens),
                    "resolved_images": group.image_count,
                    "unresolved_images": group.unresolved_image_count,
                    "qa_records": group.qa_count,
                    "parsed_objects": group.object_count,
                    "rejected_objects": group.rejected_object_count,
                }
                for group in partition.scene_groups
            ],
        }

    manifest = {
        "schema_version": 1,
        "source": {
            "file_name": source_path.name,
            "sha256": compute_file_checksum(source_path),
        },
        "split_policy": {
            "unit": "scene_token",
            "seed": scene_split.seed,
            "ratios": {
                "train": scene_split.train_ratio,
                "validation": scene_split.validation_ratio,
                "test": scene_split.test_ratio,
            },
        },
        "splits": {
            "train": partition_data(
                assignment.train,
                scene_split.train_scene_tokens,
            ),
            "validation": partition_data(
                assignment.validation,
                scene_split.validation_scene_tokens,
            ),
            "test": partition_data(
                assignment.test,
                scene_split.test_scene_tokens,
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file prevents a partially written manifest.
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)

    return output_path, compute_file_checksum(output_path)


def load_split_manifest(
    manifest_path: Path,
    *,
    source_path: Path,
    expected_manifest_checksum: str | None = None,
) -> dict[str, object]:
    """Load a split manifest and verify its manifest and source checksums."""
    manifest_path = Path(manifest_path)
    source_path = Path(source_path)

    if expected_manifest_checksum is not None:
        actual_manifest_checksum = compute_file_checksum(
            manifest_path
        )

        if actual_manifest_checksum != expected_manifest_checksum:
            raise ValueError(
                "Split manifest checksum does not match the "
                "expected checksum."
            )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    recorded_source_checksum = manifest["source"]["sha256"]
    actual_source_checksum = compute_file_checksum(source_path)

    if recorded_source_checksum != actual_source_checksum:
        raise ValueError(
            "Split manifest was created from a different "
            "annotation source."
        )

    return manifest
