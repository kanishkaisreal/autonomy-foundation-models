from __future__ import annotations

import random

from drivelm_align.data.grouping import (
    DriveLMSceneRecordGroup,
)
from drivelm_align.data.split_types import (
    DriveLMRecordSplitAssignment,
    DriveLMSplitPartition,
)


def _task_names(
    group: DriveLMSceneRecordGroup,
) -> set[str]:
    """Return the QA task names represented by one scene."""
    return {
        record.task_name
        for record in group.qa_records
    }


def _select_scene_groups(
    partition: DriveLMSplitPartition,
    *,
    scene_count: int,
    seed: int,
) -> DriveLMSplitPartition:
    """Select complete scenes while preferring uncovered QA tasks."""
    if not 1 <= scene_count <= partition.scene_count:
        raise ValueError(
            f"scene_count must be between 1 and "
            f"{partition.scene_count}."
        )

    remaining = list(partition.scene_groups)
    random.Random(seed).shuffle(remaining)

    selected: list[DriveLMSceneRecordGroup] = []
    covered_tasks: set[str] = set()

    while len(selected) < scene_count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: len(
                _task_names(remaining[index])
                - covered_tasks
            ),
        )

        group = remaining.pop(best_index)
        selected.append(group)
        covered_tasks.update(_task_names(group))

    return DriveLMSplitPartition(
        split_name=partition.split_name,
        scene_groups=tuple(
            sorted(
                selected,
                key=lambda group: group.scene_token,
            )
        ),
    )


def build_drivelm_local_subset(
    assignment: DriveLMRecordSplitAssignment,
    *,
    train_scene_count: int = 8,
    validation_scene_count: int = 4,
    seed: int = 42,
) -> dict[str, DriveLMSplitPartition]:
    """Build deterministic scene-level train and validation subsets."""
    return {
        "train": _select_scene_groups(
            assignment.train,
            scene_count=train_scene_count,
            seed=seed,
        ),
        "validation": _select_scene_groups(
            assignment.validation,
            scene_count=validation_scene_count,
            seed=seed + 1,
        ),
    }