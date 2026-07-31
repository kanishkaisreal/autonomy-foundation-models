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
    """Select deterministic, task-diverse complete-scene subsets."""
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


def main() -> None:
    """Select a small complete-scene subset using F5."""
    from drivelm_align.data._debug import build_debug_assignment

    subset = build_drivelm_local_subset(
        build_debug_assignment(),
        train_scene_count=4,
        validation_scene_count=2,
        seed=42,
    )
    for split_name, partition in subset.items():
        tasks = {
            record.task_name
            for group in partition.scene_groups
            for record in group.qa_records
        }
        print(
            f"{split_name}: scenes={partition.scene_count}, "
            f"tasks={sorted(tasks)}"
        )


if __name__ == "__main__":
    main()
