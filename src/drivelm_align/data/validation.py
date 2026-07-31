from __future__ import annotations

from pathlib import PurePosixPath

from drivelm_align.data.split_types import (
    DriveLMRecordSplitAssignment,
    DriveLMSceneTokenSplit,
    DriveLMSplitPartition,
)


def _cross_split_overlaps(
    values_by_split: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Return every pairwise overlap between the three splits."""
    return {
        "train/validation": (
            values_by_split["train"]
            & values_by_split["validation"]
        ),
        "train/test": (
            values_by_split["train"]
            & values_by_split["test"]
        ),
        "validation/test": (
            values_by_split["validation"]
            & values_by_split["test"]
        ),
    }


def assert_scene_split_disjointness(
    scene_split: DriveLMSceneTokenSplit,
) -> None:
    """Assert that no scene token occurs in multiple partitions."""
    overlaps = _cross_split_overlaps(
        {
            "train": set(scene_split.train_scene_tokens),
            "validation": set(
                scene_split.validation_scene_tokens
            ),
            "test": set(scene_split.test_scene_tokens),
        }
    )
    leaks = [
        f"{split_pair}: {sorted(shared_scenes)!r}"
        for split_pair, shared_scenes in overlaps.items()
        if shared_scenes
    ]
    if leaks:
        raise ValueError(
            "Scene leakage detected; " + "; ".join(leaks)
        )


def _partition_identities(
    partition: DriveLMSplitPartition,
) -> dict[str, set[str]]:
    """Collect frame and image identities from one partition."""
    frame_tokens = {
        frame_token
        for group in partition.scene_groups
        for frame_token in group.frame_tokens
    }
    physical_paths = {
        str(record.absolute_path.expanduser().resolve())
        for group in partition.scene_groups
        for record in group.image_records
    }
    source_aliases = {
        PurePosixPath(
            record.source_reference.replace("\\", "/")
        ).as_posix()
        for group in partition.scene_groups
        for records in (
            group.image_records,
            group.unresolved_images,
        )
        for record in records
    }
    return {
        "frame tokens": frame_tokens,
        "image paths": physical_paths,
        "source aliases": source_aliases,
    }


def assert_frame_split_disjointness(
    assignment: DriveLMRecordSplitAssignment,
) -> None:
    """Assert that frames and image identities do not cross splits."""
    identities_by_split = {
        "train": _partition_identities(assignment.train),
        "validation": _partition_identities(assignment.validation),
        "test": _partition_identities(assignment.test),
    }

    leaks: list[str] = []
    for identity_name in (
        "frame tokens",
        "image paths",
        "source aliases",
    ):
        overlaps = _cross_split_overlaps(
            {
                split_name: identities[identity_name]
                for split_name, identities
                in identities_by_split.items()
            }
        )
        leaks.extend(
            f"{identity_name} in {split_pair}: "
            f"{sorted(shared_values)!r}"
            for split_pair, shared_values in overlaps.items()
            if shared_values
        )

    if leaks:
        raise ValueError(
            "Cross-split record leakage detected; "
            + "; ".join(leaks)
        )


def main() -> None:
    """Run one valid scene-disjointness check for local F5 debugging."""
    from drivelm_align.data.splits import split_scene_tokens

    scene_split = split_scene_tokens(
        [f"scene-{index:02d}" for index in range(20)],
        seed=42,
    )
    assert_scene_split_disjointness(scene_split)
    print("Scene split disjointness: PASS")


if __name__ == "__main__":
    main()
