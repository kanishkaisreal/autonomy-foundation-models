from __future__ import annotations

from drivelm_align.data.split_types import DriveLMSceneTokenSplit
from drivelm_align.data.split_types import (
    DriveLMRecordSplitAssignment,
    DriveLMSceneTokenSplit,
)


def assert_scene_split_disjointness(
    scene_split: DriveLMSceneTokenSplit,
) -> None:
    """Fail if any scene token appears in more than one split."""
    train = set(scene_split.train_scene_tokens)
    validation = set(scene_split.validation_scene_tokens)
    test = set(scene_split.test_scene_tokens)

    overlaps = {
        "train/validation": train & validation,
        "train/test": train & test,
        "validation/test": validation & test,
    }

    for split_pair, shared_scenes in overlaps.items():
        if shared_scenes:
            raise ValueError(
                f"Scene leakage between {split_pair}: "
                f"{sorted(shared_scenes)[:5]}"
            )


def assert_frame_split_disjointness(
    assignment: DriveLMRecordSplitAssignment,
) -> None:
    """Fail if any frame token appears in more than one split."""
    train_frames = {
        frame_token
        for group in assignment.train.scene_groups
        for frame_token in group.frame_tokens
    }

    validation_frames = {
        frame_token
        for group in assignment.validation.scene_groups
        for frame_token in group.frame_tokens
    }

    test_frames = {
        frame_token
        for group in assignment.test.scene_groups
        for frame_token in group.frame_tokens
    }

    overlaps = {
        "train/validation": train_frames & validation_frames,
        "train/test": train_frames & test_frames,
        "validation/test": validation_frames & test_frames,
    }

    for split_pair, shared_frames in overlaps.items():
        if shared_frames:
            raise ValueError(
                f"Frame leakage between {split_pair}: "
                f"{sorted(shared_frames)[:5]}"
            )
            
            