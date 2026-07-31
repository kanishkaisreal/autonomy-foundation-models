from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from drivelm_align.data.raw import (
    DriveLMAnnotations,
    load_drivelm_annotations,
)

if TYPE_CHECKING:
    from drivelm_align.data.grouping import (
        DriveLMSceneGrouping,
    )
    from drivelm_align.data.images import (
        DriveLMImagePathResolution,
    )
    from drivelm_align.data.index import DriveLMSceneIndex
    from drivelm_align.data.objects import (
        DriveLMObjectTagExtraction,
    )
    from drivelm_align.data.qa import DriveLMQAExtraction
    from drivelm_align.data.split_types import (
        DriveLMRecordSplitAssignment,
    )


def repository_root() -> Path:
    """Return the repository root for local F5 drivers."""
    return Path(__file__).resolve().parents[3]


def training_annotation_path() -> Path:
    """Return the checked-in location of the DriveLM training JSON."""
    return (
        repository_root()
        / "data/drivelm/QA_dataset_nus/v1_1_train_nus.json"
    )


def training_image_root() -> Path:
    """Return the local DriveLM/nuScenes training image root."""
    return repository_root() / "data/drivelm/nuscenes/samples"


def load_debug_annotations(
    scene_count: int = 20,
) -> DriveLMAnnotations:
    """Load a deterministic complete-scene slice for local debugging."""
    annotations = load_drivelm_annotations(training_annotation_path())
    if not 1 <= scene_count <= annotations.scene_count:
        raise ValueError(
            f"scene_count must be between 1 and "
            f"{annotations.scene_count}."
        )

    scene_tokens = sorted(annotations.scenes)[:scene_count]
    scenes = {
        scene_token: annotations.scenes[scene_token]
        for scene_token in scene_tokens
    }
    frame_count = sum(
        len(scene["key_frames"])
        for scene in scenes.values()
    )
    qa_count = sum(
        len(task_records)
        for scene in scenes.values()
        for frame in scene["key_frames"].values()
        for task_records in frame["QA"].values()
    )
    return DriveLMAnnotations(
        source_path=annotations.source_path,
        scenes=scenes,
        scene_count=len(scenes),
        frame_count=frame_count,
        qa_count=qa_count,
    )


def build_debug_grouping_inputs(
    scene_count: int = 4,
) -> tuple[
    DriveLMSceneIndex,
    DriveLMImagePathResolution,
    DriveLMQAExtraction,
    DriveLMObjectTagExtraction,
]:
    """Build the four inputs required by group_records_by_scene()."""
    from drivelm_align.data.images import resolve_drivelm_image_paths
    from drivelm_align.data.index import build_drivelm_scene_index
    from drivelm_align.data.objects import extract_drivelm_object_tags
    from drivelm_align.data.qa import extract_drivelm_qa_records

    annotations = load_debug_annotations(scene_count)
    return (
        build_drivelm_scene_index(annotations),
        resolve_drivelm_image_paths(
            annotations,
            training_image_root(),
        ),
        extract_drivelm_qa_records(annotations),
        extract_drivelm_object_tags(annotations),
    )


def build_debug_grouping(
    scene_count: int = 20,
) -> DriveLMSceneGrouping:
    """Build small real-data scene groups for downstream debug mains."""
    from drivelm_align.data.grouping import group_records_by_scene

    scene_index, images, qa, objects = build_debug_grouping_inputs(
        scene_count
    )
    return group_records_by_scene(
        scene_index=scene_index,
        image_resolution=images,
        qa_extraction=qa,
        object_extraction=objects,
    )


def build_debug_assignment(
    scene_count: int = 20,
) -> DriveLMRecordSplitAssignment:
    """Build a small deterministic assignment for downstream mains."""
    from drivelm_align.data.splits import (
        assign_records_to_split,
        split_scene_tokens,
    )

    grouping = build_debug_grouping(scene_count)
    scene_split = split_scene_tokens(grouping.groups)
    return assign_records_to_split(
        grouping=grouping,
        scene_split=scene_split,
    )
