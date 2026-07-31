from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drivelm_align.data.raw import (
    DriveLMAnnotations,
    load_drivelm_annotations,
)


class DriveLMIndexError(ValueError):
    """Raised when a reliable DriveLM scene/frame index cannot be built."""


@dataclass(frozen=True, slots=True)
class DriveLMFrameIndexEntry:
    """Metadata for one DriveLM key frame."""

    scene_token: str
    frame_token: str
    qa_count: int
    qa_counts_by_task: dict[str, int]
    object_count: int
    camera_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DriveLMSceneIndexEntry:
    """Metadata for one complete DriveLM scene."""

    scene_token: str
    scene_description: str
    frame_tokens: tuple[str, ...]
    frame_count: int
    qa_count: int
    object_count: int
    image_reference_count: int


@dataclass(frozen=True, slots=True)
class DriveLMSceneIndex:
    """
    Fast scene-level and frame-level lookup tables.

    The index contains metadata only. The original questions, answers,
    object records, and image paths remain in DriveLMAnnotations.
    """

    scenes: dict[str, DriveLMSceneIndexEntry]
    frames: dict[str, DriveLMFrameIndexEntry]
    scene_count: int
    frame_count: int
    qa_count: int


def build_drivelm_scene_index(
    annotations: DriveLMAnnotations,
) -> DriveLMSceneIndex:
    """
    Build deterministic scene and frame indexes from DriveLM annotations.

    Scene and frame tokens are sorted so that downstream processing does
    not depend on the ordering of keys in the source JSON file.

    Args:
        annotations:
            Raw DriveLM annotations returned by
            load_drivelm_annotations().

    Returns:
        Scene-level and frame-level metadata lookup tables.

    Raises:
        DriveLMIndexError:
            If required frame metadata is malformed, a frame token appears
            in more than one scene, or index totals disagree with the raw
            loader totals.
    """
    scene_entries: dict[str, DriveLMSceneIndexEntry] = {}
    frame_entries: dict[str, DriveLMFrameIndexEntry] = {}

    total_frame_count = 0
    total_qa_count = 0

    for scene_token in sorted(annotations.scenes):
        scene_data = annotations.scenes[scene_token]

        scene_description = scene_data.get("scene_description", "")

        if not isinstance(scene_description, str):
            raise DriveLMIndexError(
                f"'scene_description' must be a string for "
                f"scene={scene_token!r}."
            )

        key_frames = scene_data.get("key_frames")

        if not isinstance(key_frames, dict):
            raise DriveLMIndexError(
                f"Scene {scene_token!r} does not contain a valid "
                "'key_frames' mapping."
            )

        scene_frame_tokens: list[str] = []
        scene_qa_count = 0
        scene_object_count = 0
        scene_image_reference_count = 0

        for frame_token in sorted(key_frames):
            if frame_token in frame_entries:
                previous_scene = frame_entries[frame_token].scene_token

                raise DriveLMIndexError(
                    f"Frame token {frame_token!r} appears in more than "
                    f"one scene: {previous_scene!r} and {scene_token!r}."
                )

            frame_data = key_frames[frame_token]

            if not isinstance(frame_data, dict):
                raise DriveLMIndexError(
                    f"Frame {frame_token!r} in scene {scene_token!r} "
                    "must contain a JSON object."
                )

            qa_tasks = frame_data.get("QA")

            if not isinstance(qa_tasks, dict):
                raise DriveLMIndexError(
                    f"Frame {frame_token!r} does not contain a valid "
                    "'QA' mapping."
                )

            qa_counts_by_task: dict[str, int] = {}

            for task_name in sorted(qa_tasks):
                task_records = qa_tasks[task_name]

                if not isinstance(task_records, list):
                    raise DriveLMIndexError(
                        f"QA task {task_name!r} must contain a list for "
                        f"scene={scene_token!r}, frame={frame_token!r}."
                    )

                qa_counts_by_task[task_name] = len(task_records)

            frame_qa_count = sum(qa_counts_by_task.values())

            key_object_infos = frame_data.get(
                "key_object_infos",
                {},
            )

            if not isinstance(key_object_infos, dict):
                raise DriveLMIndexError(
                    f"'key_object_infos' must be a mapping for "
                    f"scene={scene_token!r}, frame={frame_token!r}."
                )

            image_paths = frame_data.get("image_paths", {})

            if not isinstance(image_paths, dict):
                raise DriveLMIndexError(
                    f"'image_paths' must be a mapping for "
                    f"scene={scene_token!r}, frame={frame_token!r}."
                )

            camera_names = tuple(sorted(image_paths))

            frame_entries[frame_token] = DriveLMFrameIndexEntry(
                scene_token=scene_token,
                frame_token=frame_token,
                qa_count=frame_qa_count,
                qa_counts_by_task=qa_counts_by_task,
                object_count=len(key_object_infos),
                camera_names=camera_names,
            )

            scene_frame_tokens.append(frame_token)
            scene_qa_count += frame_qa_count
            scene_object_count += len(key_object_infos)
            scene_image_reference_count += len(image_paths)

        frame_count = len(scene_frame_tokens)

        scene_entries[scene_token] = DriveLMSceneIndexEntry(
            scene_token=scene_token,
            scene_description=scene_description,
            frame_tokens=tuple(scene_frame_tokens),
            frame_count=frame_count,
            qa_count=scene_qa_count,
            object_count=scene_object_count,
            image_reference_count=scene_image_reference_count,
        )

        total_frame_count += frame_count
        total_qa_count += scene_qa_count

    if len(scene_entries) != annotations.scene_count:
        raise DriveLMIndexError(
            "Indexed scene count does not match the raw loader count: "
            f"index={len(scene_entries)}, "
            f"loader={annotations.scene_count}."
        )

    if total_frame_count != annotations.frame_count:
        raise DriveLMIndexError(
            "Indexed frame count does not match the raw loader count: "
            f"index={total_frame_count}, "
            f"loader={annotations.frame_count}."
        )

    if total_qa_count != annotations.qa_count:
        raise DriveLMIndexError(
            "Indexed QA count does not match the raw loader count: "
            f"index={total_qa_count}, "
            f"loader={annotations.qa_count}."
        )

    return DriveLMSceneIndex(
        scenes=scene_entries,
        frames=frame_entries,
        scene_count=len(scene_entries),
        frame_count=total_frame_count,
        qa_count=total_qa_count,
    )

def main() -> None:
    """Build and inspect the DriveLM scene/frame index using F5."""
    repository_root = Path(__file__).resolve().parents[3]

    annotation_path = (
        repository_root
        / "data"
        / "drivelm"
        / "QA_dataset_nus"
        / "v1_1_train_nus.json"
    )

    annotations = load_drivelm_annotations(annotation_path)
    index = build_drivelm_scene_index(annotations)

    print("DriveLM scene index built successfully.")
    print()
    print(f"Scenes indexed:     {index.scene_count:,}")
    print(f"Frames indexed:     {index.frame_count:,}")
    print(f"QA records indexed: {index.qa_count:,}")

    first_scene_token = next(iter(index.scenes), None)

    if first_scene_token is None:
        return

    scene_entry = index.scenes[first_scene_token]

    print()
    print(f"First scene token:  {scene_entry.scene_token}")
    print(f"Description:        {scene_entry.scene_description}")
    print(f"Frames in scene:    {scene_entry.frame_count}")
    print(f"QA in scene:        {scene_entry.qa_count}")
    print(f"Objects in scene:   {scene_entry.object_count}")
    print(
        f"Image references:   "
        f"{scene_entry.image_reference_count}"
    )

    first_frame_token = scene_entry.frame_tokens[0]
    frame_entry = index.frames[first_frame_token]

    print()
    print(f"First frame token:  {frame_entry.frame_token}")
    print(f"Owning scene:       {frame_entry.scene_token}")
    print(f"QA in frame:        {frame_entry.qa_count}")
    print(f"Objects in frame:   {frame_entry.object_count}")
    print(f"Cameras:            {list(frame_entry.camera_names)}")
    print("QA counts by task:")

    for task_name, task_count in (
        frame_entry.qa_counts_by_task.items()
    ):
        print(f"  {task_name}: {task_count}")


if __name__ == "__main__":
    main()