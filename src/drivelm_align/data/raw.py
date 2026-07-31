from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DriveLMLoadError(ValueError):
    """Raised when DriveLM annotations cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class DriveLMAnnotations:
    """
    Raw DriveLM annotations and basic dataset counts.

    The original nested structure and source identifiers are preserved.
    """

    source_path: Path
    scenes: dict[str, Any]
    scene_count: int
    frame_count: int
    qa_count: int


def load_drivelm_annotations(
    annotation_path: str | Path,
) -> DriveLMAnnotations:
    """
    Load a DriveLM-nuScenes annotation JSON file.

    Args:
        annotation_path:
            Path to the DriveLM annotation file.

    Returns:
        Raw scene-indexed annotations and basic dataset counts.

    Raises:
        DriveLMLoadError:
            If the file is missing, malformed, or has an unexpected
            minimum structure.
    """
    source = Path(annotation_path).expanduser().resolve()

    if not source.is_file():
        raise DriveLMLoadError(
            "DriveLM annotation file does not exist:\n"
            f"  {source}\n"
            "Place v1_1_train_nus.json under "
            "data/drivelm/QA_dataset_nus/."
        )

    if source.suffix.lower() != ".json":
        raise DriveLMLoadError(
            f"DriveLM annotations must be a JSON file: {source}"
        )

    try:
        with source.open(
            mode="r",
            encoding="utf-8",
        ) as annotation_file:
            loaded_data = json.load(annotation_file)

    except json.JSONDecodeError as exc:
        raise DriveLMLoadError(
            f"Invalid JSON in {source} at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    except OSError as exc:
        raise DriveLMLoadError(
            f"Could not read DriveLM annotation file {source}: {exc}"
        ) from exc

    if not isinstance(loaded_data, dict):
        raise DriveLMLoadError(
            "The top level of the DriveLM annotation file must be "
            "a mapping from scene tokens to scene records."
        )

    scene_count = 0
    frame_count = 0
    qa_count = 0

    for scene_token, scene_data in loaded_data.items():
        if not isinstance(scene_token, str) or not scene_token.strip():
            raise DriveLMLoadError(
                f"Invalid scene token: {scene_token!r}"
            )

        if not isinstance(scene_data, dict):
            raise DriveLMLoadError(
                f"Scene {scene_token!r} must contain a JSON object."
            )

        key_frames = scene_data.get("key_frames")

        if not isinstance(key_frames, dict):
            raise DriveLMLoadError(
                f"Scene {scene_token!r} is missing a valid "
                "'key_frames' mapping."
            )

        scene_count += 1

        for frame_token, frame_data in key_frames.items():
            if not isinstance(frame_token, str) or not frame_token.strip():
                raise DriveLMLoadError(
                    f"Scene {scene_token!r} contains an invalid "
                    f"frame token: {frame_token!r}"
                )

            if not isinstance(frame_data, dict):
                raise DriveLMLoadError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} must contain a JSON object."
                )

            qa_tasks = frame_data.get("QA")

            if not isinstance(qa_tasks, dict):
                raise DriveLMLoadError(
                    f"Frame {frame_token!r} is missing a valid "
                    "'QA' task mapping."
                )

            frame_count += 1

            for task_name, task_records in qa_tasks.items():
                if not isinstance(task_records, list):
                    raise DriveLMLoadError(
                        f"QA task {task_name!r} must contain a list for "
                        f"scene={scene_token!r}, frame={frame_token!r}."
                    )

                for record_index, qa_record in enumerate(task_records):
                    if not isinstance(qa_record, dict):
                        raise DriveLMLoadError(
                            f"QA record {record_index} under task "
                            f"{task_name!r} must be a JSON object."
                        )

                qa_count += len(task_records)

    return DriveLMAnnotations(
        source_path=source,
        scenes=loaded_data,
        scene_count=scene_count,
        frame_count=frame_count,
        qa_count=qa_count,
    )


def main() -> None:
    """Load the local DriveLM annotations using F5."""
    repository_root = Path(__file__).resolve().parents[3]

    annotation_path = (
        repository_root
        / "data"
        / "drivelm"
        / "QA_dataset_nus"
        / "v1_1_train_nus.json"
    )

    annotations = load_drivelm_annotations(annotation_path)

    print("DriveLM annotations loaded successfully.")
    print()
    print(f"Source file: {annotations.source_path}")
    print(f"Scenes:      {annotations.scene_count:,}")
    print(f"Key frames:  {annotations.frame_count:,}")
    print(f"QA records:  {annotations.qa_count:,}")

    first_scene_token = next(iter(annotations.scenes), None)

    if first_scene_token is not None:
        first_scene = annotations.scenes[first_scene_token]

        print()
        print(f"First scene token: {first_scene_token}")
        print(
            "Scene description: "
            f"{first_scene.get('scene_description')}"
        )

        first_frame_token = next(
            iter(first_scene["key_frames"]),
            None,
        )

        if first_frame_token is not None:
            first_frame = first_scene["key_frames"][first_frame_token]

            print(f"First frame token: {first_frame_token}")
            print(
                "Available QA tasks: "
                f"{list(first_frame['QA'].keys())}"
            )


if __name__ == "__main__":
    main()
    
    