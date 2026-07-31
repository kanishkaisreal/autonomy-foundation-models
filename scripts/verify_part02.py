from __future__ import annotations

from pathlib import Path

from drivelm_align.data.grouping import group_records_by_scene
from drivelm_align.data.images import (
    DriveLMImageValidationError,
    resolve_drivelm_image_paths,
    validate_drivelm_images,
)
from drivelm_align.data.index import build_drivelm_scene_index
from drivelm_align.data.objects import extract_drivelm_object_tags
from drivelm_align.data.qa import extract_drivelm_qa_records
from drivelm_align.data.raw import load_drivelm_annotations
from drivelm_align.data.split_types import DriveLMSplitPartition
from drivelm_align.data.splits import (
    assign_records_to_split,
    split_scene_tokens,
)


REQUIRED_CAMERA_NAMES = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
)


def _print_partition(partition: DriveLMSplitPartition) -> None:
    """Print the preserved record totals for one split partition."""
    print()
    print(f"  Split:             {partition.split_name}")
    print(f"  Scenes:            {partition.scene_count:,}")
    print(f"  Frames:            {partition.frame_count:,}")
    print(f"  Resolved images:   {partition.resolved_image_count:,}")
    print(f"  Unresolved images: {partition.unresolved_image_count:,}")
    print(f"  QA records:        {partition.qa_count:,}")
    print(f"  Parsed objects:    {partition.parsed_object_count:,}")
    print(f"  Rejected objects:  {partition.rejected_object_count:,}")


def main() -> None:
    """Run the non-writing Part 2 integration verification using F5."""
    repository_root = Path(__file__).resolve().parents[1]
    annotation_path = (
        repository_root
        / "data"
        / "drivelm"
        / "QA_dataset_nus"
        / "v1_1_train_nus.json"
    )
    image_root = (
        repository_root
        / "data"
        / "drivelm"
        / "nuscenes"
        / "samples"
    )

    annotations = load_drivelm_annotations(annotation_path)
    scene_index = build_drivelm_scene_index(annotations)
    image_resolution = resolve_drivelm_image_paths(
        annotations=annotations,
        image_root=image_root,
    )

    try:
        image_validation = validate_drivelm_images(
            annotations=annotations,
            resolution=image_resolution,
            required_camera_names=REQUIRED_CAMERA_NAMES,
            expected_dimensions=(1600, 900),
            max_missing_fraction=0.0,
            fail_on_duplicate_paths=True,
        )
    except DriveLMImageValidationError as exc:
        print(
            "DriveLM image validation: FAIL "
            f"({len(exc.report.issues):,} issues)"
        )
        raise

    qa_extraction = extract_drivelm_qa_records(
        annotations,
        strict_answers=False,
    )
    object_extraction = extract_drivelm_object_tags(annotations)
    grouping = group_records_by_scene(
        scene_index=scene_index,
        image_resolution=image_resolution,
        qa_extraction=qa_extraction,
        object_extraction=object_extraction,
    )
    scene_split = split_scene_tokens(
        grouping.groups.keys(),
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    record_assignment = assign_records_to_split(
        grouping=grouping,
        scene_split=scene_split,
    )

    # This script targets the repository's pinned development dataset.
    assert annotations.scene_count == 696
    assert scene_index.frame_count == 4_072
    assert qa_extraction.record_count == 377_956
    assert image_resolution.resolved_count == 24_432
    assert image_resolution.unresolved_count == 0
    assert image_validation.valid_image_count == 24_432
    assert (
        scene_split.train_count,
        scene_split.validation_count,
        scene_split.test_count,
    ) == (487, 104, 105)

    print("DriveLM Part 2 integration verification")
    print()
    print(f"Scenes grouped:          {grouping.scene_count:,}")
    print(f"Frames grouped:          {grouping.frame_count:,}")
    print(
        f"Resolved images grouped: "
        f"{grouping.resolved_image_count:,}"
    )
    print(
        f"Unresolved images:       "
        f"{grouping.unresolved_image_count:,}"
    )
    print(f"QA records grouped:      {grouping.qa_count:,}")
    print(
        f"Parsed objects grouped:  "
        f"{grouping.parsed_object_count:,}"
    )
    print(
        f"Rejected objects grouped: "
        f"{grouping.rejected_object_count:,}"
    )

    print()
    print("Answer-status counts:")
    for answer_status, count in qa_extraction.answer_status_counts.items():
        print(f"  {answer_status}: {count:,}")

    print()
    print("Object-status counts:")
    for object_status, count in object_extraction.counts_by_status.items():
        print(f"  {object_status}: {count:,}")

    assert image_validation.passed
    assert image_validation.reference_count == (
        image_resolution.reference_count
    )
    print()
    print("Image verification:")
    print("  Every reference resolved: PASS")
    print("  Image validation:         PASS")

    first_scene_token = next(iter(grouping.groups))
    first_group = grouping.groups[first_scene_token]

    print()
    print("First scene group:")
    print(f"  Scene token:       {first_group.scene_token}")
    print(f"  Frames:            {first_group.frame_count}")
    print(f"  Images:            {first_group.image_count}")
    print(
        f"  Unresolved images: "
        f"{first_group.unresolved_image_count}"
    )
    print(f"  QA records:        {first_group.qa_count}")
    print(f"  Parsed objects:    {first_group.object_count}")
    print(
        f"  Rejected objects:  "
        f"{first_group.rejected_object_count}"
    )

    grouped_record_collections = (
        first_group.image_records,
        first_group.unresolved_images,
        first_group.qa_records,
        first_group.object_records,
        first_group.rejected_object_records,
    )
    assert all(
        record.scene_token == first_scene_token
        for records in grouped_record_collections
        for record in records
    )
    print()
    print("First-group ownership verification:")
    print("  Image ownership:  PASS")
    print("  QA ownership:     PASS")
    print("  Object ownership: PASS")

    print()
    print("DriveLM scene-token split:")
    print(f"  Seed:              {scene_split.seed}")
    print(f"  Training scenes:   {scene_split.train_count:,}")
    print(f"  Validation scenes: {scene_split.validation_count:,}")
    print(f"  Local-test scenes: {scene_split.test_count:,}")
    print(f"  Total scenes:      {scene_split.total_count:,}")

    repeated_split = split_scene_tokens(
        grouping.groups.keys(),
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    assert scene_split == repeated_split

    train_scenes = set(scene_split.train_scene_tokens)
    validation_scenes = set(scene_split.validation_scene_tokens)
    test_scenes = set(scene_split.test_scene_tokens)
    assert train_scenes.isdisjoint(validation_scenes)
    assert train_scenes.isdisjoint(test_scenes)
    assert validation_scenes.isdisjoint(test_scenes)
    assert train_scenes | validation_scenes | test_scenes == set(
        grouping.groups
    )

    print()
    print("Split verification:")
    print("  Same seed reproduces split: PASS")
    print("  No scene intersections:     PASS")
    print("  Every scene assigned:       PASS")

    print()
    print("First scene tokens:")
    print(f"  Train:      {scene_split.train_scene_tokens[0]}")
    print(
        f"  Validation: "
        f"{scene_split.validation_scene_tokens[0]}"
    )
    print(f"  Test:       {scene_split.test_scene_tokens[0]}")

    print()
    print("DriveLM records assigned to splits:")
    for partition in (
        record_assignment.train,
        record_assignment.validation,
        record_assignment.test,
    ):
        _print_partition(partition)

    assert record_assignment.total_scene_count == grouping.scene_count
    assert record_assignment.total_frame_count == grouping.frame_count
    assert record_assignment.total_resolved_image_count == (
        grouping.resolved_image_count
    )
    assert record_assignment.total_unresolved_image_count == (
        grouping.unresolved_image_count
    )
    assert record_assignment.total_qa_count == grouping.qa_count
    assert record_assignment.total_parsed_object_count == (
        grouping.parsed_object_count
    )
    assert record_assignment.total_rejected_object_count == (
        grouping.rejected_object_count
    )

    for partition in (
        record_assignment.train,
        record_assignment.validation,
        record_assignment.test,
    ):
        assert all(
            record_assignment.scene_to_split[group.scene_token]
            == partition.split_name
            for group in partition.scene_groups
        )

    print()
    print("Record-assignment verification:")
    print("  Every scene assigned once:   PASS")
    print("  Frame counts preserved:      PASS")
    print("  Image counts preserved:      PASS")
    print("  QA counts preserved:         PASS")
    print("  Object counts preserved:     PASS")
    print("  Records inherit scene split: PASS")


if __name__ == "__main__":
    main()
