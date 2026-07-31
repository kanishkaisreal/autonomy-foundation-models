from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from common.checksums import compute_file_checksum
from drivelm_align.data.grouping import (
    DriveLMSceneGrouping,
    group_records_by_scene,
)
from drivelm_align.data.images import (
    DriveLMImagePathResolution,
    DriveLMImageValidationError,
    resolve_drivelm_image_paths,
    validate_drivelm_images,
)
from drivelm_align.data.index import (
    DriveLMSceneIndex,
    build_drivelm_scene_index,
)
from drivelm_align.data.objects import (
    DriveLMObjectTagExtraction,
    extract_drivelm_object_tags,
)
from drivelm_align.data.qa import (
    DriveLMQAExtraction,
    extract_drivelm_qa_records,
)
from drivelm_align.data.raw import (
    DriveLMAnnotations,
    load_drivelm_annotations,
)
from drivelm_align.data.split_types import (
    DriveLMRecordSplitAssignment,
    DriveLMSceneTokenSplit,
    DriveLMSplitPartition,
)
from drivelm_align.data.splits import (
    assign_records_to_split,
    load_split_manifest,
    split_scene_tokens,
    write_split_manifests,
)
from drivelm_align.data.statistics import compute_split_statistics
from drivelm_align.data.subsets import build_drivelm_local_subset
from drivelm_align.data.validation import (
    assert_frame_split_disjointness,
    assert_scene_split_disjointness,
)
from drivelm_align.visualization.scenes import (
    render_drivelm_multiview_scene,
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
    """Print record totals for one split partition."""
    print(
        f"  {partition.split_name:<10} "
        f"scenes={partition.scene_count:,}, "
        f"frames={partition.frame_count:,}, "
        f"images={partition.resolved_image_count:,}, "
        f"QA={partition.qa_count:,}, "
        f"objects={partition.parsed_object_count:,}"
    )


def _partition_scene_tokens(
    partition: DriveLMSplitPartition,
) -> tuple[str, ...]:
    return tuple(group.scene_token for group in partition.scene_groups)


def _partition_task_names(
    partition: DriveLMSplitPartition,
) -> set[str]:
    return {
        record.task_name
        for group in partition.scene_groups
        for record in group.qa_records
    }


def _verify_source_provenance(
    annotations: DriveLMAnnotations,
    scene_index: DriveLMSceneIndex,
    image_resolution: DriveLMImagePathResolution,
    qa_extraction: DriveLMQAExtraction,
    object_extraction: DriveLMObjectTagExtraction,
) -> None:
    """Verify source identifiers and extracted values end to end."""
    source_frame_to_scene = {
        frame_token: scene_token
        for scene_token, scene in annotations.scenes.items()
        for frame_token in scene["key_frames"]
    }
    assert set(annotations.scenes) == set(scene_index.scenes)
    assert set(source_frame_to_scene) == set(scene_index.frames)

    for record in image_resolution.resolved.values():
        source_reference = annotations.scenes[
            record.scene_token
        ]["key_frames"][record.frame_token]["image_paths"][
            record.camera_name
        ]
        assert record.source_reference == source_reference

    for record in image_resolution.unresolved:
        source_reference = annotations.scenes[
            record.scene_token
        ]["key_frames"][record.frame_token]["image_paths"][
            record.camera_name
        ]
        assert record.source_reference == str(source_reference)

    for record in qa_extraction.records:
        assert source_frame_to_scene[record.frame_token] == (
            record.scene_token
        )
        source_record = annotations.scenes[
            record.scene_token
        ]["key_frames"][record.frame_token]["QA"][
            record.task_name
        ][record.task_index]
        assert record.question == source_record["Q"]
        assert record.answer == source_record.get("A")
        if "A" not in source_record:
            expected_status = "missing"
        elif source_record["A"] is None:
            expected_status = "null"
        elif not source_record["A"].strip():
            expected_status = "empty"
        else:
            expected_status = "answered"
        assert record.answer_status == expected_status

    for record in object_extraction.records:
        source_metadata = annotations.scenes[
            record.scene_token
        ]["key_frames"][record.frame_token]["key_object_infos"][
            record.raw_tag
        ]
        assert record.category == source_metadata["Category"]
        assert record.status == source_metadata.get("Status")
        assert record.visual_description == source_metadata.get(
            "Visual_description"
        )
        assert record.bbox_xyxy == tuple(
            float(value) for value in source_metadata["2d_bbox"]
        )

    for record in object_extraction.rejected:
        source_metadata = annotations.scenes[
            record.scene_token
        ]["key_frames"][record.frame_token]["key_object_infos"][
            record.raw_tag
        ]
        assert record.raw_metadata == source_metadata


def _verify_leakage_detection(
    scene_split: DriveLMSceneTokenSplit,
    assignment: DriveLMRecordSplitAssignment,
) -> None:
    """Verify valid partitions and deliberate scene/record leaks."""
    assert_scene_split_disjointness(scene_split)
    assert_frame_split_disjointness(assignment)

    leaked_scene = scene_split.train_scene_tokens[0]
    corrupted_scene_split = replace(
        scene_split,
        validation_scene_tokens=(
            leaked_scene,
            *scene_split.validation_scene_tokens,
        ),
    )
    try:
        assert_scene_split_disjointness(corrupted_scene_split)
    except ValueError as exc:
        assert leaked_scene in str(exc)
    else:
        raise AssertionError("Injected scene leakage was not detected.")

    leaked_group = assignment.train.scene_groups[0]
    corrupted_assignment = replace(
        assignment,
        validation=replace(
            assignment.validation,
            scene_groups=(
                leaked_group,
                *assignment.validation.scene_groups,
            ),
        ),
    )
    try:
        assert_frame_split_disjointness(corrupted_assignment)
    except ValueError as exc:
        message = str(exc)
        assert "frame tokens" in message
        assert "image paths" in message
        assert "source aliases" in message
        assert leaked_group.frame_tokens[0] in message
        assert str(
            leaked_group.image_records[0].absolute_path.resolve()
        ) in message
        assert leaked_group.image_records[0].source_reference in message
    else:
        raise AssertionError("Injected record leakage was not detected.")

    print("Leakage validation:")
    print("  Valid scene/frame/path/alias partitions: PASS")
    print("  Injected scene leak detected:            PASS")
    print("  Injected frame/path/alias leak detected: PASS")


def _verify_statistics(
    assignment: DriveLMRecordSplitAssignment,
    temporary_root: Path,
) -> None:
    """Persist, reload, and plot the Function 024 statistics."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(temporary_root / "matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    statistics = compute_split_statistics(assignment)
    partitions = {
        "train": assignment.train,
        "validation": assignment.validation,
        "test": assignment.test,
    }

    required_distributions = {
        "tasks",
        "cameras",
        "object_categories",
        "object_statuses",
        "scene_descriptions",
        "prompt_length_words",
        "answer_length_words",
    }
    for split_name, partition in partitions.items():
        split_statistics = statistics[split_name]
        assert required_distributions <= split_statistics.keys()
        assert split_statistics["counts"]["scenes"] == (
            partition.scene_count
        )
        assert sum(split_statistics["tasks"].values()) == (
            partition.qa_count
        )
        assert sum(split_statistics["cameras"].values()) == (
            partition.resolved_image_count
        )
        assert sum(
            split_statistics["object_categories"].values()
        ) == partition.parsed_object_count
        assert sum(
            split_statistics["scene_descriptions"].values()
        ) == partition.scene_count

    report_path = temporary_root / "split_statistics.json"
    report_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == (
        statistics
    )

    split_names = list(statistics)
    qa_counts = [
        statistics[name]["counts"]["qa_records"]
        for name in split_names
    ]
    figure, axis = plt.subplots()
    axis.bar(split_names, qa_counts)
    axis.set_ylabel("QA records")
    axis.set_title("DriveLM QA records by split")
    figure.tight_layout()

    plot_path = temporary_root / "split_qa_comparison.png"
    figure.savefig(plot_path)
    plt.close(figure)
    with Image.open(plot_path) as plot_image:
        plot_image.verify()

    print("Split statistics:")
    for split_name in split_names:
        split_statistics = statistics[split_name]
        counts = split_statistics["counts"]
        prompt_lengths = split_statistics["prompt_length_words"]
        answer_lengths = split_statistics["answer_length_words"]
        print(
            f"  {split_name}: QA={counts['qa_records']:,}, "
            f"mean prompt={prompt_lengths['mean']}, "
            f"mean answer={answer_lengths['mean']}"
        )
    print("  Machine-readable report round trip: PASS")
    print("  Comparison plot create/reopen:      PASS")


def _verify_manifests(
    assignment: DriveLMRecordSplitAssignment,
    scene_split: DriveLMSceneTokenSplit,
    annotation_path: Path,
    temporary_root: Path,
) -> None:
    """Verify deterministic manifest persistence and rejection gates."""
    manifest_path, manifest_checksum = write_split_manifests(
        assignment=assignment,
        scene_split=scene_split,
        source_path=annotation_path,
        output_path=temporary_root / "split_manifest.json",
    )
    copy_path, copy_checksum = write_split_manifests(
        assignment=assignment,
        scene_split=scene_split,
        source_path=annotation_path,
        output_path=temporary_root / "split_manifest_copy.json",
    )
    assert manifest_checksum == copy_checksum
    assert manifest_path.read_bytes() == copy_path.read_bytes()

    manifest = load_split_manifest(
        manifest_path,
        source_path=annotation_path,
        expected_manifest_checksum=manifest_checksum,
    )
    assert manifest["source"]["sha256"] == compute_file_checksum(
        annotation_path
    )
    assert manifest["split_policy"] == {
        "unit": "scene_token",
        "seed": 42,
        "ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
    }
    assert {
        name: len(split_data["records"])
        for name, split_data in manifest["splits"].items()
    } == {"train": 487, "validation": 104, "test": 105}

    try:
        load_split_manifest(
            manifest_path,
            source_path=annotation_path,
            expected_manifest_checksum="0" * 64,
        )
    except ValueError:
        bad_manifest_checksum_rejected = True
    else:
        bad_manifest_checksum_rejected = False
    assert bad_manifest_checksum_rejected

    different_source_path = temporary_root / "different_source.json"
    different_source_path.write_text("{}\n", encoding="utf-8")
    try:
        load_split_manifest(
            manifest_path,
            source_path=different_source_path,
            expected_manifest_checksum=manifest_checksum,
        )
    except ValueError:
        incompatible_source_rejected = True
    else:
        incompatible_source_rejected = False
    assert incompatible_source_rejected

    print("Split manifests:")
    print("  Deterministic write and valid reload: PASS")
    print("  Bad manifest checksum rejected:      PASS")
    print("  Incompatible source rejected:        PASS")
    print(f"  SHA-256: {manifest_checksum}")


def _verify_subset(
    assignment: DriveLMRecordSplitAssignment,
) -> None:
    """Verify deterministic, diverse, scene-complete local subsets."""
    subset = build_drivelm_local_subset(
        assignment,
        train_scene_count=8,
        validation_scene_count=4,
        seed=42,
    )
    repeated = build_drivelm_local_subset(
        assignment,
        train_scene_count=8,
        validation_scene_count=4,
        seed=42,
    )

    for split_name, source_partition in (
        ("train", assignment.train),
        ("validation", assignment.validation),
    ):
        selected_tokens = set(
            _partition_scene_tokens(subset[split_name])
        )
        assert selected_tokens <= set(
            _partition_scene_tokens(source_partition)
        )
        assert _partition_scene_tokens(subset[split_name]) == (
            _partition_scene_tokens(repeated[split_name])
        )
        assert _partition_task_names(subset[split_name]) == (
            _partition_task_names(source_partition)
        )

    assert set(_partition_scene_tokens(subset["train"])).isdisjoint(
        _partition_scene_tokens(subset["validation"])
    )
    print("Local scene subset:")
    print("  Train/validation scenes: 8/4")
    print("  Deterministic complete-scene selection: PASS")
    print("  Separation and task diversity:          PASS")


def _verify_rendering(
    grouping: DriveLMSceneGrouping,
    temporary_root: Path,
) -> None:
    """Render and reopen one synchronized multiview audit figure."""
    group = next(
        candidate
        for candidate in grouping.groups.values()
        if len(candidate.image_records) >= 6
    )
    frame_token = group.frame_tokens[0]
    frame_images = [
        record
        for record in group.image_records
        if record.frame_token == frame_token
    ]
    frame_objects = [
        record
        for record in group.object_records
        if record.frame_token == frame_token
    ]
    assert len(frame_images) == 6

    figure_path = render_drivelm_multiview_scene(
        group,
        frame_token=frame_token,
        output_path=temporary_root / "multiview.png",
    )
    assert figure_path.stat().st_size > 0
    with Image.open(figure_path) as figure_image:
        figure_image.verify()

    print("Multiview rendering:")
    print(f"  Scene/frame: {group.scene_token}/{frame_token}")
    print(
        f"  Views/overlays: {len(frame_images)}/{len(frame_objects)}"
    )
    print("  Figure create/reopen: PASS")


def main() -> None:
    """Run every non-writing Part 2 completion gate using F5."""
    repository_root = Path(__file__).resolve().parents[1]
    annotation_path = (
        repository_root
        / "data/drivelm/QA_dataset_nus/v1_1_train_nus.json"
    )
    image_root = repository_root / "data/drivelm/nuscenes/samples"

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
            "DriveLM image validation failed: "
            f"{len(exc.report.issues):,} issues"
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
        grouping.groups,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    repeated_split = split_scene_tokens(
        reversed(tuple(grouping.groups)),
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    assert scene_split == repeated_split

    assignment = assign_records_to_split(
        grouping=grouping,
        scene_split=scene_split,
    )

    assert annotations.source_path == annotation_path.resolve()
    assert annotations.scenes == json.loads(
        annotation_path.read_text(encoding="utf-8")
    )
    _verify_source_provenance(
        annotations,
        scene_index,
        image_resolution,
        qa_extraction,
        object_extraction,
    )
    assert sum(qa_extraction.answer_status_counts.values()) == (
        qa_extraction.record_count
    )
    assert object_extraction.parsed_count + (
        object_extraction.rejected_count
    ) == object_extraction.source_object_count
    assert image_resolution.image_root == image_root.resolve()
    assert image_resolution.resolved_count + (
        image_resolution.unresolved_count
    ) == image_resolution.reference_count
    assert all(
        record.absolute_path.is_absolute()
        for record in image_resolution.resolved.values()
    )

    assert (
        annotations.scene_count,
        scene_index.frame_count,
        qa_extraction.record_count,
        image_resolution.resolved_count,
        image_resolution.unresolved_count,
    ) == (696, 4_072, 377_956, 24_432, 0)
    assert image_validation.passed
    assert image_validation.valid_image_count == 24_432
    assert not image_validation.issues
    assert grouping.scene_count == scene_index.scene_count
    assert grouping.frame_count == scene_index.frame_count
    assert grouping.resolved_image_count == (
        image_resolution.resolved_count
    )
    assert grouping.unresolved_image_count == (
        image_resolution.unresolved_count
    )
    assert grouping.qa_count == qa_extraction.record_count
    assert grouping.parsed_object_count == object_extraction.parsed_count
    assert grouping.rejected_object_count == (
        object_extraction.rejected_count
    )
    assert (
        scene_split.train_count,
        scene_split.validation_count,
        scene_split.test_count,
    ) == (487, 104, 105)
    assert assignment.total_scene_count == grouping.scene_count
    assert assignment.total_frame_count == grouping.frame_count
    assert assignment.total_resolved_image_count == (
        grouping.resolved_image_count
    )
    assert assignment.total_unresolved_image_count == (
        grouping.unresolved_image_count
    )
    assert assignment.total_qa_count == grouping.qa_count
    assert assignment.total_parsed_object_count == (
        grouping.parsed_object_count
    )
    assert assignment.total_rejected_object_count == (
        grouping.rejected_object_count
    )
    for partition in (
        assignment.train,
        assignment.validation,
        assignment.test,
    ):
        assert all(
            assignment.scene_to_split[group.scene_token]
            == partition.split_name
            for group in partition.scene_groups
        )

    print("DriveLM Part 2 integration verification")
    print("Dataset and provenance:")
    print(f"  Scenes/frames: {grouping.scene_count:,}/{grouping.frame_count:,}")
    print(
        f"  Resolved/unresolved images: "
        f"{grouping.resolved_image_count:,}/"
        f"{grouping.unresolved_image_count:,}"
    )
    print(f"  QA records: {grouping.qa_count:,}")
    print(
        f"  Parsed/rejected objects: "
        f"{grouping.parsed_object_count:,}/"
        f"{grouping.rejected_object_count:,}"
    )
    print(
        f"  Answer statuses: {qa_extraction.answer_status_counts}"
    )
    print("  Source identifiers and QA ownership: PASS")
    print("  Image resolution and full validation: PASS")

    print("Scene split and assignment:")
    print(
        f"  Seed 42 train/validation/test: "
        f"{scene_split.train_count}/"
        f"{scene_split.validation_count}/"
        f"{scene_split.test_count}"
    )
    print("  Same seed and reversed input reproduce lists: PASS")
    for partition in (
        assignment.train,
        assignment.validation,
        assignment.test,
    ):
        _print_partition(partition)

    _verify_leakage_detection(scene_split, assignment)

    with TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        _verify_statistics(assignment, temporary_root)
        _verify_manifests(
            assignment,
            scene_split,
            annotation_path,
            temporary_root,
        )
        _verify_rendering(grouping, temporary_root)

    _verify_subset(assignment)
    print("DriveLM Part 2 verification: PASS")


if __name__ == "__main__":
    main()
