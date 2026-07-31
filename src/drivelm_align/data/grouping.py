from __future__ import annotations

from dataclasses import dataclass

from drivelm_align.data.images import (
    DriveLMImagePathResolution,
    ResolvedDriveLMImage,
    UnresolvedDriveLMImage,
)
from drivelm_align.data.index import DriveLMSceneIndex
from drivelm_align.data.objects import (
    DriveLMObjectTagExtraction,
    DriveLMObjectTagRecord,
    RejectedDriveLMObjectTag,
)
from drivelm_align.data.qa import (
    DriveLMQAExtraction,
    DriveLMQARecord,
)


class DriveLMSceneGroupingError(ValueError):
    """Raised when records cannot be grouped by scene reliably."""


@dataclass(frozen=True, slots=True)
class DriveLMSceneRecordGroup:
    """All indexed frames and extracted records belonging to one scene."""

    scene_token: str
    frame_tokens: tuple[str, ...]
    image_records: tuple[ResolvedDriveLMImage, ...]
    unresolved_images: tuple[UnresolvedDriveLMImage, ...]
    qa_records: tuple[DriveLMQARecord, ...]
    object_records: tuple[DriveLMObjectTagRecord, ...]
    rejected_object_records: tuple[RejectedDriveLMObjectTag, ...]

    @property
    def frame_count(self) -> int:
        """Return the number of key frames in this scene."""
        return len(self.frame_tokens)

    @property
    def image_count(self) -> int:
        """Return the number of resolved image records."""
        return len(self.image_records)

    @property
    def unresolved_image_count(self) -> int:
        """Return the number of unresolved image references."""
        return len(self.unresolved_images)

    @property
    def qa_count(self) -> int:
        """Return the number of QA records."""
        return len(self.qa_records)

    @property
    def object_count(self) -> int:
        """Return the number of parsed object records."""
        return len(self.object_records)

    @property
    def rejected_object_count(self) -> int:
        """Return the number of rejected object records."""
        return len(self.rejected_object_records)


@dataclass(frozen=True, slots=True)
class DriveLMSceneGrouping:
    """Complete deterministic collection of scene-grouped records."""

    groups: dict[str, DriveLMSceneRecordGroup]
    scene_count: int
    frame_count: int
    resolved_image_count: int
    unresolved_image_count: int
    qa_count: int
    parsed_object_count: int
    rejected_object_count: int


@dataclass(slots=True)
class _MutableSceneGroup:
    """Mutable record lists used only while constructing public groups."""

    frame_tokens: tuple[str, ...]
    image_records: list[ResolvedDriveLMImage]
    unresolved_images: list[UnresolvedDriveLMImage]
    qa_records: list[DriveLMQARecord]
    object_records: list[DriveLMObjectTagRecord]
    rejected_object_records: list[RejectedDriveLMObjectTag]


def _validate_record_location(
    *,
    scene_token: str,
    frame_token: str,
    scene_index: DriveLMSceneIndex,
    record_type: str,
) -> None:
    """Verify that a record's frame belongs to its declared scene."""
    if scene_token not in scene_index.scenes:
        raise DriveLMSceneGroupingError(
            f"{record_type} references unknown scene {scene_token!r}."
        )

    if frame_token not in scene_index.frames:
        raise DriveLMSceneGroupingError(
            f"{record_type} references unknown frame {frame_token!r}."
        )

    indexed_scene_token = scene_index.frames[frame_token].scene_token
    if indexed_scene_token != scene_token:
        raise DriveLMSceneGroupingError(
            f"{record_type} declares scene={scene_token!r} for "
            f"frame={frame_token!r}, but the scene index assigns "
            f"that frame to scene={indexed_scene_token!r}."
        )


def group_records_by_scene(
    *,
    scene_index: DriveLMSceneIndex,
    image_resolution: DriveLMImagePathResolution,
    qa_extraction: DriveLMQAExtraction,
    object_extraction: DriveLMObjectTagExtraction,
) -> DriveLMSceneGrouping:
    """
    Group image, QA, and object records under every indexed scene.

    Inputs are the outputs of Functions 014, 015, 017, and 018. The
    returned groups are deterministic, and every record must resolve to
    its declared indexed scene and frame without changing source totals.
    """
    mutable_groups = {
        scene_token: _MutableSceneGroup(
            frame_tokens=scene_index.scenes[scene_token].frame_tokens,
            image_records=[],
            unresolved_images=[],
            qa_records=[],
            object_records=[],
            rejected_object_records=[],
        )
        for scene_token in sorted(scene_index.scenes)
    }

    for reference_key in sorted(image_resolution.resolved):
        record = image_resolution.resolved[reference_key]
        _validate_record_location(
            scene_token=record.scene_token,
            frame_token=record.frame_token,
            scene_index=scene_index,
            record_type="Resolved image record",
        )
        mutable_groups[record.scene_token].image_records.append(record)

    for record in sorted(
        image_resolution.unresolved,
        key=lambda item: (
            item.scene_token,
            item.frame_token,
            item.camera_name,
        ),
    ):
        _validate_record_location(
            scene_token=record.scene_token,
            frame_token=record.frame_token,
            scene_index=scene_index,
            record_type="Unresolved image record",
        )
        mutable_groups[record.scene_token].unresolved_images.append(record)

    for record in qa_extraction.records:
        _validate_record_location(
            scene_token=record.scene_token,
            frame_token=record.frame_token,
            scene_index=scene_index,
            record_type="QA record",
        )
        mutable_groups[record.scene_token].qa_records.append(record)

    for record in object_extraction.records:
        _validate_record_location(
            scene_token=record.scene_token,
            frame_token=record.frame_token,
            scene_index=scene_index,
            record_type="Parsed object record",
        )
        mutable_groups[record.scene_token].object_records.append(record)

    # Rejected records remain assigned for auditability and split totals.
    for record in object_extraction.rejected:
        _validate_record_location(
            scene_token=record.scene_token,
            frame_token=record.frame_token,
            scene_index=scene_index,
            record_type="Rejected object record",
        )
        mutable_groups[
            record.scene_token
        ].rejected_object_records.append(record)

    groups: dict[str, DriveLMSceneRecordGroup] = {}
    for scene_token in sorted(mutable_groups):
        mutable = mutable_groups[scene_token]
        groups[scene_token] = DriveLMSceneRecordGroup(
            scene_token=scene_token,
            frame_tokens=mutable.frame_tokens,
            image_records=tuple(
                sorted(
                    mutable.image_records,
                    key=lambda item: (
                        item.frame_token,
                        item.camera_name,
                    ),
                )
            ),
            unresolved_images=tuple(
                sorted(
                    mutable.unresolved_images,
                    key=lambda item: (
                        item.frame_token,
                        item.camera_name,
                    ),
                )
            ),
            qa_records=tuple(
                sorted(
                    mutable.qa_records,
                    key=lambda item: (
                        item.frame_token,
                        item.task_name,
                        item.task_index,
                    ),
                )
            ),
            object_records=tuple(
                sorted(
                    mutable.object_records,
                    key=lambda item: (
                        item.frame_token,
                        item.raw_tag,
                    ),
                )
            ),
            rejected_object_records=tuple(
                sorted(
                    mutable.rejected_object_records,
                    key=lambda item: (
                        item.frame_token,
                        item.raw_tag,
                    ),
                )
            ),
        )

    grouped_counts = {
        "frame": sum(group.frame_count for group in groups.values()),
        "resolved image": sum(
            group.image_count for group in groups.values()
        ),
        "unresolved image": sum(
            group.unresolved_image_count for group in groups.values()
        ),
        "QA": sum(group.qa_count for group in groups.values()),
        "parsed object": sum(
            group.object_count for group in groups.values()
        ),
        "rejected object": sum(
            group.rejected_object_count for group in groups.values()
        ),
    }
    expected_counts = {
        "frame": scene_index.frame_count,
        "resolved image": image_resolution.resolved_count,
        "unresolved image": image_resolution.unresolved_count,
        "QA": qa_extraction.record_count,
        "parsed object": object_extraction.parsed_count,
        "rejected object": object_extraction.rejected_count,
    }

    if len(groups) != scene_index.scene_count:
        raise DriveLMSceneGroupingError(
            "Grouped scene count does not match Function 014: "
            f"grouped={len(groups)}, indexed={scene_index.scene_count}."
        )

    for record_type, grouped_count in grouped_counts.items():
        expected_count = expected_counts[record_type]
        if grouped_count != expected_count:
            raise DriveLMSceneGroupingError(
                f"Grouped {record_type} count does not match its source: "
                f"grouped={grouped_count}, source={expected_count}."
            )

    return DriveLMSceneGrouping(
        groups=groups,
        scene_count=len(groups),
        frame_count=grouped_counts["frame"],
        resolved_image_count=grouped_counts["resolved image"],
        unresolved_image_count=grouped_counts["unresolved image"],
        qa_count=grouped_counts["QA"],
        parsed_object_count=grouped_counts["parsed object"],
        rejected_object_count=grouped_counts["rejected object"],
    )


def main() -> None:
    """Group a few real scenes for local F5 debugging."""
    from drivelm_align.data._debug import (
        build_debug_grouping_inputs,
    )

    scene_index, images, qa, objects = build_debug_grouping_inputs(
        scene_count=3
    )
    grouping = group_records_by_scene(
        scene_index=scene_index,
        image_resolution=images,
        qa_extraction=qa,
        object_extraction=objects,
    )
    print(
        "DriveLM grouping: "
        f"scenes={grouping.scene_count}, "
        f"frames={grouping.frame_count}, "
        f"images={grouping.resolved_image_count}, "
        f"QA={grouping.qa_count:,}, "
        f"objects={grouping.parsed_object_count}"
    )


if __name__ == "__main__":
    main()
