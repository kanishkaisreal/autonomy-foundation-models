from __future__ import annotations
import math
import random
from collections.abc import Iterable



from dataclasses import dataclass
from pathlib import Path

from drivelm_align.data.images import (
    DriveLMImagePathResolution,
    ResolvedDriveLMImage,
    UnresolvedDriveLMImage,
    resolve_drivelm_image_paths,
)
from drivelm_align.data.index import (
    DriveLMSceneIndex,
    build_drivelm_scene_index,
)
from drivelm_align.data.objects import (
    DriveLMObjectTagExtraction,
    DriveLMObjectTagRecord,
    RejectedDriveLMObjectTag,
    extract_drivelm_object_tags,
)
from drivelm_align.data.qa import (
    DriveLMQAExtraction,
    DriveLMQARecord,
    extract_drivelm_qa_records,
)
from drivelm_align.data.raw import (
    load_drivelm_annotations,
)


class DriveLMSceneGroupingError(ValueError):
    """Raised when records cannot be grouped by scene reliably."""


@dataclass(frozen=True, slots=True)
class DriveLMSceneTokenSplit:
    """Deterministic train, validation, and test scene partitions."""

    train_scene_tokens: tuple[str, ...]
    validation_scene_tokens: tuple[str, ...]
    test_scene_tokens: tuple[str, ...]

    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    @property
    def train_count(self) -> int:
        """Return the number of training scenes."""
        return len(self.train_scene_tokens)

    @property
    def validation_count(self) -> int:
        """Return the number of validation scenes."""
        return len(self.validation_scene_tokens)

    @property
    def test_count(self) -> int:
        """Return the number of local-test scenes."""
        return len(self.test_scene_tokens)

    @property
    def total_count(self) -> int:
        """Return the total number of partitioned scenes."""
        return (
            self.train_count
            + self.validation_count
            + self.test_count
        )

    @property
    def scene_to_split(self) -> dict[str, str]:
        """Map every scene token to its assigned split."""
        assignments: dict[str, str] = {}

        for scene_token in self.train_scene_tokens:
            assignments[scene_token] = "train"

        for scene_token in self.validation_scene_tokens:
            assignments[scene_token] = "validation"

        for scene_token in self.test_scene_tokens:
            assignments[scene_token] = "test"

        return assignments
    
@dataclass(frozen=True, slots=True)
class DriveLMSplitPartition:
    """All scene groups assigned to one dataset split."""

    split_name: str
    scene_groups: tuple[
        DriveLMSceneRecordGroup,
        ...
    ]

    @property
    def scene_count(self) -> int:
        """Return the number of scenes in this partition."""
        return len(self.scene_groups)

    @property
    def frame_count(self) -> int:
        """Return the number of frames in this partition."""
        return sum(
            group.frame_count
            for group in self.scene_groups
        )

    @property
    def resolved_image_count(self) -> int:
        """Return the number of resolved images."""
        return sum(
            group.image_count
            for group in self.scene_groups
        )

    @property
    def unresolved_image_count(self) -> int:
        """Return the number of unresolved images."""
        return sum(
            group.unresolved_image_count
            for group in self.scene_groups
        )

    @property
    def qa_count(self) -> int:
        """Return the number of QA records."""
        return sum(
            group.qa_count
            for group in self.scene_groups
        )

    @property
    def parsed_object_count(self) -> int:
        """Return the number of parsed objects."""
        return sum(
            group.object_count
            for group in self.scene_groups
        )

    @property
    def rejected_object_count(self) -> int:
        """Return the number of rejected objects."""
        return sum(
            group.rejected_object_count
            for group in self.scene_groups
        )


@dataclass(frozen=True, slots=True)
class DriveLMRecordSplitAssignment:
    """Complete scene-inherited DriveLM record assignment."""

    train: DriveLMSplitPartition
    validation: DriveLMSplitPartition
    test: DriveLMSplitPartition

    scene_to_split: dict[str, str]

    @property
    def total_scene_count(self) -> int:
        """Return the total assigned scene count."""
        return (
            self.train.scene_count
            + self.validation.scene_count
            + self.test.scene_count
        )

    @property
    def total_frame_count(self) -> int:
        """Return the total assigned frame count."""
        return (
            self.train.frame_count
            + self.validation.frame_count
            + self.test.frame_count
        )

    @property
    def total_resolved_image_count(self) -> int:
        """Return the total assigned resolved-image count."""
        return (
            self.train.resolved_image_count
            + self.validation.resolved_image_count
            + self.test.resolved_image_count
        )

    @property
    def total_unresolved_image_count(self) -> int:
        """Return the total assigned unresolved-image count."""
        return (
            self.train.unresolved_image_count
            + self.validation.unresolved_image_count
            + self.test.unresolved_image_count
        )

    @property
    def total_qa_count(self) -> int:
        """Return the total assigned QA-record count."""
        return (
            self.train.qa_count
            + self.validation.qa_count
            + self.test.qa_count
        )

    @property
    def total_parsed_object_count(self) -> int:
        """Return the total assigned parsed-object count."""
        return (
            self.train.parsed_object_count
            + self.validation.parsed_object_count
            + self.test.parsed_object_count
        )

    @property
    def total_rejected_object_count(self) -> int:
        """Return the total assigned rejected-object count."""
        return (
            self.train.rejected_object_count
            + self.validation.rejected_object_count
            + self.test.rejected_object_count
        )
        
        
@dataclass(frozen=True, slots=True)
class DriveLMSplitPartition:
    """All scene groups assigned to one dataset split."""

    split_name: str
    scene_groups: tuple[
        DriveLMSceneRecordGroup,
        ...
    ]

    @property
    def scene_count(self) -> int:
        """Return the number of scenes in this partition."""
        return len(self.scene_groups)

    @property
    def frame_count(self) -> int:
        """Return the number of frames in this partition."""
        return sum(
            group.frame_count
            for group in self.scene_groups
        )

    @property
    def resolved_image_count(self) -> int:
        """Return the number of resolved images."""
        return sum(
            group.image_count
            for group in self.scene_groups
        )

    @property
    def unresolved_image_count(self) -> int:
        """Return the number of unresolved images."""
        return sum(
            group.unresolved_image_count
            for group in self.scene_groups
        )

    @property
    def qa_count(self) -> int:
        """Return the number of QA records."""
        return sum(
            group.qa_count
            for group in self.scene_groups
        )

    @property
    def parsed_object_count(self) -> int:
        """Return the number of parsed objects."""
        return sum(
            group.object_count
            for group in self.scene_groups
        )

    @property
    def rejected_object_count(self) -> int:
        """Return the number of rejected objects."""
        return sum(
            group.rejected_object_count
            for group in self.scene_groups
        )


@dataclass(frozen=True, slots=True)
class DriveLMRecordSplitAssignment:
    """Complete scene-inherited DriveLM record assignment."""

    train: DriveLMSplitPartition
    validation: DriveLMSplitPartition
    test: DriveLMSplitPartition

    scene_to_split: dict[str, str]

    @property
    def total_scene_count(self) -> int:
        """Return the total assigned scene count."""
        return (
            self.train.scene_count
            + self.validation.scene_count
            + self.test.scene_count
        )

    @property
    def total_frame_count(self) -> int:
        """Return the total assigned frame count."""
        return (
            self.train.frame_count
            + self.validation.frame_count
            + self.test.frame_count
        )

    @property
    def total_resolved_image_count(self) -> int:
        """Return the total assigned resolved-image count."""
        return (
            self.train.resolved_image_count
            + self.validation.resolved_image_count
            + self.test.resolved_image_count
        )

    @property
    def total_unresolved_image_count(self) -> int:
        """Return the total assigned unresolved-image count."""
        return (
            self.train.unresolved_image_count
            + self.validation.unresolved_image_count
            + self.test.unresolved_image_count
        )

    @property
    def total_qa_count(self) -> int:
        """Return the total assigned QA-record count."""
        return (
            self.train.qa_count
            + self.validation.qa_count
            + self.test.qa_count
        )

    @property
    def total_parsed_object_count(self) -> int:
        """Return the total assigned parsed-object count."""
        return (
            self.train.parsed_object_count
            + self.validation.parsed_object_count
            + self.test.parsed_object_count
        )

    @property
    def total_rejected_object_count(self) -> int:
        """Return the total assigned rejected-object count."""
        return (
            self.train.rejected_object_count
            + self.validation.rejected_object_count
            + self.test.rejected_object_count
        )
            
    
    
@dataclass(frozen=True, slots=True)
class DriveLMSceneRecordGroup:
    """All DriveLM records belonging to one scene."""

    scene_token: str
    frame_tokens: tuple[str, ...]

    image_records: tuple[
        ResolvedDriveLMImage,
        ...
    ]

    unresolved_images: tuple[
        UnresolvedDriveLMImage,
        ...
    ]

    qa_records: tuple[
        DriveLMQARecord,
        ...
    ]

    object_records: tuple[
        DriveLMObjectTagRecord,
        ...
    ]

    rejected_object_records: tuple[
        RejectedDriveLMObjectTag,
        ...
    ]

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
    """Complete collection of scene-grouped DriveLM records."""

    groups: dict[
        str,
        DriveLMSceneRecordGroup,
    ]

    scene_count: int
    frame_count: int

    resolved_image_count: int
    unresolved_image_count: int

    qa_count: int

    parsed_object_count: int
    rejected_object_count: int


@dataclass(slots=True)
class _MutableSceneGroup:
    """
    Internal mutable representation used while grouping records.

    This is converted into the frozen public representation before
    returning from group_records_by_scene().
    """

    frame_tokens: tuple[str, ...]
    image_records: list[ResolvedDriveLMImage]
    unresolved_images: list[UnresolvedDriveLMImage]
    qa_records: list[DriveLMQARecord]
    object_records: list[DriveLMObjectTagRecord]
    rejected_object_records: list[
        RejectedDriveLMObjectTag
    ]


def _validate_record_location(
    *,
    scene_token: str,
    frame_token: str,
    scene_index: DriveLMSceneIndex,
    record_type: str,
) -> None:
    """
    Verify that a record's frame belongs to its declared scene.

    This prevents an incorrectly labeled record from being silently
    grouped into the wrong scene.
    """
    if scene_token not in scene_index.scenes:
        raise DriveLMSceneGroupingError(
            f"{record_type} references unknown scene "
            f"{scene_token!r}."
        )

    if frame_token not in scene_index.frames:
        raise DriveLMSceneGroupingError(
            f"{record_type} references unknown frame "
            f"{frame_token!r}."
        )

    indexed_scene_token = (
        scene_index.frames[frame_token].scene_token
    )

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
    Group all DriveLM data records by their parent scene token.

    Every scene in the index receives one group, including scenes with no
    records of a particular type.

    The function validates that every frame belongs to the scene declared
    by each image, QA, or object record.

    Args:
        scene_index:
            Scene/frame lookup produced by Function 014.

        image_resolution:
            Resolved and unresolved image references from Function 015.

        qa_extraction:
            Flattened QA records from Function 017.

        object_extraction:
            Parsed and rejected object records from Function 018.

    Returns:
        Deterministic scene-level record groups.

    Raises:
        DriveLMSceneGroupingError:
            If a record references an unknown scene or frame, a frame is
            assigned to the wrong scene, or grouped counts disagree with
            the source-function totals.
    """
    mutable_groups: dict[
        str,
        _MutableSceneGroup,
    ] = {}

    # ---------------------------------------------------------
    # Stage 1: create exactly one group for every indexed scene.
    # ---------------------------------------------------------
    for scene_token in sorted(scene_index.scenes):
        scene_entry = scene_index.scenes[scene_token]

        mutable_groups[scene_token] = _MutableSceneGroup(
            frame_tokens=scene_entry.frame_tokens,
            image_records=[],
            unresolved_images=[],
            qa_records=[],
            object_records=[],
            rejected_object_records=[],
        )

    # ---------------------------------------------------------
    # Stage 2: group resolved image records.
    # ---------------------------------------------------------
    for reference_key in sorted(
        image_resolution.resolved
    ):
        image_record = image_resolution.resolved[
            reference_key
        ]

        _validate_record_location(
            scene_token=image_record.scene_token,
            frame_token=image_record.frame_token,
            scene_index=scene_index,
            record_type="Resolved image record",
        )

        mutable_groups[
            image_record.scene_token
        ].image_records.append(
            image_record
        )

    # ---------------------------------------------------------
    # Stage 3: group unresolved image references.
    # ---------------------------------------------------------
    for image_record in sorted(
        image_resolution.unresolved,
        key=lambda record: (
            record.scene_token,
            record.frame_token,
            record.camera_name,
        ),
    ):
        _validate_record_location(
            scene_token=image_record.scene_token,
            frame_token=image_record.frame_token,
            scene_index=scene_index,
            record_type="Unresolved image record",
        )

        mutable_groups[
            image_record.scene_token
        ].unresolved_images.append(
            image_record
        )

    # ---------------------------------------------------------
    # Stage 4: group QA records.
    # ---------------------------------------------------------
    for qa_record in qa_extraction.records:
        _validate_record_location(
            scene_token=qa_record.scene_token,
            frame_token=qa_record.frame_token,
            scene_index=scene_index,
            record_type="QA record",
        )

        mutable_groups[
            qa_record.scene_token
        ].qa_records.append(
            qa_record
        )

    # ---------------------------------------------------------
    # Stage 5: group successfully parsed object records.
    # ---------------------------------------------------------
    for object_record in object_extraction.records:
        _validate_record_location(
            scene_token=object_record.scene_token,
            frame_token=object_record.frame_token,
            scene_index=scene_index,
            record_type="Parsed object record",
        )

        mutable_groups[
            object_record.scene_token
        ].object_records.append(
            object_record
        )

    # ---------------------------------------------------------
    # Stage 6: group rejected object records as well.
    #
    # A rejected object remains part of the source dataset and must
    # retain its scene assignment for auditing and split manifests.
    # ---------------------------------------------------------
    for rejected_record in object_extraction.rejected:
        _validate_record_location(
            scene_token=rejected_record.scene_token,
            frame_token=rejected_record.frame_token,
            scene_index=scene_index,
            record_type="Rejected object record",
        )

        mutable_groups[
            rejected_record.scene_token
        ].rejected_object_records.append(
            rejected_record
        )

    # ---------------------------------------------------------
    # Stage 7: freeze the groups in deterministic order.
    # ---------------------------------------------------------
    frozen_groups: dict[
        str,
        DriveLMSceneRecordGroup,
    ] = {}

    for scene_token in sorted(mutable_groups):
        mutable_group = mutable_groups[scene_token]

        image_records = tuple(
            sorted(
                mutable_group.image_records,
                key=lambda record: (
                    record.frame_token,
                    record.camera_name,
                ),
            )
        )

        unresolved_images = tuple(
            sorted(
                mutable_group.unresolved_images,
                key=lambda record: (
                    record.frame_token,
                    record.camera_name,
                ),
            )
        )

        qa_records = tuple(
            sorted(
                mutable_group.qa_records,
                key=lambda record: (
                    record.frame_token,
                    record.task_name,
                    record.task_index,
                ),
            )
        )

        object_records = tuple(
            sorted(
                mutable_group.object_records,
                key=lambda record: (
                    record.frame_token,
                    record.raw_tag,
                ),
            )
        )

        rejected_object_records = tuple(
            sorted(
                mutable_group.rejected_object_records,
                key=lambda record: (
                    record.frame_token,
                    record.raw_tag,
                ),
            )
        )

        frozen_groups[scene_token] = (
            DriveLMSceneRecordGroup(
                scene_token=scene_token,
                frame_tokens=mutable_group.frame_tokens,
                image_records=image_records,
                unresolved_images=unresolved_images,
                qa_records=qa_records,
                object_records=object_records,
                rejected_object_records=(
                    rejected_object_records
                ),
            )
        )

    # ---------------------------------------------------------
    # Stage 8: reconcile grouped counts with source functions.
    # ---------------------------------------------------------
    grouped_frame_count = sum(
        group.frame_count
        for group in frozen_groups.values()
    )

    grouped_resolved_image_count = sum(
        group.image_count
        for group in frozen_groups.values()
    )

    grouped_unresolved_image_count = sum(
        group.unresolved_image_count
        for group in frozen_groups.values()
    )

    grouped_qa_count = sum(
        group.qa_count
        for group in frozen_groups.values()
    )

    grouped_parsed_object_count = sum(
        group.object_count
        for group in frozen_groups.values()
    )

    grouped_rejected_object_count = sum(
        group.rejected_object_count
        for group in frozen_groups.values()
    )

    if len(frozen_groups) != scene_index.scene_count:
        raise DriveLMSceneGroupingError(
            "Grouped scene count does not match Function 014: "
            f"grouped={len(frozen_groups)}, "
            f"indexed={scene_index.scene_count}."
        )

    if grouped_frame_count != scene_index.frame_count:
        raise DriveLMSceneGroupingError(
            "Grouped frame count does not match Function 014: "
            f"grouped={grouped_frame_count}, "
            f"indexed={scene_index.frame_count}."
        )

    if (
        grouped_resolved_image_count
        != image_resolution.resolved_count
    ):
        raise DriveLMSceneGroupingError(
            "Grouped resolved-image count does not match "
            "Function 015: "
            f"grouped={grouped_resolved_image_count}, "
            f"resolved={image_resolution.resolved_count}."
        )

    if (
        grouped_unresolved_image_count
        != image_resolution.unresolved_count
    ):
        raise DriveLMSceneGroupingError(
            "Grouped unresolved-image count does not match "
            "Function 015: "
            f"grouped={grouped_unresolved_image_count}, "
            f"unresolved={image_resolution.unresolved_count}."
        )

    if grouped_qa_count != qa_extraction.record_count:
        raise DriveLMSceneGroupingError(
            "Grouped QA count does not match Function 017: "
            f"grouped={grouped_qa_count}, "
            f"extracted={qa_extraction.record_count}."
        )

    if (
        grouped_parsed_object_count
        != object_extraction.parsed_count
    ):
        raise DriveLMSceneGroupingError(
            "Grouped parsed-object count does not match "
            "Function 018: "
            f"grouped={grouped_parsed_object_count}, "
            f"parsed={object_extraction.parsed_count}."
        )

    if (
        grouped_rejected_object_count
        != object_extraction.rejected_count
    ):
        raise DriveLMSceneGroupingError(
            "Grouped rejected-object count does not match "
            "Function 018: "
            f"grouped={grouped_rejected_object_count}, "
            f"rejected={object_extraction.rejected_count}."
        )

    return DriveLMSceneGrouping(
        groups=frozen_groups,
        scene_count=len(frozen_groups),
        frame_count=grouped_frame_count,
        resolved_image_count=(
            grouped_resolved_image_count
        ),
        unresolved_image_count=(
            grouped_unresolved_image_count
        ),
        qa_count=grouped_qa_count,
        parsed_object_count=(
            grouped_parsed_object_count
        ),
        rejected_object_count=(
            grouped_rejected_object_count
        ),
    )

def split_scene_tokens(
    scene_tokens: Iterable[str],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DriveLMSceneTokenSplit:
    """
    Split DriveLM scene tokens deterministically.

    The source tokens are sorted before shuffling so the result does not
    depend on dictionary, set, filesystem, or JSON key ordering.

    The train and validation counts use integer truncation. The test split
    receives all remaining scenes so that no scene is lost because of
    ratio rounding.

    Args:
        scene_tokens:
            Unique DriveLM scene tokens.

        train_ratio:
            Fraction of scenes assigned to training.

        validation_ratio:
            Fraction assigned to validation and model selection.

        test_ratio:
            Fraction assigned to the immutable local test set.

        seed:
            Seed used by an isolated Python random-number generator.

    Returns:
        Deterministic and scene-disjoint token partitions.

    Raises:
        ValueError:
            If tokens or split configuration are invalid.
    """
    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if any(
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        for ratio in ratios
    ):
        raise ValueError(
            "All split ratios must be numeric values."
        )

    if any(
        not math.isfinite(float(ratio))
        for ratio in ratios
    ):
        raise ValueError(
            "All split ratios must be finite."
        )

    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(
            "All split ratios must be greater than zero."
        )

    ratio_sum = sum(ratios)

    if not math.isclose(
        ratio_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Split ratios must sum to 1.0; "
            f"received {ratio_sum:.12f}."
        )

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "seed must be an integer."
        )

    normalized_scene_tokens: list[str] = []

    for scene_token in scene_tokens:
        if (
            not isinstance(scene_token, str)
            or not scene_token.strip()
        ):
            raise ValueError(
                "Every scene token must be a non-empty string; "
                f"received {scene_token!r}."
            )

        normalized_scene_tokens.append(scene_token)

    if not normalized_scene_tokens:
        raise ValueError(
            "scene_tokens must contain at least one scene."
        )

    unique_scene_tokens = set(normalized_scene_tokens)

    if (
        len(unique_scene_tokens)
        != len(normalized_scene_tokens)
    ):
        raise ValueError(
            "scene_tokens contains duplicate scene tokens."
        )

    if len(normalized_scene_tokens) < 3:
        raise ValueError(
            "At least three scenes are required to create "
            "train, validation, and test splits."
        )

    shuffled_scene_tokens = sorted(
        normalized_scene_tokens
    )

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_scene_tokens)

    total_scene_count = len(shuffled_scene_tokens)

    train_count = int(
        total_scene_count * train_ratio
    )

    validation_count = int(
        total_scene_count * validation_ratio
    )

    test_count = (
        total_scene_count
        - train_count
        - validation_count
    )

    if min(
        train_count,
        validation_count,
        test_count,
    ) == 0:
        raise ValueError(
            "The requested ratios produce an empty split: "
            f"train={train_count}, "
            f"validation={validation_count}, "
            f"test={test_count}."
        )

    train_end = train_count
    validation_end = (
        train_count + validation_count
    )

    # Sort each completed partition so serialized manifests have a
    # stable canonical order rather than random-generator order.
    train_scene_tokens = tuple(
        sorted(
            shuffled_scene_tokens[:train_end]
        )
    )

    validation_scene_tokens = tuple(
        sorted(
            shuffled_scene_tokens[
                train_end:validation_end
            ]
        )
    )

    test_scene_tokens = tuple(
        sorted(
            shuffled_scene_tokens[
                validation_end:
            ]
        )
    )

    split = DriveLMSceneTokenSplit(
        train_scene_tokens=train_scene_tokens,
        validation_scene_tokens=(
            validation_scene_tokens
        ),
        test_scene_tokens=test_scene_tokens,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    if split.total_count != total_scene_count:
        raise RuntimeError(
            "Split counts do not reconcile with the input: "
            f"split={split.total_count}, "
            f"input={total_scene_count}."
        )

    return split


def assign_records_to_split(
    *,
    grouping: DriveLMSceneGrouping,
    scene_split: DriveLMSceneTokenSplit,
) -> DriveLMRecordSplitAssignment:
    """
    Assign complete DriveLM scene groups to dataset splits.

    Every frame, image, QA record, parsed object, rejected object, and
    unresolved image inherits the split of its parent scene token.

    Args:
        grouping:
            Complete scene-level record groups produced by Function 019.

        scene_split:
            Scene-token partitions produced by Function 020.

    Returns:
        Train, validation, and local-test record partitions.

    Raises:
        DriveLMSceneGroupingError:
            If split tokens do not exactly match the grouped scenes or
            assigned record counts fail to reconcile with Function 019.
    """
    grouped_scene_tokens = set(grouping.groups)

    train_scene_tokens = set(
        scene_split.train_scene_tokens
    )

    validation_scene_tokens = set(
        scene_split.validation_scene_tokens
    )

    test_scene_tokens = set(
        scene_split.test_scene_tokens
    )

    assigned_scene_tokens = (
        train_scene_tokens
        | validation_scene_tokens
        | test_scene_tokens
    )

    unknown_scene_tokens = (
        assigned_scene_tokens
        - grouped_scene_tokens
    )

    missing_scene_tokens = (
        grouped_scene_tokens
        - assigned_scene_tokens
    )

    if unknown_scene_tokens:
        raise DriveLMSceneGroupingError(
            "The split references scenes that are absent from "
            "Function 019: "
            f"{sorted(unknown_scene_tokens)!r}"
        )

    if missing_scene_tokens:
        raise DriveLMSceneGroupingError(
            "Some grouped scenes were not assigned to any split: "
            f"{sorted(missing_scene_tokens)!r}"
        )

    train_partition = DriveLMSplitPartition(
        split_name="train",
        scene_groups=tuple(
            grouping.groups[scene_token]
            for scene_token
            in scene_split.train_scene_tokens
        ),
    )

    validation_partition = DriveLMSplitPartition(
        split_name="validation",
        scene_groups=tuple(
            grouping.groups[scene_token]
            for scene_token
            in scene_split.validation_scene_tokens
        ),
    )

    test_partition = DriveLMSplitPartition(
        split_name="test",
        scene_groups=tuple(
            grouping.groups[scene_token]
            for scene_token
            in scene_split.test_scene_tokens
        ),
    )

    assignment = DriveLMRecordSplitAssignment(
        train=train_partition,
        validation=validation_partition,
        test=test_partition,
        scene_to_split=scene_split.scene_to_split,
    )

    if assignment.total_scene_count != grouping.scene_count:
        raise DriveLMSceneGroupingError(
            "Assigned scene count does not match Function 019: "
            f"assigned={assignment.total_scene_count}, "
            f"grouped={grouping.scene_count}."
        )

    if assignment.total_frame_count != grouping.frame_count:
        raise DriveLMSceneGroupingError(
            "Assigned frame count does not match Function 019: "
            f"assigned={assignment.total_frame_count}, "
            f"grouped={grouping.frame_count}."
        )

    if (
        assignment.total_resolved_image_count
        != grouping.resolved_image_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned resolved-image count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_resolved_image_count}, "
            f"grouped={grouping.resolved_image_count}."
        )

    if (
        assignment.total_unresolved_image_count
        != grouping.unresolved_image_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned unresolved-image count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_unresolved_image_count}, "
            f"grouped={grouping.unresolved_image_count}."
        )

    if assignment.total_qa_count != grouping.qa_count:
        raise DriveLMSceneGroupingError(
            "Assigned QA count does not match Function 019: "
            f"assigned={assignment.total_qa_count}, "
            f"grouped={grouping.qa_count}."
        )

    if (
        assignment.total_parsed_object_count
        != grouping.parsed_object_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned parsed-object count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_parsed_object_count}, "
            f"grouped={grouping.parsed_object_count}."
        )

    if (
        assignment.total_rejected_object_count
        != grouping.rejected_object_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned rejected-object count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_rejected_object_count}, "
            f"grouped={grouping.rejected_object_count}."
        )

    return assignment



def assign_records_to_split(
    *,
    grouping: DriveLMSceneGrouping,
    scene_split: DriveLMSceneTokenSplit,
) -> DriveLMRecordSplitAssignment:
    """
    Assign complete DriveLM scene groups to dataset splits.

    Every frame, image, QA record, parsed object, rejected object, and
    unresolved image inherits the split of its parent scene token.

    Args:
        grouping:
            Complete scene-level record groups produced by Function 019.

        scene_split:
            Scene-token partitions produced by Function 020.

    Returns:
        Train, validation, and local-test record partitions.

    Raises:
        DriveLMSceneGroupingError:
            If split tokens do not exactly match the grouped scenes or
            assigned record counts fail to reconcile with Function 019.
    """
    grouped_scene_tokens = set(grouping.groups)

    train_scene_tokens = set(
        scene_split.train_scene_tokens
    )

    validation_scene_tokens = set(
        scene_split.validation_scene_tokens
    )

    test_scene_tokens = set(
        scene_split.test_scene_tokens
    )

    assigned_scene_tokens = (
        train_scene_tokens
        | validation_scene_tokens
        | test_scene_tokens
    )

    unknown_scene_tokens = (
        assigned_scene_tokens
        - grouped_scene_tokens
    )

    missing_scene_tokens = (
        grouped_scene_tokens
        - assigned_scene_tokens
    )

    if unknown_scene_tokens:
        raise DriveLMSceneGroupingError(
            "The split references scenes that are absent from "
            "Function 019: "
            f"{sorted(unknown_scene_tokens)!r}"
        )

    if missing_scene_tokens:
        raise DriveLMSceneGroupingError(
            "Some grouped scenes were not assigned to any split: "
            f"{sorted(missing_scene_tokens)!r}"
        )

    train_partition = DriveLMSplitPartition(
        split_name="train",
        scene_groups=tuple(
            grouping.groups[scene_token]
            for scene_token
            in scene_split.train_scene_tokens
        ),
    )

    validation_partition = DriveLMSplitPartition(
        split_name="validation",
        scene_groups=tuple(
            grouping.groups[scene_token]
            for scene_token
            in scene_split.validation_scene_tokens
        ),
    )

    test_partition = DriveLMSplitPartition(
        split_name="test",
        scene_groups=tuple(
            grouping.groups[scene_token]
            for scene_token
            in scene_split.test_scene_tokens
        ),
    )

    assignment = DriveLMRecordSplitAssignment(
        train=train_partition,
        validation=validation_partition,
        test=test_partition,
        scene_to_split=scene_split.scene_to_split,
    )

    if assignment.total_scene_count != grouping.scene_count:
        raise DriveLMSceneGroupingError(
            "Assigned scene count does not match Function 019: "
            f"assigned={assignment.total_scene_count}, "
            f"grouped={grouping.scene_count}."
        )

    if assignment.total_frame_count != grouping.frame_count:
        raise DriveLMSceneGroupingError(
            "Assigned frame count does not match Function 019: "
            f"assigned={assignment.total_frame_count}, "
            f"grouped={grouping.frame_count}."
        )

    if (
        assignment.total_resolved_image_count
        != grouping.resolved_image_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned resolved-image count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_resolved_image_count}, "
            f"grouped={grouping.resolved_image_count}."
        )

    if (
        assignment.total_unresolved_image_count
        != grouping.unresolved_image_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned unresolved-image count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_unresolved_image_count}, "
            f"grouped={grouping.unresolved_image_count}."
        )

    if assignment.total_qa_count != grouping.qa_count:
        raise DriveLMSceneGroupingError(
            "Assigned QA count does not match Function 019: "
            f"assigned={assignment.total_qa_count}, "
            f"grouped={grouping.qa_count}."
        )

    if (
        assignment.total_parsed_object_count
        != grouping.parsed_object_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned parsed-object count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_parsed_object_count}, "
            f"grouped={grouping.parsed_object_count}."
        )

    if (
        assignment.total_rejected_object_count
        != grouping.rejected_object_count
    ):
        raise DriveLMSceneGroupingError(
            "Assigned rejected-object count does not match "
            "Function 019: "
            f"assigned="
            f"{assignment.total_rejected_object_count}, "
            f"grouped={grouping.rejected_object_count}."
        )

    return assignment


def main() -> None:
    """Build and inspect scene-level DriveLM groups using F5."""
    repository_root = Path(__file__).resolve().parents[3]

    annotation_path = (
        repository_root
        / "data"
        / "drivelm"
        / "QA_dataset_nus"
        / "v1_1_train_nus.json"
    )

    training_image_root = (
        repository_root
        / "data"
        / "drivelm"
        / "nuscenes"
        / "samples"
    )

    annotations = load_drivelm_annotations(
        annotation_path
    )

    scene_index = build_drivelm_scene_index(
        annotations
    )

    image_resolution = resolve_drivelm_image_paths(
        annotations=annotations,
        image_root=training_image_root,
    )

    qa_extraction = extract_drivelm_qa_records(
        annotations,
        strict_answers=False,
    )

    object_extraction = extract_drivelm_object_tags(
        annotations
    )

    grouping = group_records_by_scene(
        scene_index=scene_index,
        image_resolution=image_resolution,
        qa_extraction=qa_extraction,
        object_extraction=object_extraction,
    )

    print("DriveLM records grouped by scene successfully.")
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
        f"Rejected objects grouped:"
        f" {grouping.rejected_object_count:,}"
    )

    first_scene_token = next(
        iter(grouping.groups),
        None,
    )

    if first_scene_token is None:
        return

    first_group = grouping.groups[
        first_scene_token
    ]

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

    # ---------------------------------------------------------
    # Verify that every grouped record belongs to this scene.
    # ---------------------------------------------------------
    assert all(
        record.scene_token == first_scene_token
        for record in first_group.image_records
    )

    assert all(
        record.scene_token == first_scene_token
        for record in first_group.qa_records
    )

    assert all(
        record.scene_token == first_scene_token
        for record in first_group.object_records
    )

    print()
    print("First-group ownership verification:")
    print("  Image ownership:  PASS")
    print("  QA ownership:     PASS")
    print("  Object ownership: PASS")

    scene_split = split_scene_tokens(
        grouping.groups.keys(),
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print()
    print("DriveLM scene-token split:")
    print(f"  Seed:              {scene_split.seed}")
    print(
        f"  Training scenes:   "
        f"{scene_split.train_count:,}"
    )
    print(
        f"  Validation scenes: "
        f"{scene_split.validation_count:,}"
    )
    print(
        f"  Local-test scenes: "
        f"{scene_split.test_count:,}"
    )
    print(
        f"  Total scenes:      "
        f"{scene_split.total_count:,}"
    )

    # ---------------------------------------------------------
    # Repeat the function to prove deterministic membership.
    # ---------------------------------------------------------
    repeated_split = split_scene_tokens(
        grouping.groups.keys(),
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert scene_split == repeated_split

    train_set = set(
        scene_split.train_scene_tokens
    )

    validation_set = set(
        scene_split.validation_scene_tokens
    )

    test_set = set(
        scene_split.test_scene_tokens
    )

    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)

    all_split_scenes = (
        train_set
        | validation_set
        | test_set
    )

    assert all_split_scenes == set(
        grouping.groups
    )

    print()
    print("Split verification:")
    print("  Same seed reproduces split: PASS")
    print("  No scene intersections:     PASS")
    print("  Every scene assigned:       PASS")

    print()
    print("First scene tokens:")
    print(
        f"  Train:      "
        f"{scene_split.train_scene_tokens[0]}"
    )
    print(
        f"  Validation: "
        f"{scene_split.validation_scene_tokens[0]}"
    )
    print(
        f"  Test:       "
        f"{scene_split.test_scene_tokens[0]}"
    )

    record_assignment = assign_records_to_split(
        grouping=grouping,
        scene_split=scene_split,
    )

    print()
    print("DriveLM records assigned to splits:")

    for partition in (
        record_assignment.train,
        record_assignment.validation,
        record_assignment.test,
    ):
        print()
        print(f"  Split:             {partition.split_name}")
        print(f"  Scenes:            {partition.scene_count:,}")
        print(f"  Frames:            {partition.frame_count:,}")
        print(
            f"  Resolved images:   "
            f"{partition.resolved_image_count:,}"
        )
        print(
            f"  Unresolved images: "
            f"{partition.unresolved_image_count:,}"
        )
        print(f"  QA records:        {partition.qa_count:,}")
        print(
            f"  Parsed objects:    "
            f"{partition.parsed_object_count:,}"
        )
        print(
            f"  Rejected objects:  "
            f"{partition.rejected_object_count:,}"
        )

    assert (
        record_assignment.total_scene_count
        == grouping.scene_count
    )

    assert (
        record_assignment.total_frame_count
        == grouping.frame_count
    )

    assert (
        record_assignment.total_resolved_image_count
        == grouping.resolved_image_count
    )

    assert (
        record_assignment.total_qa_count
        == grouping.qa_count
    )

    assert (
        record_assignment.total_parsed_object_count
        == grouping.parsed_object_count
    )

    for partition in (
        record_assignment.train,
        record_assignment.validation,
        record_assignment.test,
    ):
        assert all(
            record_assignment.scene_to_split[
                group.scene_token
            ]
            == partition.split_name
            for group in partition.scene_groups
        )

    print()
    print("Record-assignment verification:")
    print("  Every scene assigned once:       PASS")
    print("  Frame counts preserved:          PASS")
    print("  Image counts preserved:          PASS")
    print("  QA counts preserved:             PASS")
    print("  Object counts preserved:         PASS")
    print("  Records inherit scene split:     PASS")

    record_assignment = assign_records_to_split(
        grouping=grouping,
        scene_split=scene_split,
    )

    print()
    print("DriveLM records assigned to splits:")

    for partition in (
        record_assignment.train,
        record_assignment.validation,
        record_assignment.test,
    ):
        print()
        print(f"  Split:             {partition.split_name}")
        print(f"  Scenes:            {partition.scene_count:,}")
        print(f"  Frames:            {partition.frame_count:,}")
        print(
            f"  Resolved images:   "
            f"{partition.resolved_image_count:,}"
        )
        print(
            f"  Unresolved images: "
            f"{partition.unresolved_image_count:,}"
        )
        print(f"  QA records:        {partition.qa_count:,}")
        print(
            f"  Parsed objects:    "
            f"{partition.parsed_object_count:,}"
        )
        print(
            f"  Rejected objects:  "
            f"{partition.rejected_object_count:,}"
        )

    assert (
        record_assignment.total_scene_count
        == grouping.scene_count
    )

    assert (
        record_assignment.total_frame_count
        == grouping.frame_count
    )

    assert (
        record_assignment.total_resolved_image_count
        == grouping.resolved_image_count
    )

    assert (
        record_assignment.total_qa_count
        == grouping.qa_count
    )

    assert (
        record_assignment.total_parsed_object_count
        == grouping.parsed_object_count
    )

    for partition in (
        record_assignment.train,
        record_assignment.validation,
        record_assignment.test,
    ):
        assert all(
            record_assignment.scene_to_split[
                group.scene_token
            ]
            == partition.split_name
            for group in partition.scene_groups
        )

    print()
    print("Record-assignment verification:")
    print("  Every scene assigned once:       PASS")
    print("  Frame counts preserved:          PASS")
    print("  Image counts preserved:          PASS")
    print("  QA counts preserved:             PASS")
    print("  Object counts preserved:         PASS")
    print("  Records inherit scene split:     PASS")        
    
if __name__ == "__main__":
    main()