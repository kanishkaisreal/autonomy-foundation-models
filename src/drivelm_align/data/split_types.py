from __future__ import annotations

from dataclasses import dataclass

from drivelm_align.data.grouping import DriveLMSceneRecordGroup


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
        return self.train_count + self.validation_count + self.test_count

    @property
    def scene_to_split(self) -> dict[str, str]:
        """Map every scene token to its assigned split."""
        return {
            **{
                scene_token: "train"
                for scene_token in self.train_scene_tokens
            },
            **{
                scene_token: "validation"
                for scene_token in self.validation_scene_tokens
            },
            **{
                scene_token: "test"
                for scene_token in self.test_scene_tokens
            },
        }


@dataclass(frozen=True, slots=True)
class DriveLMSplitPartition:
    """All complete scene groups assigned to one dataset split."""

    split_name: str
    scene_groups: tuple[DriveLMSceneRecordGroup, ...]

    @property
    def scene_count(self) -> int:
        """Return the number of scenes in this partition."""
        return len(self.scene_groups)

    @property
    def frame_count(self) -> int:
        """Return the number of frames in this partition."""
        return sum(group.frame_count for group in self.scene_groups)

    @property
    def resolved_image_count(self) -> int:
        """Return the number of resolved images."""
        return sum(group.image_count for group in self.scene_groups)

    @property
    def unresolved_image_count(self) -> int:
        """Return the number of unresolved images."""
        return sum(
            group.unresolved_image_count for group in self.scene_groups
        )

    @property
    def qa_count(self) -> int:
        """Return the number of QA records."""
        return sum(group.qa_count for group in self.scene_groups)

    @property
    def parsed_object_count(self) -> int:
        """Return the number of parsed objects."""
        return sum(group.object_count for group in self.scene_groups)

    @property
    def rejected_object_count(self) -> int:
        """Return the number of rejected objects."""
        return sum(
            group.rejected_object_count for group in self.scene_groups
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
