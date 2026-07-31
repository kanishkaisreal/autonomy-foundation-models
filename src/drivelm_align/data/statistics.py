from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from drivelm_align.data.split_types import (
    DriveLMRecordSplitAssignment,
    DriveLMSplitPartition,
)


def _counts(values: Iterable[str]) -> dict[str, int]:
    """Return sorted value counts."""
    return dict(sorted(Counter(values).items()))


def _length_distribution(
    texts: Iterable[str],
) -> dict[str, object]:
    """Summarize text lengths measured in words."""
    lengths = [len(text.split()) for text in texts]

    if not lengths:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "histogram": {},
        }

    return {
        "count": len(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
        "min": min(lengths),
        "max": max(lengths),
        "histogram": {
            str(length): count
            for length, count in sorted(
                Counter(lengths).items()
            )
        },
    }


def _partition_statistics(
    partition: DriveLMSplitPartition,
) -> dict[str, object]:
    """Compute statistics for one split partition."""
    qa_records = [
        record
        for group in partition.scene_groups
        for record in group.qa_records
    ]

    image_records = [
        record
        for group in partition.scene_groups
        for record in group.image_records
    ]

    object_records = [
        record
        for group in partition.scene_groups
        for record in group.object_records
    ]

    # DriveLM supplies free-text scene descriptions rather than a
    # separate controlled scene-condition label.
    scene_descriptions = [
        next(
            (
                record.scene_description.strip()
                for record in group.qa_records
                if record.scene_description
                and record.scene_description.strip()
            ),
            "unknown",
        )
        for group in partition.scene_groups
    ]

    answers = [
        record.answer
        for record in qa_records
        if record.answer_status == "answered"
        and record.answer
    ]

    return {
        "counts": {
            "scenes": partition.scene_count,
            "frames": partition.frame_count,
            "resolved_images": partition.resolved_image_count,
            "unresolved_images": partition.unresolved_image_count,
            "qa_records": partition.qa_count,
            "parsed_objects": partition.parsed_object_count,
            "rejected_objects": partition.rejected_object_count,
        },
        "tasks": _counts(
            record.task_name
            for record in qa_records
        ),
        "cameras": _counts(
            record.camera_name
            for record in image_records
        ),
        "object_categories": _counts(
            record.category or "unknown"
            for record in object_records
        ),
        "object_statuses": _counts(
            record.status or "unknown"
            for record in object_records
        ),
        "scene_descriptions": _counts(
            scene_descriptions
        ),
        "prompt_length_words": _length_distribution(
            record.question
            for record in qa_records
        ),
        "answer_length_words": _length_distribution(
            answers
        ),
    }


def compute_split_statistics(
    assignment: DriveLMRecordSplitAssignment,
) -> dict[str, dict[str, object]]:
    """Compute source-derived distributions for all three splits."""
    return {
        "train": _partition_statistics(
            assignment.train
        ),
        "validation": _partition_statistics(
            assignment.validation
        ),
        "test": _partition_statistics(
            assignment.test
        ),
    }


def main() -> None:
    """Compute compact real-data split statistics using F5."""
    from drivelm_align.data._debug import build_debug_assignment

    statistics = compute_split_statistics(build_debug_assignment())
    for split_name, split_statistics in statistics.items():
        counts = split_statistics["counts"]
        print(
            f"{split_name}: scenes={counts['scenes']}, "
            f"QA={counts['qa_records']:,}, "
            f"tasks={len(split_statistics['tasks'])}"
        )


if __name__ == "__main__":
    main()
