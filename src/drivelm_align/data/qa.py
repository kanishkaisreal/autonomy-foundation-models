from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from drivelm_align.data.raw import (
    DriveLMAnnotations,
    load_drivelm_annotations,
)


DriveLMAnswerStatus = Literal[
    "answered",
    "missing",
    "null",
    "empty",
]


class DriveLMQAExtractionError(ValueError):
    """Raised when DriveLM QA records cannot be extracted reliably."""


@dataclass(frozen=True, slots=True)
class DriveLMQARecord:
    """
    One flattened DriveLM question-answer record.

    The source hierarchy is retained through the scene token, frame token,
    task name, and original task-list index.
    """

    scene_token: str
    frame_token: str
    task_name: str
    task_index: int

    scene_description: str
    question: str
    answer: str | None
    answer_status: DriveLMAnswerStatus

    context: Any
    con_up: Any
    con_down: Any
    cluster: Any
    layer: Any

    extra_fields: tuple[tuple[str, Any], ...]

    @property
    def record_id(self) -> str:
        """Return a deterministic ID derived from the source location."""
        return (
            f"{self.scene_token}:"
            f"{self.frame_token}:"
            f"{self.task_name}:"
            f"{self.task_index}"
        )


@dataclass(frozen=True, slots=True)
class DriveLMQAExtraction:
    """All flattened DriveLM QA records and extraction statistics."""

    records: tuple[DriveLMQARecord, ...]

    record_count: int
    answered_count: int
    unanswered_count: int

    counts_by_task: dict[str, int]
    answer_status_counts: dict[str, int]


KNOWN_QA_FIELDS = frozenset(
    {
        "Q",
        "A",
        "C",
        "con_up",
        "con_down",
        "cluster",
        "layer",
    }
)


def extract_drivelm_qa_records(
    annotations: DriveLMAnnotations,
    *,
    strict_answers: bool = False,
) -> DriveLMQAExtraction:
    """
    Flatten native DriveLM QA lists into typed records.

    The function preserves every record's original:

        scene token
        frame token
        QA task name
        task-list index
        question
        answer
        graph/context metadata

    Answer status is represented explicitly:

        answered:
            A non-empty string answer exists.

        missing:
            The source QA dictionary has no ``A`` field.

        null:
            The source QA dictionary contains ``"A": null``.

        empty:
            The source contains an empty or whitespace-only string.

    Args:
        annotations:
            Raw DriveLM annotations returned by Function 013.

        strict_answers:
            When True, extraction fails upon encountering any record
            whose answer status is not ``answered``.

            When False, imperfect records are preserved and reported.
            This should be used while inspecting the public dataset.

    Returns:
        Typed QA records and extraction statistics.

    Raises:
        DriveLMQAExtractionError:
            If the QA structure is malformed, source locations collide,
            questions are invalid, answer types are unsupported, or
            extracted counts disagree with Function 013.
    """
    extracted_records: list[DriveLMQARecord] = []

    counts_by_task: dict[str, int] = {}

    answer_status_counts: dict[str, int] = {
        "answered": 0,
        "missing": 0,
        "null": 0,
        "empty": 0,
    }

    answered_count = 0
    unanswered_count = 0

    seen_source_locations: set[
        tuple[str, str, str, int]
    ] = set()

    for scene_token in sorted(annotations.scenes):
        scene_data = annotations.scenes[scene_token]

        if not isinstance(scene_data, dict):
            raise DriveLMQAExtractionError(
                f"Scene {scene_token!r} must contain a mapping."
            )

        scene_description = scene_data.get(
            "scene_description",
            "",
        )

        if not isinstance(scene_description, str):
            raise DriveLMQAExtractionError(
                "'scene_description' must be a string for "
                f"scene={scene_token!r}."
            )

        key_frames = scene_data.get("key_frames")

        if not isinstance(key_frames, dict):
            raise DriveLMQAExtractionError(
                f"Scene {scene_token!r} does not contain a valid "
                "'key_frames' mapping."
            )

        for frame_token in sorted(key_frames):
            frame_data = key_frames[frame_token]

            if not isinstance(frame_data, dict):
                raise DriveLMQAExtractionError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} must contain a mapping."
                )

            qa_tasks = frame_data.get("QA")

            if not isinstance(qa_tasks, dict):
                raise DriveLMQAExtractionError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} does not contain a valid "
                    "'QA' task mapping."
                )

            for task_name in sorted(qa_tasks):
                if (
                    not isinstance(task_name, str)
                    or not task_name.strip()
                ):
                    raise DriveLMQAExtractionError(
                        "Invalid QA task name for "
                        f"scene={scene_token!r}, "
                        f"frame={frame_token!r}: "
                        f"{task_name!r}"
                    )

                task_records = qa_tasks[task_name]

                if not isinstance(task_records, list):
                    raise DriveLMQAExtractionError(
                        f"QA task {task_name!r} must contain a list for "
                        f"scene={scene_token!r}, "
                        f"frame={frame_token!r}."
                    )

                for task_index, qa_record in enumerate(task_records):
                    if not isinstance(qa_record, dict):
                        raise DriveLMQAExtractionError(
                            f"QA record {task_index} under task "
                            f"{task_name!r} must contain a mapping for "
                            f"scene={scene_token!r}, "
                            f"frame={frame_token!r}."
                        )

                    source_location = (
                        scene_token,
                        frame_token,
                        task_name,
                        task_index,
                    )

                    if source_location in seen_source_locations:
                        raise DriveLMQAExtractionError(
                            "Duplicate QA source location encountered: "
                            f"{source_location!r}"
                        )

                    seen_source_locations.add(source_location)

                    question = qa_record.get("Q")

                    if (
                        not isinstance(question, str)
                        or not question.strip()
                    ):
                        raise DriveLMQAExtractionError(
                            "QA record is missing a non-empty question: "
                            f"scene={scene_token!r}, "
                            f"frame={frame_token!r}, "
                            f"task={task_name!r}, "
                            f"index={task_index}."
                        )

                    # ---------------------------------------------
                    # Classify the answer without altering it.
                    # ---------------------------------------------
                    if "A" not in qa_record:
                        answer: str | None = None
                        answer_status: DriveLMAnswerStatus = "missing"

                    else:
                        raw_answer = qa_record["A"]

                        if raw_answer is None:
                            answer = None
                            answer_status = "null"

                        elif not isinstance(raw_answer, str):
                            raise DriveLMQAExtractionError(
                                "Answer must be a string or None for "
                                f"scene={scene_token!r}, "
                                f"frame={frame_token!r}, "
                                f"task={task_name!r}, "
                                f"index={task_index}; "
                                f"received "
                                f"{type(raw_answer).__name__}."
                            )

                        elif not raw_answer.strip():
                            # Preserve the original empty or whitespace
                            # string rather than replacing it with None.
                            answer = raw_answer
                            answer_status = "empty"

                        else:
                            answer = raw_answer
                            answer_status = "answered"

                    answer_status_counts[answer_status] += 1

                    if answer_status == "answered":
                        answered_count += 1
                    else:
                        unanswered_count += 1

                    if (
                        strict_answers
                        and answer_status != "answered"
                    ):
                        raise DriveLMQAExtractionError(
                            "QA record does not contain a usable answer "
                            "while strict answer validation is enabled: "
                            f"scene={scene_token!r}, "
                            f"frame={frame_token!r}, "
                            f"task={task_name!r}, "
                            f"index={task_index}, "
                            f"answer_status={answer_status!r}."
                        )

                    extra_fields = tuple(
                        (
                            field_name,
                            qa_record[field_name],
                        )
                        for field_name in sorted(qa_record)
                        if field_name not in KNOWN_QA_FIELDS
                    )

                    extracted_records.append(
                        DriveLMQARecord(
                            scene_token=scene_token,
                            frame_token=frame_token,
                            task_name=task_name,
                            task_index=task_index,
                            scene_description=scene_description,
                            question=question,
                            answer=answer,
                            answer_status=answer_status,
                            context=qa_record.get("C"),
                            con_up=qa_record.get("con_up"),
                            con_down=qa_record.get("con_down"),
                            cluster=qa_record.get("cluster"),
                            layer=qa_record.get("layer"),
                            extra_fields=extra_fields,
                        )
                    )

                    counts_by_task[task_name] = (
                        counts_by_task.get(task_name, 0) + 1
                    )

    record_count = len(extracted_records)

    if record_count != annotations.qa_count:
        raise DriveLMQAExtractionError(
            "Extracted QA count does not match Function 013: "
            f"extracted={record_count}, "
            f"loader={annotations.qa_count}."
        )

    if answered_count + unanswered_count != record_count:
        raise DriveLMQAExtractionError(
            "Answered and unanswered QA counts do not reconcile: "
            f"answered={answered_count}, "
            f"unanswered={unanswered_count}, "
            f"total={record_count}."
        )

    if sum(answer_status_counts.values()) != record_count:
        raise DriveLMQAExtractionError(
            "Answer-status counts do not reconcile with the total: "
            f"status_total={sum(answer_status_counts.values())}, "
            f"record_count={record_count}."
        )

    if sum(counts_by_task.values()) != record_count:
        raise DriveLMQAExtractionError(
            "Task counts do not reconcile with the total: "
            f"task_total={sum(counts_by_task.values())}, "
            f"record_count={record_count}."
        )

    return DriveLMQAExtraction(
        records=tuple(extracted_records),
        record_count=record_count,
        answered_count=answered_count,
        unanswered_count=unanswered_count,
        counts_by_task=dict(
            sorted(counts_by_task.items())
        ),
        answer_status_counts=dict(
            sorted(answer_status_counts.items())
        ),
    )


def main() -> None:
    """Extract and inspect DriveLM training QA records using F5."""
    repository_root = Path(__file__).resolve().parents[3]

    annotation_path = (
        repository_root
        / "data"
        / "drivelm"
        / "QA_dataset_nus"
        / "v1_1_train_nus.json"
    )

    annotations = load_drivelm_annotations(annotation_path)

    extraction = extract_drivelm_qa_records(
        annotations,
        strict_answers=False,
    )

    print("DriveLM QA extraction completed successfully.")
    print()
    print(
        f"QA records extracted: "
        f"{extraction.record_count:,}"
    )
    print(
        f"Answered records:     "
        f"{extraction.answered_count:,}"
    )
    print(
        f"Unanswered records:   "
        f"{extraction.unanswered_count:,}"
    )

    print()
    print("QA counts by task:")

    for task_name, task_count in (
        extraction.counts_by_task.items()
    ):
        print(f"  {task_name}: {task_count:,}")

    print()
    print("Answer-status counts:")

    for answer_status, count in (
        extraction.answer_status_counts.items()
    ):
        print(f"  {answer_status}: {count:,}")

    if not extraction.records:
        print()
        print("No QA records were extracted.")
        return

    first_record = extraction.records[0]

    print()
    print("First flattened QA record:")
    print(f"  Record ID:     {first_record.record_id}")
    print(f"  Scene:         {first_record.scene_token}")
    print(f"  Frame:         {first_record.frame_token}")
    print(f"  Task:          {first_record.task_name}")
    print(f"  Index:         {first_record.task_index}")
    print(f"  Question:      {first_record.question}")
    print(f"  Answer:        {first_record.answer!r}")
    print(
        f"  Answer status: "
        f"{first_record.answer_status}"
    )

    # ---------------------------------------------------------
    # Manual round-trip verification.
    # ---------------------------------------------------------
    source_scene = annotations.scenes[
        first_record.scene_token
    ]

    source_frame = source_scene["key_frames"][
        first_record.frame_token
    ]

    source_qa_record = source_frame["QA"][
        first_record.task_name
    ][first_record.task_index]

    assert source_qa_record["Q"] == first_record.question

    if first_record.answer_status == "missing":
        assert "A" not in source_qa_record
    else:
        assert "A" in source_qa_record
        assert source_qa_record["A"] == first_record.answer

    print()
    print("Round-trip verification:")
    print("  Scene token recovered:   PASS")
    print("  Frame token recovered:   PASS")
    print("  Original question:       PASS")
    print("  Original answer state:   PASS")

    first_unanswered_record = next(
        (
            record
            for record in extraction.records
            if record.answer_status != "answered"
        ),
        None,
    )

    if first_unanswered_record is not None:
        print()
        print("First QA record without a usable answer:")
        print(
            f"  Record ID:     "
            f"{first_unanswered_record.record_id}"
        )
        print(
            f"  Scene:         "
            f"{first_unanswered_record.scene_token}"
        )
        print(
            f"  Frame:         "
            f"{first_unanswered_record.frame_token}"
        )
        print(
            f"  Task:          "
            f"{first_unanswered_record.task_name}"
        )
        print(
            f"  Index:         "
            f"{first_unanswered_record.task_index}"
        )
        print(
            f"  Question:      "
            f"{first_unanswered_record.question}"
        )
        print(
            f"  Answer:        "
            f"{first_unanswered_record.answer!r}"
        )
        print(
            f"  Answer status: "
            f"{first_unanswered_record.answer_status}"
        )


if __name__ == "__main__":
    main()