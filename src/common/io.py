from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable, Iterator, Mapping

class JsonlWriteError(ValueError):
    """Raised when records cannot be written as valid JSONL."""

class JsonlReadError(ValueError):
    """Raised when a JSONL file contains an invalid record."""
    

def save_jsonl_records(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> int:
    """
    Write mapping records to a UTF-8 JSONL file atomically.

    Each input record is serialized as one JSON object per line. Dictionary
    keys are sorted so identical records produce byte-identical output.

    Args:
        records:
            Iterable of dictionary-like records.
        output_path:
            Destination JSONL file.

    Returns:
        Number of records written.

    Raises:
        JsonlWriteError:
            If a record is not a mapping or cannot be serialized to JSON.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    record_count = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            for line_number, record in enumerate(records, start=1):
                if not isinstance(record, Mapping):
                    raise JsonlWriteError(
                        f"Record {line_number} must be a mapping, "
                        f"but received {type(record).__name__}."
                    )

                try:
                    serialized_record = json.dumps(
                        dict(record),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError) as exc:
                    raise JsonlWriteError(
                        f"Record {line_number} is not JSON serializable: {exc}"
                    ) from exc

                temporary_file.write(serialized_record)
                temporary_file.write("\n")
                record_count += 1

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        raise

    return record_count

def load_jsonl_records(
    input_path: str | Path,
    *,
    limit: int | None = None,
    validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> Iterator[Any]:
    """
    Stream records from a UTF-8 JSONL file.

    Args:
        input_path:
            JSONL file to read.
        limit:
            Optional maximum number of records to return.
        validator:
            Optional function that validates or converts each record.

    Yields:
        One parsed or validated record at a time.

    Raises:
        JsonlReadError:
            If the file is missing, a line is malformed, a record is not a
            JSON object, or schema validation fails.
    """
    source = Path(input_path).expanduser().resolve()

    if not source.is_file():
        raise JsonlReadError(
            f"JSONL file does not exist: {source}"
        )

    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise JsonlReadError(
                f"limit must be an integer or None; "
                f"received {type(limit).__name__}."
            )

        if limit < 0:
            raise JsonlReadError(
                f"limit must be nonnegative; received {limit}."
            )

        if limit == 0:
            return

    records_yielded = 0

    try:
        source_file = source.open(
            mode="r",
            encoding="utf-8",
        )
    except OSError as exc:
        raise JsonlReadError(
            f"Could not open JSONL file {source}: {exc}"
        ) from exc

    with source_file:
        for line_number, raw_line in enumerate(source_file, start=1):
            line = raw_line.strip()

            if not line:
                raise JsonlReadError(
                    f"Empty JSONL record at line {line_number} "
                    f"in {source}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlReadError(
                    f"Invalid JSON at line {line_number} in {source}: "
                    f"{exc.msg} at column {exc.colno}."
                ) from exc

            if not isinstance(record, Mapping):
                raise JsonlReadError(
                    f"Record at line {line_number} in {source} "
                    f"must be a JSON object, but received "
                    f"{type(record).__name__}."
                )

            if validator is not None:
                try:
                    record = validator(record)
                except Exception as exc:
                    raise JsonlReadError(
                        f"Schema validation failed at line {line_number} "
                        f"in {source}: {exc}"
                    ) from exc

            yield record

            records_yielded += 1

            if limit is not None and records_yielded >= limit:
                break
            
            
            
            
def main() -> None:
    """Write and reload temporary JSONL records using F5."""
    sample_records = [
        {
            "scene_id": "scene-001",
            "question": "What is directly ahead of the ego vehicle?",
            "objects": ["car-1", "pedestrian-1"],
        },
        {
            "scene_id": "scene-002",
            "question": "Is the lead vehicle moving?",
            "objects": ["car-2"],
        },
    ]

    def validate_sample_record(
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        required_fields = {
            "scene_id",
            "question",
            "objects",
        }

        missing_fields = required_fields - record.keys()

        if missing_fields:
            raise ValueError(
                f"Missing required fields: {sorted(missing_fields)}"
            )

        return dict(record)

    with tempfile.TemporaryDirectory(
        prefix="afm-jsonl-check-"
    ) as temporary_directory:
        output_path = (
            Path(temporary_directory)
            / "sample_records.jsonl"
        )

        save_jsonl_records(
            records=sample_records,
            output_path=output_path,
        )

        loaded_records = list(
            load_jsonl_records(
                input_path=output_path,
                validator=validate_sample_record,
            )
        )

        first_record_only = list(
            load_jsonl_records(
                input_path=output_path,
                limit=1,
            )
        )

        print("JSONL records loaded successfully.")
        print()
        print(f"Records loaded: {len(loaded_records)}")
        print(f"First record:   {loaded_records[0]}")
        print(f"Limit=1 count:  {len(first_record_only)}")
        print(
            "Round trip matches:",
            loaded_records == sample_records,
        )

    print()
    print("Temporary JSONL file cleaned up automatically.")


if __name__ == "__main__":
    main()