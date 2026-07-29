from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path


class ChecksumError(ValueError):
    """Raised when a file checksum cannot be computed."""


def compute_file_checksum(
    file_path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Compute the SHA-256 checksum of a file.

    The file is read in chunks so large datasets and model checkpoints do not
    need to be loaded entirely into memory.

    Args:
        file_path:
            File whose checksum should be computed.
        chunk_size:
            Number of bytes read from disk at a time.

    Returns:
        A 64-character lowercase SHA-256 hexadecimal digest.

    Raises:
        ChecksumError:
            If the path is not a file, the chunk size is invalid, or the file
            cannot be read.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise ChecksumError(
            f"Checksum source must be an existing file: {path}"
        )

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ChecksumError(
            f"chunk_size must be an integer; "
            f"received {type(chunk_size).__name__}."
        )

    if chunk_size <= 0:
        raise ChecksumError(
            f"chunk_size must be greater than zero; "
            f"received {chunk_size}."
        )

    checksum = hashlib.sha256()

    try:
        with path.open("rb") as source_file:
            while chunk := source_file.read(chunk_size):
                checksum.update(chunk)

    except OSError as exc:
        raise ChecksumError(
            f"Could not read file {path}: {exc}"
        ) from exc

    return checksum.hexdigest()


def main() -> None:
    """Demonstrate checksum stability and change detection using F5."""
    with tempfile.TemporaryDirectory(
        prefix="afm-checksum-check-"
    ) as temporary_directory:
        sample_path = Path(temporary_directory) / "sample.txt"

        sample_path.write_text(
            "autonomy foundation models\n",
            encoding="utf-8",
        )

        first_checksum = compute_file_checksum(sample_path)
        second_checksum = compute_file_checksum(sample_path)

        sample_path.write_text(
            "autonomy foundation models!\n",
            encoding="utf-8",
        )

        changed_checksum = compute_file_checksum(sample_path)

        print("File checksums computed successfully.")
        print()
        print(f"First checksum:   {first_checksum}")
        print(f"Second checksum:  {second_checksum}")
        print(f"Changed checksum: {changed_checksum}")
        print()
        print(
            "Same file gives same checksum:",
            first_checksum == second_checksum,
        )
        print(
            "Changed file gives different checksum:",
            first_checksum != changed_checksum,
        )

    print()
    print("Temporary checksum file cleaned up automatically.")


if __name__ == "__main__":
    main()