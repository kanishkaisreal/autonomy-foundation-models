from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivelm_align.data.raw import (
    DriveLMAnnotations,
    load_drivelm_annotations,
)


DEFAULT_CAMERA_NAMES = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
)

KNOWN_OBJECT_FIELDS = frozenset(
    {
        "Category",
        "Status",
        "Visual_description",
        "2d_bbox",
    }
)

OBJECT_IDENTIFIER_PATTERN = re.compile(r"c[0-9]+")


class DriveLMObjectExtractionError(ValueError):
    """Raised when the object collection cannot be extracted reliably."""


class DriveLMObjectTagParseError(ValueError):
    """Raised when one native DriveLM object tag cannot be parsed."""


@dataclass(frozen=True, slots=True)
class DriveLMObjectTagRecord:
    """One successfully extracted DriveLM key-object record."""

    scene_token: str
    frame_token: str

    raw_tag: str
    object_id: str
    camera_name: str

    center_x: float
    center_y: float

    category: str
    status: str | None
    visual_description: str | None

    bbox_xyxy: tuple[float, float, float, float]

    extra_fields: tuple[tuple[str, Any], ...]

    @property
    def record_id(self) -> str:
        """Return a deterministic source-derived object-record ID."""
        return (
            f"{self.scene_token}:"
            f"{self.frame_token}:"
            f"{self.raw_tag}"
        )

    @property
    def bbox_width(self) -> float:
        """Return the native 2D bounding-box width."""
        x_min, _, x_max, _ = self.bbox_xyxy
        return x_max - x_min

    @property
    def bbox_height(self) -> float:
        """Return the native 2D bounding-box height."""
        _, y_min, _, y_max = self.bbox_xyxy
        return y_max - y_min

    @property
    def bbox_center(self) -> tuple[float, float]:
        """Return the center calculated from the native bounding box."""
        x_min, y_min, x_max, y_max = self.bbox_xyxy

        return (
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
        )

    @property
    def center_error_pixels(self) -> float:
        """
        Return the distance between tag center and bounding-box center.

        A small non-zero value can occur because the tag coordinates may
        be rounded in the source annotations.
        """
        bbox_center_x, bbox_center_y = self.bbox_center

        return math.hypot(
            self.center_x - bbox_center_x,
            self.center_y - bbox_center_y,
        )


@dataclass(frozen=True, slots=True)
class RejectedDriveLMObjectTag:
    """One source object that could not be converted into a typed record."""

    scene_token: str
    frame_token: str
    raw_tag: str
    reason: str
    raw_metadata: Any


@dataclass(frozen=True, slots=True)
class DriveLMObjectTagExtraction:
    """Extracted DriveLM object tags and data-quality statistics."""

    records: tuple[DriveLMObjectTagRecord, ...]
    rejected: tuple[RejectedDriveLMObjectTag, ...]

    source_object_count: int
    parsed_count: int
    rejected_count: int

    counts_by_camera: dict[str, int]
    counts_by_category: dict[str, int]
    counts_by_status: dict[str, int]

    maximum_center_error_pixels: float


def _parse_finite_float(
    value: Any,
    *,
    field_name: str,
) -> float:
    """Convert a numeric value into a finite float."""
    if isinstance(value, bool):
        raise DriveLMObjectTagParseError(
            f"{field_name} must be numeric, not boolean."
        )

    try:
        parsed_value = float(value)

    except (TypeError, ValueError) as exc:
        raise DriveLMObjectTagParseError(
            f"{field_name} must be numeric; received {value!r}."
        ) from exc

    if not math.isfinite(parsed_value):
        raise DriveLMObjectTagParseError(
            f"{field_name} must be finite; received {value!r}."
        )

    return parsed_value


def _parse_native_object_tag(
    raw_tag: str,
    *,
    allowed_camera_names: frozenset[str],
) -> tuple[str, str, float, float]:
    """
    Parse a native tag such as:

        <c1,CAM_FRONT,258.3,442.5>

    Returns:
        ``(object_id, camera_name, center_x, center_y)``.
    """
    if not isinstance(raw_tag, str) or not raw_tag.strip():
        raise DriveLMObjectTagParseError(
            "Object tag must be a non-empty string."
        )

    normalized_tag = raw_tag.strip()

    if not (
        normalized_tag.startswith("<")
        and normalized_tag.endswith(">")
    ):
        raise DriveLMObjectTagParseError(
            "Object tag must begin with '<' and end with '>'."
        )

    inner_tag = normalized_tag[1:-1]

    tag_parts = tuple(
        part.strip()
        for part in inner_tag.split(",")
    )

    if len(tag_parts) != 4:
        raise DriveLMObjectTagParseError(
            "Object tag must contain exactly four comma-separated "
            f"components; received {len(tag_parts)}."
        )

    object_id, camera_name, raw_center_x, raw_center_y = tag_parts

    if OBJECT_IDENTIFIER_PATTERN.fullmatch(object_id) is None:
        raise DriveLMObjectTagParseError(
            "Object identifier must use the form c<number>; "
            f"received {object_id!r}."
        )

    if camera_name not in allowed_camera_names:
        raise DriveLMObjectTagParseError(
            f"Unknown camera name {camera_name!r}."
        )

    center_x = _parse_finite_float(
        raw_center_x,
        field_name="tag center_x",
    )

    center_y = _parse_finite_float(
        raw_center_y,
        field_name="tag center_y",
    )

    return (
        object_id,
        camera_name,
        center_x,
        center_y,
    )


def _parse_bbox(
    raw_bbox: Any,
) -> tuple[float, float, float, float]:
    """Parse and validate a native ``[x_min, y_min, x_max, y_max]`` box."""
    if not isinstance(raw_bbox, (list, tuple)):
        raise DriveLMObjectTagParseError(
            "'2d_bbox' must be a list or tuple."
        )

    if len(raw_bbox) != 4:
        raise DriveLMObjectTagParseError(
            "'2d_bbox' must contain exactly four values; "
            f"received {len(raw_bbox)}."
        )

    x_min = _parse_finite_float(
        raw_bbox[0],
        field_name="bbox x_min",
    )

    y_min = _parse_finite_float(
        raw_bbox[1],
        field_name="bbox y_min",
    )

    x_max = _parse_finite_float(
        raw_bbox[2],
        field_name="bbox x_max",
    )

    y_max = _parse_finite_float(
        raw_bbox[3],
        field_name="bbox y_max",
    )

    if x_max < x_min:
        raise DriveLMObjectTagParseError(
            f"bbox x_max={x_max} is smaller than x_min={x_min}."
        )

    if y_max < y_min:
        raise DriveLMObjectTagParseError(
            f"bbox y_max={y_max} is smaller than y_min={y_min}."
        )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def _parse_optional_string(
    value: Any,
    *,
    field_name: str,
) -> str | None:
    """Validate a source string that may legitimately be null."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise DriveLMObjectTagParseError(
            f"{field_name} must be a string or None; "
            f"received {type(value).__name__}."
        )

    return value


def _build_object_record(
    *,
    scene_token: str,
    frame_token: str,
    raw_tag: str,
    raw_metadata: Any,
    allowed_camera_names: frozenset[str],
) -> DriveLMObjectTagRecord:
    """Convert one native object entry into a typed object record."""
    if not isinstance(raw_metadata, dict):
        raise DriveLMObjectTagParseError(
            "Object metadata must be a mapping."
        )

    (
        object_id,
        camera_name,
        center_x,
        center_y,
    ) = _parse_native_object_tag(
        raw_tag,
        allowed_camera_names=allowed_camera_names,
    )

    category = raw_metadata.get("Category")

    if not isinstance(category, str) or not category.strip():
        raise DriveLMObjectTagParseError(
            "'Category' must be a non-empty string."
        )

    status = _parse_optional_string(
        raw_metadata.get("Status"),
        field_name="Status",
    )

    visual_description = _parse_optional_string(
        raw_metadata.get("Visual_description"),
        field_name="Visual_description",
    )

    bbox_xyxy = _parse_bbox(
        raw_metadata.get("2d_bbox")
    )

    extra_fields = tuple(
        (
            field_name,
            raw_metadata[field_name],
        )
        for field_name in sorted(raw_metadata)
        if field_name not in KNOWN_OBJECT_FIELDS
    )

    return DriveLMObjectTagRecord(
        scene_token=scene_token,
        frame_token=frame_token,
        raw_tag=raw_tag,
        object_id=object_id,
        camera_name=camera_name,
        center_x=center_x,
        center_y=center_y,
        category=category,
        status=status,
        visual_description=visual_description,
        bbox_xyxy=bbox_xyxy,
        extra_fields=extra_fields,
    )


def extract_drivelm_object_tags(
    annotations: DriveLMAnnotations,
    *,
    allowed_camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
) -> DriveLMObjectTagExtraction:
    """
    Extract typed object records from DriveLM ``key_object_infos``.

    A malformed object does not terminate the complete extraction. It is
    placed in ``rejected`` with its exact source location, original tag,
    original metadata, and rejection reason.

    Args:
        annotations:
            Raw DriveLM annotations returned by Function 013.

        allowed_camera_names:
            Camera identifiers accepted inside native object tags.

    Returns:
        Parsed object records, rejected source objects, and aggregate
        data-quality statistics.

    Raises:
        DriveLMObjectExtractionError:
            If the scene/frame/object hierarchy is malformed or the final
            parsed and rejected counts do not reconcile with the source.
    """
    if not allowed_camera_names:
        raise ValueError(
            "allowed_camera_names must contain at least one camera."
        )

    if len(set(allowed_camera_names)) != len(allowed_camera_names):
        raise ValueError(
            "allowed_camera_names contains duplicate camera names."
        )

    allowed_cameras = frozenset(allowed_camera_names)

    parsed_records: list[DriveLMObjectTagRecord] = []
    rejected_records: list[RejectedDriveLMObjectTag] = []

    counts_by_camera: dict[str, int] = {}
    counts_by_category: dict[str, int] = {}
    counts_by_status: dict[str, int] = {}

    seen_source_locations: set[
        tuple[str, str, str]
    ] = set()

    source_object_count = 0
    maximum_center_error_pixels = 0.0

    for scene_token in sorted(annotations.scenes):
        scene_data = annotations.scenes[scene_token]

        if not isinstance(scene_data, dict):
            raise DriveLMObjectExtractionError(
                f"Scene {scene_token!r} must contain a mapping."
            )

        key_frames = scene_data.get("key_frames")

        if not isinstance(key_frames, dict):
            raise DriveLMObjectExtractionError(
                f"Scene {scene_token!r} does not contain a valid "
                "'key_frames' mapping."
            )

        for frame_token in sorted(key_frames):
            frame_data = key_frames[frame_token]

            if not isinstance(frame_data, dict):
                raise DriveLMObjectExtractionError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} must contain a mapping."
                )

            key_object_infos = frame_data.get(
                "key_object_infos"
            )

            if not isinstance(key_object_infos, dict):
                raise DriveLMObjectExtractionError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} does not contain a valid "
                    "'key_object_infos' mapping."
                )

            for raw_tag in sorted(key_object_infos):
                raw_metadata = key_object_infos[raw_tag]
                source_object_count += 1

                source_location = (
                    scene_token,
                    frame_token,
                    raw_tag,
                )

                if source_location in seen_source_locations:
                    raise DriveLMObjectExtractionError(
                        "Duplicate object source location encountered: "
                        f"{source_location!r}"
                    )

                seen_source_locations.add(source_location)

                try:
                    object_record = _build_object_record(
                        scene_token=scene_token,
                        frame_token=frame_token,
                        raw_tag=raw_tag,
                        raw_metadata=raw_metadata,
                        allowed_camera_names=allowed_cameras,
                    )

                except DriveLMObjectTagParseError as exc:
                    rejected_records.append(
                        RejectedDriveLMObjectTag(
                            scene_token=scene_token,
                            frame_token=frame_token,
                            raw_tag=raw_tag,
                            reason=str(exc),
                            raw_metadata=raw_metadata,
                        )
                    )
                    continue

                parsed_records.append(object_record)

                counts_by_camera[object_record.camera_name] = (
                    counts_by_camera.get(
                        object_record.camera_name,
                        0,
                    )
                    + 1
                )

                counts_by_category[object_record.category] = (
                    counts_by_category.get(
                        object_record.category,
                        0,
                    )
                    + 1
                )

                status_key = (
                    object_record.status
                    if object_record.status is not None
                    else "<NULL>"
                )

                counts_by_status[status_key] = (
                    counts_by_status.get(status_key, 0) + 1
                )

                maximum_center_error_pixels = max(
                    maximum_center_error_pixels,
                    object_record.center_error_pixels,
                )

    parsed_count = len(parsed_records)
    rejected_count = len(rejected_records)

    if parsed_count + rejected_count != source_object_count:
        raise DriveLMObjectExtractionError(
            "Parsed and rejected object counts do not reconcile: "
            f"parsed={parsed_count}, "
            f"rejected={rejected_count}, "
            f"source={source_object_count}."
        )

    if sum(counts_by_camera.values()) != parsed_count:
        raise DriveLMObjectExtractionError(
            "Camera counts do not reconcile with parsed object count."
        )

    if sum(counts_by_category.values()) != parsed_count:
        raise DriveLMObjectExtractionError(
            "Category counts do not reconcile with parsed object count."
        )

    if sum(counts_by_status.values()) != parsed_count:
        raise DriveLMObjectExtractionError(
            "Status counts do not reconcile with parsed object count."
        )

    return DriveLMObjectTagExtraction(
        records=tuple(parsed_records),
        rejected=tuple(rejected_records),
        source_object_count=source_object_count,
        parsed_count=parsed_count,
        rejected_count=rejected_count,
        counts_by_camera=dict(
            sorted(counts_by_camera.items())
        ),
        counts_by_category=dict(
            sorted(counts_by_category.items())
        ),
        counts_by_status=dict(
            sorted(counts_by_status.items())
        ),
        maximum_center_error_pixels=(
            maximum_center_error_pixels
        ),
    )


def main() -> None:
    """Extract and inspect DriveLM object tags using F5."""
    repository_root = Path(__file__).resolve().parents[3]

    annotation_path = (
        repository_root
        / "data"
        / "drivelm"
        / "QA_dataset_nus"
        / "v1_1_train_nus.json"
    )

    annotations = load_drivelm_annotations(
        annotation_path
    )

    extraction = extract_drivelm_object_tags(
        annotations
    )

    print("DriveLM object-tag extraction completed.")
    print()
    print(
        f"Source objects:   "
        f"{extraction.source_object_count:,}"
    )
    print(
        f"Parsed objects:   "
        f"{extraction.parsed_count:,}"
    )
    print(
        f"Rejected objects: "
        f"{extraction.rejected_count:,}"
    )
    print(
        "Maximum tag/bbox center error: "
        f"{extraction.maximum_center_error_pixels:.4f} pixels"
    )

    print()
    print("Object counts by camera:")

    for camera_name, count in (
        extraction.counts_by_camera.items()
    ):
        print(f"  {camera_name}: {count:,}")

    print()
    print("Object counts by category:")

    for category, count in (
        extraction.counts_by_category.items()
    ):
        print(f"  {category}: {count:,}")

    print()
    print("Object counts by status:")

    for status, count in (
        extraction.counts_by_status.items()
    ):
        print(f"  {status}: {count:,}")

    if extraction.records:
        first_record = extraction.records[0]

        print()
        print("First parsed object:")
        print(
            f"  Record ID:          "
            f"{first_record.record_id}"
        )
        print(
            f"  Raw tag:            "
            f"{first_record.raw_tag}"
        )
        print(
            f"  Object ID:          "
            f"{first_record.object_id}"
        )
        print(
            f"  Camera:             "
            f"{first_record.camera_name}"
        )
        print(
            f"  Tag center:         "
            f"({first_record.center_x}, "
            f"{first_record.center_y})"
        )
        print(
            f"  Category:           "
            f"{first_record.category}"
        )
        print(
            f"  Status:             "
            f"{first_record.status!r}"
        )
        print(
            f"  Description:        "
            f"{first_record.visual_description!r}"
        )
        print(
            f"  Bounding box:       "
            f"{first_record.bbox_xyxy}"
        )
        print(
            f"  Box width × height: "
            f"{first_record.bbox_width:.2f} × "
            f"{first_record.bbox_height:.2f}"
        )
        print(
            f"  Center error:       "
            f"{first_record.center_error_pixels:.4f} pixels"
        )

        # -----------------------------------------------------
        # Manual source round-trip verification.
        # -----------------------------------------------------
        source_metadata = annotations.scenes[
            first_record.scene_token
        ]["key_frames"][
            first_record.frame_token
        ]["key_object_infos"][
            first_record.raw_tag
        ]

        assert (
            source_metadata["Category"]
            == first_record.category
        )

        assert (
            source_metadata.get("Status")
            == first_record.status
        )

        assert (
            source_metadata.get("Visual_description")
            == first_record.visual_description
        )

        assert tuple(
            float(value)
            for value in source_metadata["2d_bbox"]
        ) == first_record.bbox_xyxy

        print()
        print("Round-trip verification:")
        print("  Scene token recovered:       PASS")
        print("  Frame token recovered:       PASS")
        print("  Original object tag:         PASS")
        print("  Original object metadata:    PASS")

    if extraction.rejected:
        print()
        print("First rejected object tags:")

        for rejected_record in extraction.rejected[:10]:
            print()
            print(
                f"  Scene:  "
                f"{rejected_record.scene_token}"
            )
            print(
                f"  Frame:  "
                f"{rejected_record.frame_token}"
            )
            print(
                f"  Tag:    "
                f"{rejected_record.raw_tag!r}"
            )
            print(
                f"  Reason: "
                f"{rejected_record.reason}"
            )


if __name__ == "__main__":
    main()