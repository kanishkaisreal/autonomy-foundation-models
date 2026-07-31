from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from PIL import Image, UnidentifiedImageError

from drivelm_align.data.raw import (
    DriveLMAnnotations,
    load_drivelm_annotations,
)


class DriveLMImageResolutionError(ValueError):
    """Raised when DriveLM image references cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ResolvedDriveLMImage:
    """One DriveLM camera reference resolved to a local image file."""

    scene_token: str
    frame_token: str
    camera_name: str
    source_reference: str
    absolute_path: Path


@dataclass(frozen=True, slots=True)
class UnresolvedDriveLMImage:
    """One DriveLM camera reference that could not be found locally."""

    scene_token: str
    frame_token: str
    camera_name: str
    source_reference: str
    candidate_paths: tuple[Path, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class DriveLMImagePathResolution:
    """Results from resolving every DriveLM camera reference."""

    image_root: Path
    resolved: dict[
        tuple[str, str, str],
        ResolvedDriveLMImage,
    ]
    unresolved: tuple[UnresolvedDriveLMImage, ...]
    reference_count: int
    resolved_count: int
    unresolved_count: int


def _is_within_root(
    candidate: Path,
    root: Path,
) -> bool:
    """Return True when candidate is located inside root."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False

    return True


def _build_image_candidates(
    source_reference: str,
    camera_name: str,
    image_root: Path,
) -> tuple[Path, ...]:
    """
    Build possible local paths for one DriveLM image reference.

    The DriveLM JSON may contain references such as:

        nuscenes/samples/CAM_FRONT/image.jpg
        samples/CAM_FRONT/image.jpg
        CAM_FRONT/image.jpg
        image.jpg

    All candidates are normalized against image_root.
    """
    referenced_path = Path(source_reference)
    candidates: list[Path] = []

    if referenced_path.is_absolute():
        candidates.append(referenced_path.resolve())

    else:
        # Works when the JSON already stores CAM_FRONT/image.jpg
        # or another path relative to the supplied image root.
        candidates.append(
            (image_root / referenced_path).resolve()
        )

        # Remove any source-specific prefixes before the camera folder.
        #
        # Example:
        # nuscenes/samples/CAM_FRONT/image.jpg
        # becomes:
        # image_root/CAM_FRONT/image.jpg
        reference_parts = referenced_path.parts

        if camera_name in reference_parts:
            camera_position = reference_parts.index(camera_name)
            camera_relative_path = Path(
                *reference_parts[camera_position:]
            )

            candidates.append(
                (image_root / camera_relative_path).resolve()
            )

        # Final fallback when only the image filename is useful.
        candidates.append(
            (
                image_root
                / camera_name
                / referenced_path.name
            ).resolve()
        )

    unique_candidates: list[Path] = []
    seen_candidates: set[Path] = set()

    for candidate in candidates:
        if candidate in seen_candidates:
            continue

        # Do not allow a relative dataset reference to escape image_root.
        if (
            not referenced_path.is_absolute()
            and not _is_within_root(candidate, image_root)
        ):
            continue

        seen_candidates.add(candidate)
        unique_candidates.append(candidate)

    return tuple(unique_candidates)


def resolve_drivelm_image_paths(
    annotations: DriveLMAnnotations,
    image_root: str | Path,
) -> DriveLMImagePathResolution:
    """
    Resolve DriveLM camera references against a local image root.

    Args:
        annotations:
            Raw DriveLM annotations returned by
            load_drivelm_annotations().

        image_root:
            Directory containing the camera folders. For training data:

                data/drivelm/nuscenes/samples

            For official validation data:

                data/drivelm/val_data

    Returns:
        Resolved absolute image paths and separately recorded unresolved
        references.

    Raises:
        DriveLMImageResolutionError:
            If the image root is invalid or image-reference metadata is
            malformed.
    """
    normalized_image_root = (
        Path(image_root)
        .expanduser()
        .resolve()
    )

    if not normalized_image_root.is_dir():
        raise DriveLMImageResolutionError(
            "DriveLM image root does not exist or is not a directory:\n"
            f"  {normalized_image_root}"
        )

    resolved: dict[
        tuple[str, str, str],
        ResolvedDriveLMImage,
    ] = {}

    unresolved: list[UnresolvedDriveLMImage] = []
    reference_count = 0

    for scene_token in sorted(annotations.scenes):
        scene_data = annotations.scenes[scene_token]
        key_frames = scene_data.get("key_frames")

        if not isinstance(key_frames, dict):
            raise DriveLMImageResolutionError(
                f"Scene {scene_token!r} is missing a valid "
                "'key_frames' mapping."
            )

        for frame_token in sorted(key_frames):
            frame_data = key_frames[frame_token]

            if not isinstance(frame_data, dict):
                raise DriveLMImageResolutionError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} must be a mapping."
                )

            image_paths = frame_data.get("image_paths")

            if not isinstance(image_paths, dict):
                raise DriveLMImageResolutionError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} is missing a valid "
                    "'image_paths' mapping."
                )

            for camera_name in sorted(image_paths):
                source_reference = image_paths[camera_name]
                reference_count += 1

                if (
                    not isinstance(camera_name, str)
                    or not camera_name.strip()
                ):
                    raise DriveLMImageResolutionError(
                        f"Invalid camera name for scene={scene_token!r}, "
                        f"frame={frame_token!r}: {camera_name!r}"
                    )

                if (
                    not isinstance(source_reference, str)
                    or not source_reference.strip()
                ):
                    unresolved.append(
                        UnresolvedDriveLMImage(
                            scene_token=scene_token,
                            frame_token=frame_token,
                            camera_name=camera_name,
                            source_reference=str(source_reference),
                            candidate_paths=(),
                            reason=(
                                "Image reference is not a non-empty string."
                            ),
                        )
                    )
                    continue

                candidate_paths = _build_image_candidates(
                    source_reference=source_reference,
                    camera_name=camera_name,
                    image_root=normalized_image_root,
                )

                matched_path = next(
                    (
                        candidate
                        for candidate in candidate_paths
                        if candidate.is_file()
                    ),
                    None,
                )

                if matched_path is None:
                    unresolved.append(
                        UnresolvedDriveLMImage(
                            scene_token=scene_token,
                            frame_token=frame_token,
                            camera_name=camera_name,
                            source_reference=source_reference,
                            candidate_paths=candidate_paths,
                            reason=(
                                "No candidate path points to an "
                                "existing file."
                            ),
                        )
                    )
                    continue

                reference_key = (
                    scene_token,
                    frame_token,
                    camera_name,
                )

                if reference_key in resolved:
                    raise DriveLMImageResolutionError(
                        "Duplicate image-reference key encountered: "
                        f"{reference_key!r}"
                    )

                resolved[reference_key] = ResolvedDriveLMImage(
                    scene_token=scene_token,
                    frame_token=frame_token,
                    camera_name=camera_name,
                    source_reference=source_reference,
                    absolute_path=matched_path,
                )

    return DriveLMImagePathResolution(
        image_root=normalized_image_root,
        resolved=resolved,
        unresolved=tuple(unresolved),
        reference_count=reference_count,
        resolved_count=len(resolved),
        unresolved_count=len(unresolved),
    )

def validate_drivelm_images(
    annotations: DriveLMAnnotations,
    resolution: DriveLMImagePathResolution,
    *,
    required_camera_names: tuple[str, ...],
    expected_dimensions: tuple[int, int] | None = None,
    max_missing_fraction: float = 0.0,
    fail_on_duplicate_paths: bool = True,
) -> DriveLMImageValidationReport:
    """
    Validate resolved DriveLM images and produce a quality report.

    Validation covers:

    1. Required camera views declared for every frame.
    2. Resolution of each declared image reference.
    3. File existence.
    4. Image decoding and readability.
    5. Expected image dimensions.
    6. Agreement between the declared camera and camera directory.
    7. Duplicate physical paths referenced by multiple records.

    Args:
        annotations:
            Raw DriveLM annotations loaded by Function 013.

        resolution:
            Image-path resolution produced by Function 015.

        required_camera_names:
            Camera views that every frame is expected to contain.

        expected_dimensions:
            Expected image dimensions as ``(width, height)``.
            Pass None to accept any positive dimensions.

        max_missing_fraction:
            Maximum permitted fraction of required camera views that
            are absent or unresolved.

        fail_on_duplicate_paths:
            Whether duplicate physical image paths should make the
            validation fail.

    Returns:
        A complete image-quality report.

    Raises:
        ValueError:
            If configuration values are invalid.

        DriveLMImageValidationError:
            If the completed report violates the configured quality
            requirements.
    """
    if not required_camera_names:
        raise ValueError(
            "required_camera_names must contain at least one camera."
        )

    if len(set(required_camera_names)) != len(required_camera_names):
        raise ValueError(
            "required_camera_names contains duplicate camera names."
        )

    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError(
            "max_missing_fraction must be between 0.0 and 1.0."
        )

    if expected_dimensions is not None:
        expected_width, expected_height = expected_dimensions

        if expected_width <= 0 or expected_height <= 0:
            raise ValueError(
                "expected_dimensions must contain positive values."
            )

    required_cameras = set(required_camera_names)

    issues: list[DriveLMImageIssue] = []
    validated_images: dict[
        tuple[str, str, str],
        ValidatedDriveLMImage,
    ] = {}

    invalid_reference_keys: set[
        tuple[str, str, str]
    ] = set()

    path_to_reference_keys: dict[
        Path,
        list[tuple[str, str, str]],
    ] = defaultdict(list)

    dimension_counts: dict[str, int] = defaultdict(int)
    format_counts: dict[str, int] = defaultdict(int)

    frame_count = 0
    missing_required_view_count = 0

    # ---------------------------------------------------------
    # Stage 1: detect camera views absent from the source JSON.
    # ---------------------------------------------------------
    for scene_token in sorted(annotations.scenes):
        scene_data = annotations.scenes[scene_token]
        key_frames = scene_data.get("key_frames")

        if not isinstance(key_frames, dict):
            raise ValueError(
                f"Scene {scene_token!r} does not contain a valid "
                "'key_frames' mapping."
            )

        for frame_token in sorted(key_frames):
            frame_count += 1
            frame_data = key_frames[frame_token]

            if not isinstance(frame_data, dict):
                raise ValueError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} must be a mapping."
                )

            image_paths = frame_data.get("image_paths")

            if not isinstance(image_paths, dict):
                raise ValueError(
                    f"Frame {frame_token!r} in scene "
                    f"{scene_token!r} does not contain a valid "
                    "'image_paths' mapping."
                )

            declared_cameras = set(image_paths)
            missing_cameras = (
                required_cameras - declared_cameras
            )

            for camera_name in sorted(missing_cameras):
                missing_required_view_count += 1

                issues.append(
                    DriveLMImageIssue(
                        issue_type="missing_required_view",
                        scene_token=scene_token,
                        frame_token=frame_token,
                        camera_name=camera_name,
                        absolute_path=None,
                        detail=(
                            "Required camera view is absent from the "
                            "frame's image_paths mapping."
                        ),
                    )
                )

    # ---------------------------------------------------------
    # Stage 2: carry unresolved Function 015 references forward.
    # ---------------------------------------------------------
    for unresolved_image in resolution.unresolved:
        reference_key = (
            unresolved_image.scene_token,
            unresolved_image.frame_token,
            unresolved_image.camera_name,
        )

        invalid_reference_keys.add(reference_key)

        issues.append(
            DriveLMImageIssue(
                issue_type="unresolved_reference",
                scene_token=unresolved_image.scene_token,
                frame_token=unresolved_image.frame_token,
                camera_name=unresolved_image.camera_name,
                absolute_path=None,
                detail=(
                    f"{unresolved_image.reason} "
                    f"Source reference: "
                    f"{unresolved_image.source_reference!r}."
                ),
            )
        )

    # ---------------------------------------------------------
    # Stage 3: open and decode every resolved image.
    # ---------------------------------------------------------
    unreadable_image_count = 0
    dimension_mismatch_count = 0
    camera_mismatch_count = 0

    for reference_key in sorted(resolution.resolved):
        resolved_image = resolution.resolved[reference_key]

        image_path = resolved_image.absolute_path
        path_to_reference_keys[image_path].append(reference_key)

        if not image_path.is_file():
            unreadable_image_count += 1
            invalid_reference_keys.add(reference_key)

            issues.append(
                DriveLMImageIssue(
                    issue_type="missing_file",
                    scene_token=resolved_image.scene_token,
                    frame_token=resolved_image.frame_token,
                    camera_name=resolved_image.camera_name,
                    absolute_path=image_path,
                    detail=(
                        "The file existed during path resolution but "
                        "is no longer present."
                    ),
                )
            )
            continue

        # The resolved training layout should end with:
        #
        # CAM_FRONT/<image filename>
        #
        # Therefore, the immediate parent directory is expected to
        # equal the camera name declared by the JSON.
        actual_camera_directory = image_path.parent.name

        if actual_camera_directory != resolved_image.camera_name:
            camera_mismatch_count += 1
            invalid_reference_keys.add(reference_key)

            issues.append(
                DriveLMImageIssue(
                    issue_type="camera_identity_mismatch",
                    scene_token=resolved_image.scene_token,
                    frame_token=resolved_image.frame_token,
                    camera_name=resolved_image.camera_name,
                    absolute_path=image_path,
                    detail=(
                        "Declared camera is "
                        f"{resolved_image.camera_name!r}, but the "
                        "file is stored under directory "
                        f"{actual_camera_directory!r}."
                    ),
                )
            )

        try:
            # verify() checks the image container for structural
            # corruption without retaining decoded pixel data.
            with Image.open(image_path) as image:
                image_format = image.format or "UNKNOWN"
                width, height = image.size
                image.verify()

            # Reopen after verify(), then load() to force actual pixel
            # decoding. A file can have a readable header but still
            # contain truncated or invalid image data.
            with Image.open(image_path) as image:
                image.load()
                image_mode = image.mode

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as exc:
            unreadable_image_count += 1
            invalid_reference_keys.add(reference_key)

            issues.append(
                DriveLMImageIssue(
                    issue_type="unreadable_image",
                    scene_token=resolved_image.scene_token,
                    frame_token=resolved_image.frame_token,
                    camera_name=resolved_image.camera_name,
                    absolute_path=image_path,
                    detail=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue

        dimension_key = f"{width}x{height}"
        dimension_counts[dimension_key] += 1
        format_counts[image_format] += 1

        if (
            expected_dimensions is not None
            and (width, height) != expected_dimensions
        ):
            dimension_mismatch_count += 1
            invalid_reference_keys.add(reference_key)

            issues.append(
                DriveLMImageIssue(
                    issue_type="dimension_mismatch",
                    scene_token=resolved_image.scene_token,
                    frame_token=resolved_image.frame_token,
                    camera_name=resolved_image.camera_name,
                    absolute_path=image_path,
                    detail=(
                        f"Expected {expected_dimensions[0]}x"
                        f"{expected_dimensions[1]}, but received "
                        f"{width}x{height}."
                    ),
                )
            )

        validated_images[reference_key] = ValidatedDriveLMImage(
            scene_token=resolved_image.scene_token,
            frame_token=resolved_image.frame_token,
            camera_name=resolved_image.camera_name,
            absolute_path=image_path,
            width=width,
            height=height,
            image_format=image_format,
            image_mode=image_mode,
        )

    # ---------------------------------------------------------
    # Stage 4: detect reuse of one path by multiple references.
    # ---------------------------------------------------------
    duplicate_path_groups = {
        image_path: tuple(reference_keys)
        for image_path, reference_keys
        in path_to_reference_keys.items()
        if len(reference_keys) > 1
    }

    duplicate_reference_count = sum(
        len(reference_keys) - 1
        for reference_keys in duplicate_path_groups.values()
    )

    for image_path, reference_keys in sorted(
        duplicate_path_groups.items(),
        key=lambda item: str(item[0]),
    ):
        if fail_on_duplicate_paths:
            invalid_reference_keys.update(reference_keys)

        first_reference = reference_keys[0]

        issues.append(
            DriveLMImageIssue(
                issue_type="duplicate_image_path",
                scene_token=first_reference[0],
                frame_token=first_reference[1],
                camera_name=first_reference[2],
                absolute_path=image_path,
                detail=(
                    "The same physical file is referenced by "
                    f"{len(reference_keys)} scene/frame/camera keys."
                ),
            )
        )

    # ---------------------------------------------------------
    # Stage 5: calculate missing-view tolerance.
    # ---------------------------------------------------------
    expected_required_view_count = (
        frame_count * len(required_camera_names)
    )

    unresolved_required_count = sum(
        1
        for unresolved_image in resolution.unresolved
        if unresolved_image.camera_name in required_cameras
    )

    required_view_failure_count = (
        missing_required_view_count
        + unresolved_required_count
    )

    if expected_required_view_count == 0:
        required_view_failure_fraction = 0.0
    else:
        required_view_failure_fraction = (
            required_view_failure_count
            / expected_required_view_count
        )

    strict_quality_failure = any(
        (
            unreadable_image_count > 0,
            dimension_mismatch_count > 0,
            camera_mismatch_count > 0,
            (
                fail_on_duplicate_paths
                and bool(duplicate_path_groups)
            ),
        )
    )

    missing_tolerance_failure = (
        required_view_failure_fraction
        > max_missing_fraction
    )

    passed = not (
        strict_quality_failure
        or missing_tolerance_failure
    )

    valid_image_count = (
        resolution.resolved_count
        - len(invalid_reference_keys)
    )

    report = DriveLMImageValidationReport(
        frame_count=frame_count,
        reference_count=resolution.reference_count,
        expected_required_view_count=(
            expected_required_view_count
        ),
        validated_images=validated_images,
        valid_image_count=valid_image_count,
        missing_required_view_count=(
            missing_required_view_count
        ),
        unresolved_reference_count=(
            resolution.unresolved_count
        ),
        unreadable_image_count=unreadable_image_count,
        dimension_mismatch_count=dimension_mismatch_count,
        camera_mismatch_count=camera_mismatch_count,
        duplicate_path_group_count=len(
            duplicate_path_groups
        ),
        duplicate_reference_count=(
            duplicate_reference_count
        ),
        required_view_failure_fraction=(
            required_view_failure_fraction
        ),
        dimension_counts=dict(
            sorted(dimension_counts.items())
        ),
        format_counts=dict(
            sorted(format_counts.items())
        ),
        issues=tuple(issues),
        passed=passed,
    )

    if not passed:
        raise DriveLMImageValidationError(
            "DriveLM image validation failed. "
            f"issues={len(report.issues)}, "
            "required-view failure fraction="
            f"{report.required_view_failure_fraction:.6f}, "
            f"allowed={max_missing_fraction:.6f}.",
            report=report,
        )

    return report

def _print_image_validation_report(
    report: DriveLMImageValidationReport,
) -> None:
    """Print a concise DriveLM image-quality report."""
    print("DriveLM image validation report")
    print()
    print(f"Passed:                       {report.passed}")
    print(f"Frames examined:              {report.frame_count:,}")
    print(
        f"Image references:             "
        f"{report.reference_count:,}"
    )
    print(
        f"Expected required views:      "
        f"{report.expected_required_view_count:,}"
    )
    print(
        f"Valid images:                 "
        f"{report.valid_image_count:,}"
    )
    print(
        f"Missing required views:       "
        f"{report.missing_required_view_count:,}"
    )
    print(
        f"Unresolved references:        "
        f"{report.unresolved_reference_count:,}"
    )
    print(
        f"Unreadable images:            "
        f"{report.unreadable_image_count:,}"
    )
    print(
        f"Dimension mismatches:         "
        f"{report.dimension_mismatch_count:,}"
    )
    print(
        f"Camera mismatches:            "
        f"{report.camera_mismatch_count:,}"
    )
    print(
        f"Duplicate path groups:        "
        f"{report.duplicate_path_group_count:,}"
    )
    print(
        f"Duplicate extra references:   "
        f"{report.duplicate_reference_count:,}"
    )
    print(
        f"Required-view failure rate:   "
        f"{report.required_view_failure_fraction:.6%}"
    )

    print()
    print("Dimension distribution:")

    for dimensions, count in report.dimension_counts.items():
        print(f"  {dimensions}: {count:,}")

    print()
    print("Image format distribution:")

    for image_format, count in report.format_counts.items():
        print(f"  {image_format}: {count:,}")

    if report.issues:
        print()
        print("First image-quality issues:")

        for issue in report.issues[:10]:
            print()
            print(f"  Type:    {issue.issue_type}")
            print(
                f"  Source:  scene={issue.scene_token}, "
                f"frame={issue.frame_token}, "
                f"camera={issue.camera_name}"
            )
            print(f"  Path:    {issue.absolute_path}")
            print(f"  Detail:  {issue.detail}")
            
            
            

@dataclass(frozen=True, slots=True)
class DriveLMImageIssue:
    """One image-quality problem discovered during validation."""

    issue_type: str
    scene_token: str
    frame_token: str
    camera_name: str
    absolute_path: Path | None
    detail: str


@dataclass(frozen=True, slots=True)
class ValidatedDriveLMImage:
    """Metadata collected from one successfully decoded image."""

    scene_token: str
    frame_token: str
    camera_name: str
    absolute_path: Path
    width: int
    height: int
    image_format: str
    image_mode: str


@dataclass(frozen=True, slots=True)
class DriveLMImageValidationReport:
    """Aggregate data-quality report for DriveLM camera images."""

    frame_count: int
    reference_count: int
    expected_required_view_count: int

    validated_images: dict[
        tuple[str, str, str],
        ValidatedDriveLMImage,
    ]

    valid_image_count: int
    missing_required_view_count: int
    unresolved_reference_count: int
    unreadable_image_count: int
    dimension_mismatch_count: int
    camera_mismatch_count: int
    duplicate_path_group_count: int
    duplicate_reference_count: int

    required_view_failure_fraction: float
    dimension_counts: dict[str, int]
    format_counts: dict[str, int]
    issues: tuple[DriveLMImageIssue, ...]
    passed: bool


class DriveLMImageValidationError(ValueError):
    """Raised when DriveLM image quality fails configured requirements."""

    def __init__(
        self,
        message: str,
        report: DriveLMImageValidationReport,
    ) -> None:
        super().__init__(message)
        self.report = report
        
        
        
        
def main() -> None:
    """Resolve and validate DriveLM training images using F5."""
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

    annotations = load_drivelm_annotations(annotation_path)

    resolution = resolve_drivelm_image_paths(
        annotations=annotations,
        image_root=training_image_root,
    )

    required_cameras = (
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    )

    try:
        report = validate_drivelm_images(
            annotations=annotations,
            resolution=resolution,
            required_camera_names=required_cameras,
            expected_dimensions=(1600, 900),
            max_missing_fraction=0.0,
            fail_on_duplicate_paths=True,
        )

    except DriveLMImageValidationError as exc:
        _print_image_validation_report(exc.report)
        raise

    _print_image_validation_report(report)


if __name__ == "__main__":
    main()