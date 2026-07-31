from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from drivelm_align.data.grouping import (
    DriveLMSceneRecordGroup,
)


CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


def render_drivelm_multiview_scene(
    group: DriveLMSceneRecordGroup,
    *,
    frame_token: str,
    output_path: str | Path,
) -> Path:
    """Render one synchronized frame with identifiers and object boxes."""
    import matplotlib.pyplot as plt

    if frame_token not in group.frame_tokens:
        raise ValueError(
            f"Frame {frame_token!r} does not belong to "
            f"scene {group.scene_token!r}."
        )

    images_by_camera = {
        record.camera_name: record
        for record in group.image_records
        if record.frame_token == frame_token
    }

    objects_by_camera = {
        camera_name: [
            record
            for record in group.object_records
            if record.frame_token == frame_token
            and record.camera_name == camera_name
        ]
        for camera_name in CAMERA_ORDER
    }

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    for axis, camera_name in zip(
        axes.flat,
        CAMERA_ORDER,
    ):
        image_record = images_by_camera.get(camera_name)

        if image_record is None:
            axis.text(
                0.5,
                0.5,
                "Missing image",
                ha="center",
                va="center",
            )
            axis.set_title(camera_name)
            axis.axis("off")
            continue

        with Image.open(
            image_record.absolute_path
        ) as source_image:
            image = source_image.convert("RGB")

        draw = ImageDraw.Draw(image)

        for object_record in objects_by_camera[
            camera_name
        ]:
            x_min, y_min, x_max, y_max = (
                object_record.bbox_xyxy
            )

            box = (
                round(x_min),
                round(y_min),
                round(x_max),
                round(y_max),
            )

            draw.rectangle(
                box,
                outline="red",
                width=4,
            )

            label = (
                f"{object_record.object_id}: "
                f"{object_record.category}"
            )

            if object_record.status:
                label += f" ({object_record.status})"

            draw.text(
                (box[0] + 4, box[1] + 4),
                label,
                fill="red",
            )

        axis.imshow(image)
        axis.set_title(
            f"{camera_name} "
            f"({len(objects_by_camera[camera_name])} objects)"
        )
        axis.axis("off")

    figure.suptitle(
        f"Scene: {group.scene_token}\n"
        f"Frame: {frame_token}",
        fontsize=12,
    )

    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main() -> None:
    """Render one real multiview frame to a temporary file using F5."""
    from drivelm_align.data._debug import build_debug_grouping

    grouping = build_debug_grouping(scene_count=1)
    group = next(iter(grouping.groups.values()))
    frame_token = group.frame_tokens[0]

    with TemporaryDirectory(prefix="drivelm-scene-debug-") as directory:
        os.environ.setdefault("MPLCONFIGDIR", directory)
        output_path = render_drivelm_multiview_scene(
            group,
            frame_token=frame_token,
            output_path=Path(directory) / "multiview.png",
        )
        print(f"Rendered multiview figure: {output_path}")


if __name__ == "__main__":
    main()
