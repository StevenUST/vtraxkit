"""COCO keypoint format export."""

from __future__ import annotations

import json
from pathlib import Path

from ..data.collection import TrackCollection


def save_coco(collection: TrackCollection, path: str | Path) -> None:
    """Save tracks in COCO keypoint annotation format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    annotations = []
    ann_id = 1

    for track in collection.tracks:
        for i, (frame_idx, bbox, skeleton) in enumerate(
            zip(track.frames, track.bboxes, track.skeletons)
        ):
            if skeleton is None:
                continue

            pose = skeleton.pose
            if pose is None:
                continue

            # COCO format: [x1, y1, v1, x2, y2, v2, ...]
            keypoints = []
            num_visible = 0
            for kp in pose:
                x, y = float(kp[0]), float(kp[1])
                v = int(kp[3] > 0.5) * 2 if len(kp) > 3 else 2
                keypoints.extend([x, y, v])
                if v > 0:
                    num_visible += 1

            annotations.append({
                "id": ann_id,
                "track_id": track.track_id,
                "image_id": frame_idx,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": num_visible,
                "bbox": list(bbox),
                "area": bbox.area,
            })
            ann_id += 1

    output = {
        "info": {"description": "Exported by vtraxkit"},
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle",
                *[f"pose_{i}" for i in range(17, 33)],
            ],
        }],
        "annotations": annotations,
        "video_metadata": collection.video_metadata,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
