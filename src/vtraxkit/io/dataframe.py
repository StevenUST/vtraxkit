"""Pandas DataFrame conversion."""

from __future__ import annotations

from ..data.collection import TrackCollection


def to_dataframe(collection: TrackCollection):
    """Convert a TrackCollection to a pandas DataFrame.

    Columns: track_id, frame_idx, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
             and one column per keypoint coordinate (e.g. pose_0_x, pose_0_y, ...).

    Requires ``pip install vtraxkit[pandas]``.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install vtraxkit[pandas]"
        )

    rows = []
    for track in collection.tracks:
        for i, (frame_idx, ts, bbox, skel) in enumerate(
            zip(track.frames, track.timestamps, track.bboxes, track.skeletons)
        ):
            row = {
                "track_id": track.track_id,
                "frame_idx": frame_idx,
                "timestamp": ts,
                "bbox_x": bbox.x,
                "bbox_y": bbox.y,
                "bbox_w": bbox.w,
                "bbox_h": bbox.h,
            }

            if skel is not None:
                for group_name, arr in skel.keypoints.items():
                    for kp_idx in range(arr.shape[0]):
                        for d, dim_name in enumerate(("x", "y", "z", "v")):
                            if d < arr.shape[1]:
                                row[f"{group_name}_{kp_idx}_{dim_name}"] = float(arr[kp_idx, d])

            rows.append(row)

    return pd.DataFrame(rows)
