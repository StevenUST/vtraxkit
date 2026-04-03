"""NumPy .npy serialization for tracks.

Supports both the new vtraxkit format and the legacy VideoScreener format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data.bbox import BBox
from ..data.collection import TrackCollection
from ..data.skeleton import Skeleton
from ..data.track import Track


def save_npy(collection: TrackCollection, path: str | Path) -> None:
    """Save a TrackCollection to a .npy file.

    Format: dict with "version", "video_metadata", "tracks".
    Each track contains frame_indices, timestamps, bboxes, and skeleton keypoints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tracks_data = []
    for track in collection.tracks:
        skeletons = []
        for s in track.skeletons:
            if s is not None:
                skeletons.append(s.to_dict())
            else:
                skeletons.append(None)

        tracks_data.append({
            "track_id": track.track_id,
            "frames": np.array(track.frames, dtype=np.int32),
            "timestamps": np.array(track.timestamps, dtype=np.float32),
            "bboxes": np.array(track.bboxes, dtype=np.int32),
            "skeletons": skeletons,
            "metadata": track.metadata,
        })

    data = {
        "version": "vtraxkit-v1",
        "video_metadata": collection.video_metadata,
        "tracks": tracks_data,
    }
    np.save(str(path), data, allow_pickle=True)


def load_npy(path: str | Path) -> TrackCollection:
    """Load a TrackCollection from a .npy file.

    Supports both vtraxkit-v1 format and legacy VideoScreener format.
    """
    path = Path(path)
    raw = np.load(str(path), allow_pickle=True)

    # np.save wraps dicts in a 0-d array
    if raw.ndim == 0:
        raw = raw.item()

    if isinstance(raw, dict) and raw.get("version") == "vtraxkit-v1":
        return _load_v1(raw)

    # Legacy format: list of frame dicts
    if isinstance(raw, (list, np.ndarray)):
        return _load_legacy(raw, source=str(path))

    raise ValueError(f"Unrecognized .npy format in {path}")


def _load_v1(data: dict) -> TrackCollection:
    tracks = []
    for td in data["tracks"]:
        frames = td["frames"].tolist()
        timestamps = td["timestamps"].tolist()
        bboxes = [BBox(*b) for b in td["bboxes"].tolist()]
        skeletons = []
        for s in td["skeletons"]:
            if s is not None:
                skeletons.append(Skeleton(
                    {k: np.asarray(v, dtype=np.float32) for k, v in s.items()}
                ))
            else:
                skeletons.append(None)

        tracks.append(Track(
            track_id=td["track_id"],
            frames=frames,
            timestamps=timestamps,
            bboxes=bboxes,
            skeletons=skeletons,
            metadata=td.get("metadata", {}),
        ))

    return TrackCollection(
        tracks=tracks,
        video_metadata=data.get("video_metadata", {}),
    )


def _load_legacy(frames_data: list[dict[str, Any]], source: str = "") -> TrackCollection:
    """Load from VideoScreener's legacy format.

    Legacy format is a flat list of frame dicts, all belonging to a single person::

        [{"frame_idx": int, "timestamp": float,
          "pose": ndarray(33,4), "left_hand": ndarray(21,4), ...}, ...]
    """
    if len(frames_data) == 0:
        return TrackCollection()

    # All frames belong to track_id=1
    track = Track(track_id=1)
    for fd in frames_data:
        frame_idx = int(fd.get("frame_idx", 0))
        timestamp = float(fd.get("timestamp", 0.0))
        # Legacy format has no bboxes
        track.frames.append(frame_idx)
        track.timestamps.append(timestamp)
        track.bboxes.append(BBox(0, 0, 0, 0))
        track.skeletons.append(Skeleton.from_legacy_dict(fd))

    return TrackCollection(
        tracks=[track],
        video_metadata={"source": source, "format": "legacy-videoscreener"},
    )
