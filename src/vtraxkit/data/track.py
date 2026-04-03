"""Single-person trajectory across frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bbox import BBox
from .skeleton import Skeleton


@dataclass
class Track:
    """One tracked person's trajectory.

    Attributes:
        track_id:   Unique ID assigned during tracking.
        frames:     Frame indices where this person was detected.
        timestamps: Corresponding timestamps in seconds.
        bboxes:     Bounding boxes per frame.
        skeletons:  Skeleton keypoints per frame (None before pose estimation).
        metadata:   Arbitrary metadata (e.g. age classification results).
    """

    track_id: int
    frames: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    bboxes: list[BBox] = field(default_factory=list)
    skeletons: list[Skeleton | None] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_frame(self, frame_idx: int, timestamp: float, bbox: BBox) -> None:
        self.frames.append(frame_idx)
        self.timestamps.append(timestamp)
        self.bboxes.append(bbox)
        self.skeletons.append(None)

    # -- properties -------------------------------------------------------------

    @property
    def start_frame(self) -> int:
        return self.frames[0] if self.frames else 0

    @property
    def end_frame(self) -> int:
        return self.frames[-1] if self.frames else 0

    @property
    def start_time(self) -> float:
        return self.timestamps[0] if self.timestamps else 0.0

    @property
    def end_time(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def has_skeletons(self) -> bool:
        return any(s is not None for s in self.skeletons)

    # -- array helpers ----------------------------------------------------------

    def skeleton_array(self, group: str = "pose") -> np.ndarray:
        """Stack a keypoint group across all frames.

        Returns:
            Array of shape (T, K, D) where T=num_frames, K=num_keypoints, D=dims.
            Frames without skeletons are filled with zeros.
        """
        arrays = []
        shape = None
        for s in self.skeletons:
            if s is not None and group in s.keypoints:
                arr = s.keypoints[group]
                if shape is None:
                    shape = arr.shape
                arrays.append(arr)
            else:
                arrays.append(None)

        if shape is None:
            return np.empty((0, 0, 0), dtype=np.float32)

        result = np.zeros((len(arrays), *shape), dtype=np.float32)
        for i, arr in enumerate(arrays):
            if arr is not None:
                result[i] = arr
        return result

    def bbox_centers(self) -> np.ndarray:
        """Return array of bbox centers, shape (T, 2)."""
        return np.array([b.center for b in self.bboxes], dtype=np.float32)

    def __repr__(self) -> str:
        skel = " +skel" if self.has_skeletons else ""
        return (
            f"Track(id={self.track_id}, frames={self.num_frames}, "
            f"duration={self.duration:.1f}s{skel})"
        )
