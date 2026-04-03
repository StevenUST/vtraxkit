"""Single-frame keypoint container."""

from __future__ import annotations

from typing import Any

import numpy as np


class Skeleton:
    """Keypoints for one person in one frame.

    Stores named groups of keypoints, e.g.::

        {"pose": ndarray(33, 4), "left_hand": ndarray(21, 4), ...}

    Each group is a float32 array of shape (num_keypoints, num_dims).
    """

    __slots__ = ("keypoints",)

    def __init__(self, keypoints: dict[str, np.ndarray] | None = None):
        self.keypoints: dict[str, np.ndarray] = keypoints or {}

    # -- convenience accessors --------------------------------------------------

    @property
    def pose(self) -> np.ndarray | None:
        return self.keypoints.get("pose")

    @property
    def left_hand(self) -> np.ndarray | None:
        return self.keypoints.get("left_hand")

    @property
    def right_hand(self) -> np.ndarray | None:
        return self.keypoints.get("right_hand")

    @property
    def face(self) -> np.ndarray | None:
        return self.keypoints.get("face")

    # -- conversions ------------------------------------------------------------

    def to_flat_array(self) -> np.ndarray:
        """Concatenate all groups into a single (N, D) array."""
        if not self.keypoints:
            return np.empty((0, 0), dtype=np.float32)
        arrays = list(self.keypoints.values())
        max_dims = max(a.shape[1] for a in arrays)
        padded = []
        for a in arrays:
            if a.shape[1] < max_dims:
                pad = np.zeros((a.shape[0], max_dims - a.shape[1]), dtype=np.float32)
                a = np.concatenate([a, pad], axis=1)
            padded.append(a)
        return np.concatenate(padded, axis=0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for serialization)."""
        return {name: arr.copy() for name, arr in self.keypoints.items()}

    @classmethod
    def from_legacy_dict(cls, d: dict[str, Any]) -> Skeleton:
        """Construct from the VideoScreener .npy frame dict format.

        Expected keys: pose (33,4), left_hand (21,4), right_hand (21,4), face (468,3).
        """
        kps = {}
        for key in ("pose", "left_hand", "right_hand", "face"):
            if key in d and d[key] is not None:
                arr = np.asarray(d[key], dtype=np.float32)
                if arr.size > 0:
                    kps[key] = arr
        return cls(kps)

    def __repr__(self) -> str:
        parts = [f"{k}: ({v.shape[0]},{v.shape[1]})" for k, v in self.keypoints.items()]
        return f"Skeleton({', '.join(parts)})"
