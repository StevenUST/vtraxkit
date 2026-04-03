"""Abstract base for pose estimation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..data.bbox import BBox
from ..data.skeleton import Skeleton


@dataclass
class KeypointLayout:
    """Describes the keypoint schema for a pose backend."""

    groups: dict[str, int]  # group_name -> num_keypoints, e.g. {"pose": 33, "left_hand": 21}
    dims: dict[str, int]    # group_name -> num_dims, e.g. {"pose": 4, "face": 3}


class PoseBackend(ABC):
    """Abstract base for pose estimation backends."""

    @abstractmethod
    def estimate(self, frame: np.ndarray, bbox: BBox | None = None) -> Skeleton:
        """Extract keypoints from a frame, optionally cropped to bbox."""

    @property
    @abstractmethod
    def layout(self) -> KeypointLayout:
        """Describe the keypoint schema."""

    def close(self) -> None:
        """Release resources."""
