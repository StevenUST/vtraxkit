"""Abstract base for detection backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.bbox import BBox


class Detection:
    """A single detection result."""

    __slots__ = ("bbox", "confidence", "track_id")

    def __init__(self, bbox: BBox, confidence: float, track_id: int | None = None):
        self.bbox = bbox
        self.confidence = confidence
        self.track_id = track_id


class DetectorBackend(ABC):
    """Abstract base for person detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """Return person detections for a single frame."""

    def reset(self) -> None:
        """Reset internal state (e.g. on scene change)."""

    def close(self) -> None:
        """Release resources."""
