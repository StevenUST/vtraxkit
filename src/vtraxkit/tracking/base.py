"""Abstract base for tracking backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..detection.base import Detection


class TrackedDetection(Detection):
    """A detection with an assigned persistent track ID."""
    pass


class TrackerBackend(ABC):
    """Abstract base for multi-person tracking backends."""

    @abstractmethod
    def update(self, detections: list[Detection], frame_idx: int) -> list[TrackedDetection]:
        """Associate detections across frames and assign track IDs."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all active tracks (e.g. on scene change)."""
