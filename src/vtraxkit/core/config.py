"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline.

    All parameters have sensible defaults so you can start with just::

        config = PipelineConfig()

    Attributes:
        device:                   "cpu" or "cuda".
        frame_skip:               Process every N-th frame for detection/tracking.
        scene_change_threshold:   Histogram correlation below this triggers a scene reset.
        max_missing_frames:       Track dies after this many consecutive frames without detection.
        min_confidence:           Minimum detection confidence.
        min_track_duration:       Discard tracks shorter than this (seconds).
        motion_threshold:         Minimum bbox center std-dev to keep a track (pixels).
                                  Set to None to disable motion filtering.
        pose_min_detection_conf:  MediaPipe min_detection_confidence.
        pose_min_tracking_conf:   MediaPipe min_tracking_confidence.
    """

    device: str = "cpu"
    frame_skip: int = 3
    scene_change_threshold: float = 0.4
    max_missing_frames: int = 60
    min_confidence: float = 0.4
    min_track_duration: float = 1.0
    motion_threshold: float | None = 5.0
    pose_min_detection_conf: float = 0.5
    pose_min_tracking_conf: float = 0.5
