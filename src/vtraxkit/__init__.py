"""Vtraxkit: Extract multi-person skeleton trajectories from videos.

Usage::

    import vtraxkit

    # One-liner
    tracks = vtraxkit.extract("video.mp4")

    # With options
    tracks = vtraxkit.extract("video.mp4", device="cuda", detector="yolo:yolov8s.pt")

    # Filter + save
    tracks.filter(min_duration=2.0).save("output.npy")

    # Load previously saved tracks
    tracks = vtraxkit.load("output.npy")
"""

from ._version import __version__
from .core.config import PipelineConfig
from .core.pipeline import Pipeline
from .data.collection import TrackCollection
from .data.skeleton import Skeleton
from .data.track import Track

__all__ = [
    "__version__",
    "extract",
    "load",
    "Pipeline",
    "PipelineConfig",
    "Track",
    "TrackCollection",
    "Skeleton",
]


def extract(
    source: str,
    *,
    device: str = "cpu",
    detector: str = "yolo",
    pose: str = "mediapipe",
    frame_skip: int = 3,
    min_duration: float = 1.0,
    show_progress: bool = True,
    **kwargs,
) -> TrackCollection:
    """Extract skeleton trajectories from a video file.

    This is the main entry point for quick usage. For processing multiple
    videos, create a :class:`Pipeline` instance for better performance.

    Args:
        source:        Path to a video file.
        device:        "cpu" or "cuda".
        detector:      Detector backend name (default: "yolo").
                       Use "yolo:yolov8s.pt" to specify a model variant.
        pose:          Pose backend name (default: "mediapipe").
        frame_skip:    Process every N-th frame for detection (default: 3).
        min_duration:  Discard tracks shorter than this (seconds, default: 1.0).
        show_progress: Show a tqdm progress bar (default: True).
        **kwargs:      Additional arguments passed to PipelineConfig.

    Returns:
        TrackCollection with all detected person skeleton trajectories.
    """
    config = PipelineConfig(
        device=device,
        frame_skip=frame_skip,
        min_track_duration=min_duration,
        **kwargs,
    )
    pipeline = Pipeline(config=config, detector=detector, pose=pose)
    try:
        return pipeline.run(source, show_progress=show_progress)
    finally:
        pipeline.close()


def load(path: str) -> TrackCollection:
    """Load previously saved tracks from file.

    Supports .npy files (both vtraxkit and legacy VideoScreener format).
    """
    return TrackCollection.load(path)
