"""Main extraction pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from ..data.bbox import BBox
from ..data.collection import TrackCollection
from ..data.track import Track
from ..detection.base import DetectorBackend
from ..detection.registry import get_detector
from ..filters.scene import scene_changed
from ..pose.base import PoseBackend
from ..pose.registry import get_pose_backend
from .config import PipelineConfig
from .video import VideoReader


class Pipeline:
    """Reusable extraction pipeline.

    Usage::

        pipeline = Pipeline(device="cuda")
        tracks = pipeline.run("video.mp4")
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        detector: str | DetectorBackend = "yolo",
        pose: str | PoseBackend = "mediapipe",
        device: str | None = None,
        **kwargs: Any,
    ):
        self.config = config or PipelineConfig()
        if device is not None:
            self.config.device = device

        # Resolve backends (lazy — only instantiated when needed)
        self._detector_spec = detector
        self._pose_spec = pose
        self._extra_kwargs = kwargs
        self._detector: DetectorBackend | None = None
        self._pose: PoseBackend | None = None

    def _get_detector(self) -> DetectorBackend:
        if self._detector is None:
            self._detector = get_detector(
                self._detector_spec,
                device=self.config.device,
                min_confidence=self.config.min_confidence,
                **{k: v for k, v in self._extra_kwargs.items()
                   if k in ("model_name",)},
            )
        return self._detector

    def _get_pose(self) -> PoseBackend:
        if self._pose is None:
            self._pose = get_pose_backend(
                self._pose_spec,
                min_detection_confidence=self.config.pose_min_detection_conf,
                min_tracking_confidence=self.config.pose_min_tracking_conf,
            )
        return self._pose

    def run(self, source: str, *, show_progress: bool = True) -> TrackCollection:
        """Extract skeleton trajectories from a video.

        Args:
            source:        Path to video file.
            show_progress: Show a tqdm progress bar.

        Returns:
            TrackCollection with all detected person tracks and their skeletons.
        """
        cfg = self.config
        detector = self._get_detector()

        # -- Phase 1: Detection + Tracking ------------------------------------
        tracks_by_id: dict[int, Track] = {}
        yolo_to_track: dict[int, int] = {}  # yolo_track_id -> track_id
        active_yolo: dict[int, int] = {}    # yolo_track_id -> last_frame_idx
        next_track_id = 1
        prev_frame: np.ndarray | None = None

        with VideoReader(source, frame_skip=cfg.frame_skip) as reader:
            video_meta = reader.metadata()
            fps = reader.fps

            frames_iter = reader
            if show_progress:
                frames_iter = tqdm(
                    reader,
                    total=reader.total_frames // cfg.frame_skip,
                    desc="Tracking",
                    unit="frame",
                )

            for info in frames_iter:
                # Scene change detection
                if prev_frame is not None:
                    if scene_changed(prev_frame, info.frame, cfg.scene_change_threshold):
                        detector.reset()
                        yolo_to_track.clear()
                        active_yolo.clear()

                prev_frame = info.frame.copy()

                # Detect + track
                detections = detector.detect(info.frame, info.frame_idx)
                current_yolo_ids = set()

                for det in detections:
                    yolo_id = det.track_id
                    if yolo_id is None:
                        continue

                    current_yolo_ids.add(yolo_id)

                    if yolo_id not in yolo_to_track:
                        tid = next_track_id
                        next_track_id += 1
                        yolo_to_track[yolo_id] = tid
                        tracks_by_id[tid] = Track(track_id=tid)

                    tid = yolo_to_track[yolo_id]
                    tracks_by_id[tid].add_frame(info.frame_idx, info.timestamp, det.bbox)
                    active_yolo[yolo_id] = info.frame_idx

                # Expire stale tracks
                stale = [
                    yid for yid, last_f in active_yolo.items()
                    if info.frame_idx - last_f > cfg.max_missing_frames
                    and yid not in current_yolo_ids
                ]
                for yid in stale:
                    active_yolo.pop(yid, None)
                    yolo_to_track.pop(yid, None)

        all_tracks = list(tracks_by_id.values())

        # -- Phase 2: Filter --------------------------------------------------
        if cfg.min_track_duration > 0:
            all_tracks = [t for t in all_tracks if t.duration >= cfg.min_track_duration]

        if cfg.motion_threshold is not None:
            kept = []
            for t in all_tracks:
                if t.num_frames < 3:
                    continue
                centers = np.array([b.center for b in t.bboxes], dtype=np.float32)
                std = float(np.std(centers, axis=0).mean())
                if std >= cfg.motion_threshold:
                    kept.append(t)
            all_tracks = kept

        # -- Phase 3: Pose Estimation -----------------------------------------
        pose_backend = self._get_pose()

        with VideoReader(source, frame_skip=1) as reader:
            for track in tqdm(all_tracks, desc="Pose estimation", disable=not show_progress):
                for i, (frame_idx, bbox) in enumerate(zip(track.frames, track.bboxes)):
                    for info in reader.read_range(frame_idx, frame_idx):
                        skeleton = pose_backend.estimate(info.frame, bbox)
                        track.skeletons[i] = skeleton

        return TrackCollection(tracks=all_tracks, video_metadata=video_meta)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        if self._pose is not None:
            self._pose.close()
