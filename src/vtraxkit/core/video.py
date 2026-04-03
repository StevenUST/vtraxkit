"""Video reader utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameInfo:
    """A single video frame with its metadata."""

    frame: np.ndarray
    frame_idx: int
    timestamp: float


class VideoReader:
    """Iterator over video frames with metadata.

    Usage::

        with VideoReader("video.mp4") as reader:
            for info in reader:
                process(info.frame, info.frame_idx)
    """

    def __init__(self, path: str, frame_skip: int = 1):
        self.path = path
        self.frame_skip = max(1, frame_skip)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> VideoReader:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened")
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened")
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened")
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened")
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def metadata(self) -> dict:
        return {
            "source": self.path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
        }

    def __iter__(self) -> Iterator[FrameInfo]:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened. Use `with` or call open().")

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                timestamp = frame_idx / self.fps
                yield FrameInfo(frame=frame, frame_idx=frame_idx, timestamp=timestamp)

            frame_idx += 1

    def read_range(self, start_frame: int, end_frame: int) -> Iterator[FrameInfo]:
        """Read a specific frame range (inclusive)."""
        if self._cap is None:
            raise RuntimeError("VideoReader not opened")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for idx in range(start_frame, end_frame + 1):
            ret, frame = self._cap.read()
            if not ret:
                break
            yield FrameInfo(
                frame=frame,
                frame_idx=idx,
                timestamp=idx / self.fps,
            )

    def __enter__(self) -> VideoReader:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
