"""YOLOv8 person detection + ByteTrack tracking backend."""

from __future__ import annotations

import numpy as np

from ..data.bbox import BBox
from .base import Detection, DetectorBackend
from .registry import register_detector


@register_detector("yolo")
class YoloDetector(DetectorBackend):
    """Person detector using YOLOv8 + ByteTrack.

    Requires ``pip install vtraxkit[yolo]``.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        min_confidence: float = 0.4,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for the YOLO backend. "
                "Install it with: pip install vtraxkit[yolo]"
            )

        import torch

        self._model = YOLO(model_name)
        self._device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._min_confidence = min_confidence

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            classes=[0],  # person class only
            conf=self._min_confidence,
            device=self._device,
            verbose=False,
        )

        detections = []
        if results and results[0].boxes.id is not None:
            for box, track_id, conf in zip(
                results[0].boxes.xyxy,
                results[0].boxes.id,
                results[0].boxes.conf,
            ):
                x1, y1, x2, y2 = map(int, box)
                bbox = BBox.from_xyxy(x1, y1, x2, y2)
                detections.append(Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    track_id=int(track_id),
                ))

        return detections

    def reset(self) -> None:
        # Reset YOLO tracker state by creating a new tracker
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.trackers = []

    def close(self) -> None:
        pass
