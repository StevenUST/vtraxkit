"""MediaPipe Holistic pose estimation backend."""

from __future__ import annotations

import cv2
import numpy as np

from ..data.bbox import BBox
from ..data.skeleton import Skeleton
from .base import KeypointLayout, PoseBackend
from .registry import register_pose


def _lm_to_array(landmark_list, n_points: int, n_dims: int) -> np.ndarray:
    """Convert MediaPipe landmark list to float32 array. Zeros if missing."""
    arr = np.zeros((n_points, n_dims), dtype=np.float32)
    if landmark_list is None:
        return arr
    for i, lm in enumerate(landmark_list.landmark):
        if i >= n_points:
            break
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
        if n_dims == 4:
            arr[i, 3] = getattr(lm, "visibility", 1.0)
    return arr


@register_pose("mediapipe")
class MediaPipePose(PoseBackend):
    """Pose estimation using MediaPipe Holistic.

    Extracts pose (33 kpts), left_hand (21), right_hand (21), face (468).
    Requires ``pip install vtraxkit[mediapipe]``.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is required for this backend. "
                "Install it with: pip install vtraxkit[mediapipe]"
            )

        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame: np.ndarray, bbox: BBox | None = None) -> Skeleton:
        if bbox is not None:
            x, y, w, h = bbox
            # Clamp to frame bounds
            fh, fw = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(fw, x + w)
            y2 = min(fh, y + h)
            frame = frame[y1:y2, x1:x2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._holistic.process(rgb)

        keypoints = {
            "pose": _lm_to_array(res.pose_landmarks, 33, 4),
            "left_hand": _lm_to_array(res.left_hand_landmarks, 21, 4),
            "right_hand": _lm_to_array(res.right_hand_landmarks, 21, 4),
            "face": _lm_to_array(res.face_landmarks, 468, 3),
        }
        return Skeleton(keypoints)

    @property
    def layout(self) -> KeypointLayout:
        return KeypointLayout(
            groups={"pose": 33, "left_hand": 21, "right_hand": 21, "face": 468},
            dims={"pose": 4, "left_hand": 4, "right_hand": 4, "face": 3},
        )

    def close(self) -> None:
        if self._holistic:
            self._holistic.close()
