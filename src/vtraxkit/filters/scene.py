"""Scene change detection via histogram correlation."""

from __future__ import annotations

import cv2
import numpy as np


def scene_changed(prev_frame: np.ndarray, curr_frame: np.ndarray,
                  threshold: float = 0.4) -> bool:
    """Detect scene change by comparing grayscale histogram correlation.

    Returns True if the frames appear to be from different scenes.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return corr < threshold
