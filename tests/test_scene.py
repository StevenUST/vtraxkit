"""Tests for scene change detection."""

import numpy as np
import pytest

from skeletrack.filters.scene import scene_changed


class TestSceneChanged:
    def test_identical_frames(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert scene_changed(frame, frame) is False

    def test_similar_frames(self):
        frame1 = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Small noise
        frame2 = frame1.copy()
        frame2[:10, :10] = 200
        assert scene_changed(frame1, frame2) is False

    def test_completely_different_frames(self):
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        assert scene_changed(black, white) is True

    def test_high_threshold_always_triggers(self):
        """A threshold above 1.0 means any correlation is below it."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert scene_changed(frame, frame, threshold=1.1) is True

    def test_low_threshold_never_triggers(self):
        """A threshold of -2.0 means even uncorrelated frames won't trigger."""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        assert scene_changed(black, white, threshold=-2.0) is False
