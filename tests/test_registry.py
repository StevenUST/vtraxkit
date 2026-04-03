"""Tests for backend registry mechanism."""

import pytest

from vtraxkit.detection.base import DetectorBackend
from vtraxkit.detection.registry import _DETECTORS, register_detector
from vtraxkit.pose.base import PoseBackend
from vtraxkit.pose.registry import _POSE_BACKENDS, register_pose


class TestDetectorRegistry:
    def test_register_and_retrieve(self):
        @register_detector("_test_det")
        class _TestDetector(DetectorBackend):
            def detect(self, frame, frame_idx):
                return []

        assert "_test_det" in _DETECTORS
        assert _DETECTORS["_test_det"] is _TestDetector

        # Cleanup
        del _DETECTORS["_test_det"]

    def test_unknown_detector_raises(self):
        from vtraxkit.detection.registry import get_detector
        with pytest.raises(ValueError, match="Unknown detector"):
            get_detector("nonexistent_backend_xyz")

    def test_passthrough_instance(self):
        from vtraxkit.detection.registry import get_detector

        class _DummyDetector(DetectorBackend):
            def detect(self, frame, frame_idx):
                return []

        instance = _DummyDetector()
        assert get_detector(instance) is instance


class TestPoseRegistry:
    def test_register_and_retrieve(self):
        @register_pose("_test_pose")
        class _TestPose(PoseBackend):
            def estimate(self, frame, bbox=None):
                pass

            @property
            def layout(self):
                pass

        assert "_test_pose" in _POSE_BACKENDS
        assert _POSE_BACKENDS["_test_pose"] is _TestPose

        # Cleanup
        del _POSE_BACKENDS["_test_pose"]

    def test_unknown_pose_raises(self):
        from vtraxkit.pose.registry import get_pose_backend
        with pytest.raises(ValueError, match="Unknown pose backend"):
            get_pose_backend("nonexistent_backend_xyz")

    def test_passthrough_instance(self):
        from vtraxkit.pose.registry import get_pose_backend

        class _DummyPose(PoseBackend):
            def estimate(self, frame, bbox=None):
                pass

            @property
            def layout(self):
                pass

        instance = _DummyPose()
        assert get_pose_backend(instance) is instance
