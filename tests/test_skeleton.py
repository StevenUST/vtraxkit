"""Tests for Skeleton keypoint container."""

import numpy as np
import pytest

from skeletrack.data.skeleton import Skeleton


class TestSkeleton:
    def test_empty(self):
        s = Skeleton()
        assert s.pose is None
        assert s.left_hand is None
        assert s.right_hand is None
        assert s.face is None

    def test_accessors(self):
        kps = {
            "pose": np.zeros((33, 4), dtype=np.float32),
            "left_hand": np.ones((21, 4), dtype=np.float32),
        }
        s = Skeleton(kps)
        assert s.pose is not None
        assert s.pose.shape == (33, 4)
        assert s.left_hand is not None
        assert s.left_hand.shape == (21, 4)
        assert s.right_hand is None
        assert s.face is None

    def test_to_flat_array_single_group(self):
        kps = {"pose": np.ones((33, 4), dtype=np.float32)}
        s = Skeleton(kps)
        flat = s.to_flat_array()
        assert flat.shape == (33, 4)

    def test_to_flat_array_mixed_dims(self):
        kps = {
            "pose": np.ones((33, 4), dtype=np.float32),
            "face": np.ones((468, 3), dtype=np.float32),
        }
        s = Skeleton(kps)
        flat = s.to_flat_array()
        # pose padded to 4 dims, face padded to 4 dims
        assert flat.shape == (33 + 468, 4)
        # face z-column should be 1, padded column should be 0
        assert flat[33, 2] == 1.0  # z
        assert flat[33, 3] == 0.0  # padded

    def test_to_flat_array_empty(self):
        s = Skeleton()
        flat = s.to_flat_array()
        assert flat.shape == (0, 0)

    def test_to_dict(self):
        arr = np.random.rand(33, 4).astype(np.float32)
        s = Skeleton({"pose": arr})
        d = s.to_dict()
        assert "pose" in d
        np.testing.assert_array_equal(d["pose"], arr)
        # Should be a copy, not the same object
        assert d["pose"] is not arr

    def test_from_legacy_dict(self):
        legacy = {
            "pose": np.ones((33, 4), dtype=np.float32),
            "left_hand": np.zeros((21, 4), dtype=np.float32),
            "right_hand": None,
        }
        s = Skeleton.from_legacy_dict(legacy)
        assert s.pose is not None
        assert s.left_hand is not None
        assert s.right_hand is None  # None is skipped

    def test_from_legacy_dict_empty_array(self):
        legacy = {"pose": np.array([], dtype=np.float32)}
        s = Skeleton.from_legacy_dict(legacy)
        assert s.pose is None  # empty array is skipped

    def test_repr(self):
        kps = {"pose": np.zeros((33, 4), dtype=np.float32)}
        s = Skeleton(kps)
        r = repr(s)
        assert "pose" in r
        assert "33" in r
