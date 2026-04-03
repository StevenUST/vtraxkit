"""Tests for Track trajectory container."""

import numpy as np
import pytest

from skeletrack.data.bbox import BBox
from skeletrack.data.skeleton import Skeleton
from skeletrack.data.track import Track


def _make_track(n_frames=10, with_skeletons=False) -> Track:
    """Helper to build a Track with n_frames."""
    t = Track(track_id=1)
    for i in range(n_frames):
        t.add_frame(
            frame_idx=i * 3,
            timestamp=i * 0.1,
            bbox=BBox(i * 10, i * 5, 50, 100),
        )
    if with_skeletons:
        for i in range(n_frames):
            t.skeletons[i] = Skeleton({"pose": np.random.rand(33, 4).astype(np.float32)})
    return t


class TestTrack:
    def test_add_frame(self):
        t = Track(track_id=1)
        t.add_frame(0, 0.0, BBox(10, 20, 30, 40))
        assert t.num_frames == 1
        assert t.frames == [0]
        assert t.bboxes[0] == BBox(10, 20, 30, 40)
        assert t.skeletons[0] is None

    def test_properties_empty(self):
        t = Track(track_id=1)
        assert t.start_frame == 0
        assert t.end_frame == 0
        assert t.duration == 0.0
        assert t.num_frames == 0
        assert t.has_skeletons is False

    def test_properties(self):
        t = _make_track(10)
        assert t.start_frame == 0
        assert t.end_frame == 27
        assert t.start_time == 0.0
        assert t.end_time == pytest.approx(0.9)
        assert t.duration == pytest.approx(0.9)
        assert t.num_frames == 10

    def test_has_skeletons_false(self):
        t = _make_track(5)
        assert t.has_skeletons is False

    def test_has_skeletons_true(self):
        t = _make_track(5, with_skeletons=True)
        assert t.has_skeletons is True

    def test_skeleton_array(self):
        t = _make_track(5, with_skeletons=True)
        arr = t.skeleton_array("pose")
        assert arr.shape == (5, 33, 4)
        assert arr.dtype == np.float32

    def test_skeleton_array_missing_group(self):
        t = _make_track(5, with_skeletons=True)
        arr = t.skeleton_array("face")
        assert arr.shape == (0, 0, 0)

    def test_skeleton_array_partial(self):
        """Frames without skeletons should be zeros."""
        t = _make_track(5)
        t.skeletons[2] = Skeleton({"pose": np.ones((33, 4), dtype=np.float32)})
        arr = t.skeleton_array("pose")
        assert arr.shape == (5, 33, 4)
        assert arr[2].sum() > 0
        assert arr[0].sum() == 0.0
        assert arr[4].sum() == 0.0

    def test_bbox_centers(self):
        t = _make_track(3)
        centers = t.bbox_centers()
        assert centers.shape == (3, 2)
        assert centers.dtype == np.float32

    def test_repr(self):
        t = _make_track(5)
        r = repr(t)
        assert "id=1" in r
        assert "frames=5" in r

    def test_repr_with_skeleton(self):
        t = _make_track(3, with_skeletons=True)
        r = repr(t)
        assert "+skel" in r
