"""Tests for TrackCollection container and filtering."""

import numpy as np
import pytest

from skeletrack.data.bbox import BBox
from skeletrack.data.collection import TrackCollection
from skeletrack.data.track import Track


def _make_track(track_id, n_frames=10, duration=2.0, x_offset=0) -> Track:
    """Helper to build a Track with controllable properties."""
    t = Track(track_id=track_id)
    for i in range(n_frames):
        t.add_frame(
            frame_idx=i * 3,
            timestamp=i * (duration / max(n_frames - 1, 1)),
            bbox=BBox(x_offset + i * 10, 0, 50, 100),
        )
    return t


def _make_static_track(track_id, n_frames=10, duration=2.0) -> Track:
    """A track where the bbox never moves."""
    t = Track(track_id=track_id)
    for i in range(n_frames):
        t.add_frame(
            frame_idx=i * 3,
            timestamp=i * (duration / max(n_frames - 1, 1)),
            bbox=BBox(100, 100, 50, 100),  # same position every frame
        )
    return t


class TestTrackCollection:
    def test_len(self):
        tc = TrackCollection([_make_track(1), _make_track(2)])
        assert len(tc) == 2

    def test_empty(self):
        tc = TrackCollection()
        assert len(tc) == 0

    def test_iter(self):
        tracks = [_make_track(1), _make_track(2)]
        tc = TrackCollection(tracks)
        assert list(tc) == tracks

    def test_getitem_int(self):
        t1, t2 = _make_track(1), _make_track(2)
        tc = TrackCollection([t1, t2])
        assert tc[0] is t1
        assert tc[1] is t2

    def test_getitem_slice(self):
        tracks = [_make_track(i) for i in range(5)]
        tc = TrackCollection(tracks)
        sliced = tc[1:3]
        assert isinstance(sliced, TrackCollection)
        assert len(sliced) == 2

    def test_repr(self):
        tc = TrackCollection(
            [_make_track(1)],
            video_metadata={"source": "test.mp4"},
        )
        r = repr(tc)
        assert "tracks=1" in r
        assert "test.mp4" in r


class TestTrackCollectionFilter:
    def test_filter_min_duration(self):
        short = _make_track(1, n_frames=5, duration=0.5)
        long = _make_track(2, n_frames=20, duration=5.0)
        tc = TrackCollection([short, long])
        filtered = tc.filter(min_duration=1.0)
        assert len(filtered) == 1
        assert filtered[0].track_id == 2

    def test_filter_min_frames(self):
        few = _make_track(1, n_frames=3)
        many = _make_track(2, n_frames=20)
        tc = TrackCollection([few, many])
        filtered = tc.filter(min_frames=10)
        assert len(filtered) == 1
        assert filtered[0].track_id == 2

    def test_filter_motion_threshold(self):
        moving = _make_track(1, n_frames=10, x_offset=0)   # bbox moves
        static = _make_static_track(2, n_frames=10)          # bbox stays put
        tc = TrackCollection([moving, static])
        filtered = tc.filter(motion_threshold=1.0)
        assert len(filtered) == 1
        assert filtered[0].track_id == 1

    def test_filter_custom(self):
        t1 = _make_track(1, n_frames=5)
        t2 = _make_track(2, n_frames=10)
        tc = TrackCollection([t1, t2])
        filtered = tc.filter(custom=lambda t: t.track_id == 2)
        assert len(filtered) == 1
        assert filtered[0].track_id == 2

    def test_filter_combined(self):
        t1 = _make_track(1, n_frames=5, duration=0.5)
        t2 = _make_track(2, n_frames=20, duration=5.0)
        t3 = _make_track(3, n_frames=15, duration=3.0)
        tc = TrackCollection([t1, t2, t3])
        filtered = tc.filter(min_duration=1.0, min_frames=16)
        assert len(filtered) == 1
        assert filtered[0].track_id == 2

    def test_filter_returns_new_collection(self):
        tc = TrackCollection([_make_track(1)])
        filtered = tc.filter(min_duration=0.0)
        assert filtered is not tc

    def test_filter_preserves_metadata(self):
        meta = {"source": "test.mp4", "fps": 30.0}
        tc = TrackCollection([_make_track(1)], video_metadata=meta)
        filtered = tc.filter(min_duration=0.0)
        assert filtered.video_metadata == meta
