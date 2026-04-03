"""Tests for npy serialization roundtrip."""

import numpy as np
import pytest

from skeletrack.data.bbox import BBox
from skeletrack.data.collection import TrackCollection
from skeletrack.data.skeleton import Skeleton
from skeletrack.data.track import Track
from skeletrack.io.npy import load_npy, save_npy


def _make_collection() -> TrackCollection:
    """Build a small TrackCollection with skeleton data."""
    t = Track(track_id=1)
    for i in range(3):
        t.add_frame(i * 3, i * 0.1, BBox(i * 10, 0, 50, 100))
        t.skeletons[i] = Skeleton({
            "pose": np.random.rand(33, 4).astype(np.float32),
        })

    return TrackCollection(
        tracks=[t],
        video_metadata={"source": "test.mp4", "fps": 30.0},
    )


class TestNpyRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        original = _make_collection()
        path = tmp_path / "tracks.npy"

        save_npy(original, path)
        loaded = load_npy(path)

        assert len(loaded) == len(original)
        assert loaded.video_metadata["source"] == "test.mp4"
        assert loaded.video_metadata["fps"] == 30.0

    def test_track_data_preserved(self, tmp_path):
        original = _make_collection()
        path = tmp_path / "tracks.npy"

        save_npy(original, path)
        loaded = load_npy(path)

        orig_t = original.tracks[0]
        load_t = loaded.tracks[0]

        assert load_t.track_id == orig_t.track_id
        assert load_t.frames == orig_t.frames
        assert load_t.timestamps == pytest.approx(orig_t.timestamps)
        assert len(load_t.bboxes) == len(orig_t.bboxes)
        for a, b in zip(load_t.bboxes, orig_t.bboxes):
            assert a == b

    def test_skeleton_data_preserved(self, tmp_path):
        original = _make_collection()
        path = tmp_path / "tracks.npy"

        save_npy(original, path)
        loaded = load_npy(path)

        for orig_s, load_s in zip(
            original.tracks[0].skeletons, loaded.tracks[0].skeletons
        ):
            assert load_s is not None
            np.testing.assert_array_almost_equal(
                load_s.keypoints["pose"], orig_s.keypoints["pose"]
            )

    def test_none_skeletons_preserved(self, tmp_path):
        t = Track(track_id=1)
        t.add_frame(0, 0.0, BBox(0, 0, 50, 100))
        # skeleton stays None
        tc = TrackCollection(tracks=[t])

        path = tmp_path / "tracks.npy"
        save_npy(tc, path)
        loaded = load_npy(path)
        assert loaded.tracks[0].skeletons[0] is None

    def test_creates_parent_dirs(self, tmp_path):
        tc = TrackCollection(tracks=[])
        path = tmp_path / "sub" / "dir" / "tracks.npy"
        save_npy(tc, path)
        assert path.exists()

    def test_load_unknown_format(self, tmp_path):
        path = tmp_path / "bad.npy"
        np.save(str(path), "not a valid format", allow_pickle=True)
        with pytest.raises(ValueError, match="Unrecognized"):
            load_npy(path)


class TestNpyLegacy:
    def test_load_legacy_format(self, tmp_path):
        """Simulate VideoScreener's legacy format."""
        frames = [
            {
                "frame_idx": 0,
                "timestamp": 0.0,
                "pose": np.ones((33, 4), dtype=np.float32),
                "left_hand": np.zeros((21, 4), dtype=np.float32),
                "right_hand": np.zeros((21, 4), dtype=np.float32),
                "face": np.zeros((468, 3), dtype=np.float32),
            },
            {
                "frame_idx": 3,
                "timestamp": 0.1,
                "pose": np.ones((33, 4), dtype=np.float32) * 2,
                "left_hand": None,
                "right_hand": None,
                "face": None,
            },
        ]
        path = tmp_path / "legacy.npy"
        np.save(str(path), frames, allow_pickle=True)

        loaded = load_npy(path)
        assert len(loaded) == 1  # single track
        assert loaded.tracks[0].track_id == 1
        assert loaded.tracks[0].num_frames == 2
        assert loaded.tracks[0].skeletons[0].pose is not None

    def test_load_legacy_empty(self, tmp_path):
        path = tmp_path / "empty.npy"
        np.save(str(path), [], allow_pickle=True)
        loaded = load_npy(path)
        assert len(loaded) == 0
