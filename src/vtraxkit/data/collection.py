"""Multi-person track container with chainable operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, overload

from .track import Track


class TrackCollection:
    """Container for multiple person tracks from a single video.

    Supports iteration, indexing, filtering, and export::

        tracks = vtraxkit.extract("video.mp4")
        children = tracks.filter(min_duration=2.0)
        children.save("output.npy")
    """

    def __init__(
        self,
        tracks: list[Track] | None = None,
        video_metadata: dict[str, Any] | None = None,
    ):
        self.tracks: list[Track] = tracks or []
        self.video_metadata: dict[str, Any] = video_metadata or {}

    # -- container protocol -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self.tracks)

    @overload
    def __getitem__(self, idx: int) -> Track: ...
    @overload
    def __getitem__(self, idx: slice) -> TrackCollection: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TrackCollection(self.tracks[idx], self.video_metadata)
        return self.tracks[idx]

    def __repr__(self) -> str:
        src = self.video_metadata.get("source", "unknown")
        return f"TrackCollection(tracks={len(self.tracks)}, source={src!r})"

    # -- filtering --------------------------------------------------------------

    def filter(
        self,
        *,
        min_duration: float | None = None,
        min_frames: int | None = None,
        motion_threshold: float | None = None,
        custom: Any | None = None,
    ) -> TrackCollection:
        """Return a new collection with only tracks that pass all criteria.

        Args:
            min_duration:     Minimum track duration in seconds.
            min_frames:       Minimum number of detected frames.
            motion_threshold: Minimum bbox center std-dev (filters static objects).
            custom:           A callable ``(Track) -> bool`` for custom filtering.
        """
        import numpy as np

        result = self.tracks

        if min_duration is not None:
            result = [t for t in result if t.duration >= min_duration]

        if min_frames is not None:
            result = [t for t in result if t.num_frames >= min_frames]

        if motion_threshold is not None:
            kept = []
            for t in result:
                if t.num_frames < 3:
                    kept.append(t)
                    continue
                centers = np.array([b.center for b in t.bboxes], dtype=np.float32)
                std = np.std(centers, axis=0).mean()
                if std >= motion_threshold:
                    kept.append(t)
            result = kept

        if custom is not None:
            result = [t for t in result if custom(t)]

        return TrackCollection(result, self.video_metadata)

    # -- export -----------------------------------------------------------------

    def save(self, path: str | Path, *, format: str = "auto") -> None:
        """Save tracks to file.

        Args:
            path:   Output file path.
            format: One of "auto", "npy", "coco". Auto-detects from extension.
        """
        path = Path(path)
        if format == "auto":
            ext = path.suffix.lower()
            fmt_map = {".npy": "npy", ".json": "coco"}
            format = fmt_map.get(ext, "npy")

        if format == "npy":
            from ..io.npy import save_npy
            save_npy(self, path)
        elif format == "coco":
            from ..io.coco import save_coco
            save_coco(self, path)
        else:
            raise ValueError(f"Unknown format: {format!r}")

    def to_dataframe(self):
        """Convert to pandas DataFrame. Requires ``pip install vtraxkit[pandas]``."""
        from ..io.dataframe import to_dataframe
        return to_dataframe(self)

    # -- class methods ----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> TrackCollection:
        """Load tracks from a saved file."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".npy":
            from ..io.npy import load_npy
            return load_npy(path)
        raise ValueError(f"Unsupported file format: {ext!r}")
