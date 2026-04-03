"""Bounding box value object."""

from __future__ import annotations

from typing import NamedTuple


class BBox(NamedTuple):
    """Axis-aligned bounding box in (x, y, w, h) format."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> BBox:
        return cls(x1, y1, x2 - x1, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union between two boxes."""
    ix = max(a.x, b.x)
    iy = max(a.y, b.y)
    iw = min(a.x + a.w, b.x + b.w) - ix
    ih = min(a.y + a.h, b.y + b.h) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
