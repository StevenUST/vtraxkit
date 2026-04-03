"""Tests for BBox value object."""

import pytest

from vtraxkit.data.bbox import BBox, iou


class TestBBox:
    def test_construction(self):
        b = BBox(10, 20, 100, 200)
        assert b.x == 10
        assert b.y == 20
        assert b.w == 100
        assert b.h == 200

    def test_center(self):
        b = BBox(0, 0, 100, 200)
        assert b.center == (50.0, 100.0)

    def test_center_offset(self):
        b = BBox(10, 20, 100, 200)
        assert b.center == (60.0, 120.0)

    def test_area(self):
        b = BBox(0, 0, 50, 80)
        assert b.area == 4000

    def test_area_zero(self):
        b = BBox(0, 0, 0, 0)
        assert b.area == 0

    def test_to_xyxy(self):
        b = BBox(10, 20, 30, 40)
        assert b.to_xyxy() == (10, 20, 40, 60)

    def test_from_xyxy(self):
        b = BBox.from_xyxy(10, 20, 40, 60)
        assert b == BBox(10, 20, 30, 40)

    def test_from_xyxy_roundtrip(self):
        original = BBox(5, 10, 50, 80)
        rebuilt = BBox.from_xyxy(*original.to_xyxy())
        assert rebuilt == original

    def test_named_tuple_unpacking(self):
        b = BBox(10, 20, 30, 40)
        x, y, w, h = b
        assert (x, y, w, h) == (10, 20, 30, 40)


class TestIoU:
    def test_identical_boxes(self):
        b = BBox(0, 0, 100, 100)
        assert iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 10, 10)
        assert iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        # intersection: 5x5=25, union: 100+100-25=175
        assert iou(a, b) == pytest.approx(25 / 175)

    def test_one_inside_other(self):
        outer = BBox(0, 0, 100, 100)
        inner = BBox(25, 25, 50, 50)
        # intersection: 2500, union: 10000+2500-2500=10000
        assert iou(outer, inner) == pytest.approx(2500 / 10000)

    def test_touching_edges(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 10, 10)
        assert iou(a, b) == 0.0

    def test_symmetry(self):
        a = BBox(0, 0, 20, 20)
        b = BBox(10, 10, 20, 20)
        assert iou(a, b) == iou(b, a)
