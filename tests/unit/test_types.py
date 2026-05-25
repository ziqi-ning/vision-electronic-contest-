"""test_types.py — 核心类型单元测试"""

import pytest
from src.core.types import BoundingBox, DetectionResult, RadarScan, FusionResult


class TestBoundingBox:
    def test_center(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=60)
        assert bbox.center == (60, 50)

    def test_area(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        assert bbox.area == 5000

    def test_from_rect(self):
        bbox = BoundingBox.from_rect((100, 100, 80, 40))
        assert bbox.x == 60
        assert bbox.y == 80
        assert bbox.width == 80
        assert bbox.height == 40
        assert bbox.center == (100, 100)

    def test_from_rect_odd_dimensions(self):
        bbox = BoundingBox.from_rect((100, 100, 81, 41))
        assert bbox.x == 60
        assert bbox.y == 80

    def test_area_zero(self):
        bbox = BoundingBox(x=0, y=0, width=0, height=0)
        assert bbox.area == 0


class TestDetectionResult:
    def test_creation(self):
        result = DetectionResult(type="ellipse", confidence=0.85)
        assert result.type == "ellipse"
        assert result.confidence == 0.85
        assert result.bbox is None
        assert result.center is None
        assert result.extra == {}

    def test_with_bbox(self):
        bbox = BoundingBox(x=10, y=20, width=50, height=30)
        result = DetectionResult(type="trapezoid", confidence=0.9, bbox=bbox, center=(35, 35))
        assert result.bbox is bbox
        assert result.center == (35, 35)

    def test_with_extra(self):
        result = DetectionResult(
            type="triangle",
            confidence=0.7,
            extra={"color": "red", "lengh": 120}
        )
        assert result.extra["color"] == "red"
        assert result.extra["lengh"] == 120


class TestRadarScan:
    def test_creation(self):
        scan = RadarScan(timestamp=1.0, points=[(1.0, 0.0), (2.0, 9000)])
        assert scan.timestamp == 1.0
        assert scan.points == [(1.0, 0.0), (2.0, 9000)]
        assert scan.obstacles == []

    def test_with_obstacles(self):
        scan = RadarScan(
            timestamp=2.5,
            points=[(1.0, 5000)],
            obstacles=[(0.8, 4500)]
        )
        assert len(scan.obstacles) == 1


class TestFusionResult:
    def test_full(self):
        result = FusionResult(pixel=(320, 240), distance=1.5, angle=30.0, obstacle_detected=True)
        assert result.pixel == (320, 240)
        assert result.distance == 1.5
        assert result.angle == 30.0
        assert result.obstacle_detected is True

    def test_default(self):
        result = FusionResult(pixel=(100, 200))
        assert result.distance is None
        assert result.angle is None
        assert result.obstacle_detected is False
