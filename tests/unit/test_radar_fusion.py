"""test_radar_fusion.py — 雷达融合单元测试"""

import pytest
import asyncio
import math
import numpy as np

from src.radar.fusion import RadarFusion
from src.radar.base import RadarSource, RadarScanResult, RadarPoint, RadarScanResult


class FakeRadarSource(RadarSource):
    """稳定的假数据源，供测试使用"""

    def __init__(self, points=None):
        self._points = points or []
        self._available = True

    def is_available(self) -> bool:
        return self._available

    async def get_scan(self):
        await asyncio.sleep(0)
        return RadarScanResult(
            timestamp=0.0,
            points=self._points,
            obstacles=[]
        )


class TestRadarFusion:
    def test_init_auto_detects_source(self):
        """不传 source 时自动检测（测试环境走 SimRadarSource）"""
        fusion = RadarFusion(source=FakeRadarSource([]))
        assert fusion._source is not None

    def test_angle_to_distance_one_point(self):
        """单点雷达在正前方 → 返回该点"""
        source = FakeRadarSource([
            RadarPoint(distance_m=1.5, angle_rad=0.0),
        ])
        fusion = RadarFusion(source=source)

        result = asyncio.get_event_loop().run_until_complete(
            fusion.angle_to_distance(search_start=-180, search_end=180)
        )
        dist_cm, angle_cd = result
        assert dist_cm == 150
        assert angle_cd == 18000  # 180.00° → 18000 centidegree

    def test_angle_to_distance_no_points(self):
        """无点时返回默认值"""
        source = FakeRadarSource([])
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(
            fusion.angle_to_distance(search_start=0, search_end=90)
        )
        assert result == (3000, 40000)

    def test_angle_to_distance_empty_range(self):
        """范围内无点时返回默认值"""
        source = FakeRadarSource([
            RadarPoint(distance_m=1.0, angle_rad=math.radians(90)),
        ])
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(
            fusion.angle_to_distance(search_start=0, search_end=30)
        )
        assert result == (3000, 40000)

    def test_get_obstacle_empty(self):
        """无障碍物返回默认值"""
        source = FakeRadarSource([])
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(fusion.get_obstacle())
        assert result == [(3000, 40000)]

    def test_get_obstacle_cluster(self):
        """点群应被正确聚类"""
        points = [
            RadarPoint(distance_m=1.0, angle_rad=math.radians(-10)),
            RadarPoint(distance_m=1.01, angle_rad=math.radians(-9)),
            RadarPoint(distance_m=1.02, angle_rad=math.radians(-8)),
        ]
        source = FakeRadarSource(points)
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(fusion.get_obstacle())
        assert len(result) == 1
        dist_cm, angle_cd = result[0]
        assert 90 < dist_cm < 110  # ~100cm

    def test_get_obstacle_isolated_point(self):
        """孤立点（与周围差 > 3cm）应被检测"""
        points = [
            RadarPoint(distance_m=2.0, angle_rad=math.radians(0)),
            RadarPoint(distance_m=2.5, angle_rad=math.radians(10)),  # 孤立点
            RadarPoint(distance_m=2.0, angle_rad=math.radians(20)),
            RadarPoint(distance_m=2.0, angle_rad=math.radians(30)),
        ]
        source = FakeRadarSource(points)
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(fusion.get_obstacle())
        # 10°处孤立点应被检测
        assert len(result) >= 1

    def test_get_obstacle_beyond_max_distance(self):
        """超过 10m 的点不参与障碍物检测"""
        points = [
            RadarPoint(distance_m=15.0, angle_rad=0.0),
            RadarPoint(distance_m=1.0, angle_rad=math.radians(10)),
        ]
        source = FakeRadarSource(points)
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(fusion.get_obstacle())
        assert len(result) == 1

    def test_site_to_distance_no_radar_data(self):
        """无雷达数据返回默认值"""
        source = FakeRadarSource([])
        fusion = RadarFusion(source=source)
        result = asyncio.get_event_loop().run_until_complete(
            fusion.site_to_distance(320, 240, (500.0, 500.0, 320, 240, 0, 0, 0, 0, 0.5, 0.03))
        )
        assert result == (3000, 40000)

    def test_detect_obstacles_empty(self):
        """_detect_obstacles 空输入返回空"""
        fusion = RadarFusion(source=FakeRadarSource([]))
        obstacles = fusion._detect_obstacles(
            ranges=np.array([]),
            angles=np.array([]),
            max_dist=10.0,
            threshold=0.03
        )
        assert obstacles == []

    def test_detect_obstacles_single_cluster(self):
        """单点群检测"""
        fusion = RadarFusion(source=FakeRadarSource([]))
        ranges = np.array([1.0, 1.01, 1.02, 1.03])
        angles = np.array([0.0, 0.1, 0.2, 0.3])
        obstacles = fusion._detect_obstacles(ranges, angles, 10.0, 0.03)
        assert len(obstacles) == 1

    def test_is_valid_isolated_point_true(self):
        """孤立点判定逻辑"""
        fusion = RadarFusion(source=FakeRadarSource([]))
        ranges = np.array([1.0, 2.0, 2.0, 2.0])
        angles = np.array([0.0, 0.5, 1.0, 1.5])
        # idx=0: 左边无点, 右边 2.0, 最近聚类 2.0, diff=1.0 > 0.03
        assert fusion._is_valid_isolated_point(0, ranges, angles, [], 10.0, 0.03) == True
