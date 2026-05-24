"""
雷达模块
Phase 3 T3.1：解耦 ROS 依赖

导出：
- RadarSource: 抽象基类
- RadarFusion: 融合器（自动检测 ROS/模拟数据源）
- ROSRadarSource: ROS 数据源（lazy import）
- SimRadarSource: 模拟数据源
"""

from .base import RadarSource, RadarScanResult, RadarPoint
from .fusion import RadarFusion
from .sim_source import SimRadarSource

__all__ = [
    "RadarSource",
    "RadarScanResult",
    "RadarPoint",
    "RadarFusion",
    "SimRadarSource",
]
