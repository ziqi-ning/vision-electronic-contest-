"""
雷达数据源抽象基类
Phase 3 T3.1：解耦 ROS 依赖
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class RadarPoint:
    """单点雷达数据"""
    distance_m: float  # 距离（米）
    angle_rad: float   # 角度（弧度，0为正前方，逆时针为正）


@dataclass
class RadarScanResult:
    """一次完整雷达扫描结果"""
    timestamp: float
    points: List[RadarPoint]  # 所有有效点
    obstacles: List[RadarPoint] = None  # 障碍物点群

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []


class RadarSource(ABC):
    """
    雷达数据源抽象基类。

    定义统一的雷达数据获取接口，不同数据源（ROS / 模拟 / 文件）
    均实现此接口，上层 RadarFusion 不感知具体实现。
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查数据源是否可用。

        Returns:
            True: 数据源正常工作
            False: 数据源不可用（如 ROS 未启动、无数据文件等）
        """
        pass

    @abstractmethod
    async def get_scan(self) -> Optional[RadarScanResult]:
        """
        获取最新一次雷达扫描数据。

        Returns:
            RadarScanResult: 扫描结果
            None: 获取失败（如超时、无数据）
        """
        pass
