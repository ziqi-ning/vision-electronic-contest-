"""
模拟雷达数据源
Phase 3 T3.1：解耦 ROS 依赖
用于无 ROS 环境的开发/测试场景
"""

import asyncio
import time
import random
import math
from typing import Optional, List
import logging

from .base import RadarSource, RadarScanResult, RadarPoint

logger = logging.getLogger(__name__)


class SimRadarSource(RadarSource):
    """
    模拟雷达数据源。

    在无 ROS 环境下，提供模拟的雷达扫描数据。
    可用于本地开发、Windows 环境测试等场景。

    模拟规则：
    - 生成 360° 均匀分布的模拟点
    - 大部分区域返回空旷（inf），偶尔返回模拟障碍物
    - 模拟相机正前方有一固定障碍物，便于调试融合算法
    """

    def __init__(self,
                 front_distance: float = 2.0,
                 obstacle_probability: float = 0.1,
                 num_simulated_points: int = 360,
                 max_distance: float = 10.0):
        """
        Args:
            front_distance: 模拟正前方障碍物距离（米），默认 2.0m
            obstacle_probability: 随机生成额外障碍物的概率
            num_simulated_points: 每帧模拟的点数量（分辨率）
            max_distance: 最大模拟距离（米）
        """
        self.front_distance = front_distance
        self.obstacle_probability = obstacle_probability
        self.num_simulated_points = num_simulated_points
        self.max_distance = max_distance
        self._available = True

        logger.info(
            f"模拟雷达数据源已初始化（正前方障碍物: {front_distance}m, "
            f"模拟点数: {num_simulated_points}）"
        )

    def is_available(self) -> bool:
        return self._available

    async def get_scan(self) -> Optional[RadarScanResult]:
        if not self._available:
            return None

        await asyncio.sleep(0.05)

        angle_step = 2 * math.pi / self.num_simulated_points
        points: List[RadarPoint] = []

        for i in range(self.num_simulated_points):
            angle = -math.pi + i * angle_step

            dist = self.max_distance

            if abs(angle) < 0.1:
                dist = self.front_distance
            elif random.random() < self.obstacle_probability:
                dist = random.uniform(0.5, self.max_distance)

            points.append(RadarPoint(distance_m=dist, angle_rad=angle))

        return RadarScanResult(
            timestamp=time.time(),
            points=points,
            obstacles=[]
        )
