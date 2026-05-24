"""
ROS 雷达数据源
Phase 3 T3.1：解耦 ROS 依赖
lazy import rospy，只有在 ROS 环境才加载
"""

import asyncio
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

from .base import RadarSource, RadarScanResult, RadarPoint


_ROS_AVAILABLE = False
_sensor_msgs = None
_rospy = None


def _try_import_ros():
    """延迟导入 ROS，仅在 ROS 环境存在时成功"""
    global _ROS_AVAILABLE, _sensor_msgs, _rospy
    if _ROS_AVAILABLE:
        return True
    try:
        import sys
        sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
        import rospy
        from sensor_msgs.msg import LaserScan
        _rospy = rospy
        _sensor_msgs = LaserScan
        _ROS_AVAILABLE = True
        return True
    except (ImportError, ModuleNotFoundError):
        _ROS_AVAILABLE = False
        return False


class ROSRadarSource(RadarSource):
    """
    ROS LaserScan 话题数据源。

    订阅 ROS 的 /scan 话题，将 sensor_msgs/LaserScan 转换为 RadarScanResult。
    仅在 ROS 环境存在时可用，否则 is_available() 返回 False。
    """

    def __init__(self, topic: str = "/scan", timeout: float = 1.0):
        self.topic = topic
        self.timeout = timeout
        self._latest_scan = None
        self._lock = asyncio.Lock()
        self._subscriber = None
        self._rospy_initialized = False

        if not _try_import_ros():
            logger.warning("ROS 不可用，ROSRadarSource 将不可用")
            return

        try:
            if not _rospy.core.is_initialized():
                _rospy.init_node('vision_radar_node', anonymous=True, disable_rosout=True, disable_signals=True)
            self._rospy_initialized = True
            self._subscriber = _rospy.Subscriber(
                self.topic, _sensor_msgs, self._callback, queue_size=1
            )
            logger.info(f"ROS 雷达数据源已订阅话题: {self.topic}")
        except Exception as e:
            logger.warning(f"ROS 雷达订阅失败: {e}")
            self._subscriber = None

    def _callback(self, msg):
        """ROS 回调：更新最新扫描数据"""
        import threading
        if hasattr(self, '_lock'):
            with self._lock:
                self._latest_scan = msg

    def is_available(self) -> bool:
        return _ROS_AVAILABLE and self._subscriber is not None and self._rospy_initialized

    async def get_scan(self) -> Optional[RadarScanResult]:
        if not self.is_available():
            return None

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            async with self._lock:
                msg = self._latest_scan
            if msg is not None:
                return self._convert_scan(msg)
            await asyncio.sleep(0.01)

        logger.warning("ROS 雷达数据获取超时")
        return None

    def _convert_scan(self, msg) -> RadarScanResult:
        """将 sensor_msgs/LaserScan 转换为 RadarScanResult"""
        import numpy as np

        ranges = msg.ranges
        n = len(ranges)

        angles = np.arange(
            msg.angle_min,
            msg.angle_min + n * msg.angle_increment,
            msg.angle_increment
        )[:n]

        points = []
        for dist, angle in zip(ranges, angles):
            if np.isfinite(dist) and dist > 0:
                points.append(RadarPoint(distance_m=float(dist), angle_rad=float(angle)))

        return RadarScanResult(
            timestamp=msg.header.stamp.to_sec() if hasattr(msg.header, 'stamp') else time.time(),
            points=points,
            obstacles=[]
        )
