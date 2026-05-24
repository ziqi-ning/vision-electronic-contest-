"""
相机-雷达融合模块
Phase 3 T3.1：解耦 ROS 依赖

将 radar5.py 的核心算法（像素→雷达距离映射、角度范围测距、障碍物检测）
封装为 RadarFusion 类，对上层（ModeHandler）提供统一接口，
不感知底层数据源是 ROS 还是模拟数据。
"""

import asyncio
import math
import numpy as np
from typing import Optional, Tuple, List, Any
import logging

from .base import RadarSource, RadarScanResult, RadarPoint

logger = logging.getLogger(__name__)


class RadarFusion:
    """
    相机-雷达融合器。

    封装原有 radar5.py 的核心逻辑：
    - angle_to_distance: 指定角度范围 → 最近障碍物距离和角度
    - site_to_distance: 指定像素坐标 → 融合相机外参后测距
    - get_obstacle: 获取所有检测到的障碍物

    底层数据源通过 RadarSource 抽象接口注入，
    支持 ROS 数据源和模拟数据源自动切换。
    """

    def __init__(self, source: RadarSource = None):
        """
        Args:
            source: 雷达数据源，不传则自动检测可用数据源
        """
        self._source = source
        self._last_scan: Optional[RadarScanResult] = None
        self._last_scan_time: float = 0

        if source is None:
            self._source = self._auto_detect_source()

        logger.info(f"RadarFusion 初始化，数据源: {type(self._source).__name__}")

    def _auto_detect_source(self) -> RadarSource:
        """自动检测并返回可用的雷达数据源"""
        from .ros_source import ROSRadarSource, _ROS_AVAILABLE as ros_available

        if ros_available:
            try:
                ros_source = ROSRadarSource()
                if ros_source.is_available():
                    logger.info("自动选择 ROS 雷达数据源")
                    return ros_source
            except Exception:
                pass

        from .sim_source import SimRadarSource
        logger.info("自动选择模拟雷达数据源（ROS 不可用）")
        return SimRadarSource()

    async def _get_scan(self, timeout: float = 0.1) -> Optional[RadarScanResult]:
        """获取最新扫描数据，带缓存（100ms 内复用）"""
        now = asyncio.get_event_loop().time()
        if now - self._last_scan_time < 0.1:
            return self._last_scan

        scan = await self._source.get_scan()
        if scan is not None:
            self._last_scan = scan
            self._last_scan_time = now
        return scan

    async def angle_to_distance(self, search_start: float,
                                 search_end: float,
                                 timeout: float = 3.0) -> Tuple[int, int]:
        """
        指定角度范围 → 最近障碍物距离（cm）和角度（百分之一度）

        等价于 radar5.py 的 angle_to_distance()。
        角度使用竞赛协议标准：度数（而非弧度），返回 (距离_cm, 角度_centidegree)

        Args:
            search_start: 搜索起始角度（度）
            search_end: 搜索结束角度（度）
            timeout: 超时时间（秒）

        Returns:
            (距离_cm, 角度_centidegree)，无数据时返回 (3000, 40000)
        """
        scan = await self._get_scan(timeout)
        if scan is None or not scan.points:
            logger.warning("angle_to_distance: 无雷达数据")
            return 3000, 40000

        def angle_in_range(angle_rad: float) -> bool:
            if search_start <= search_end:
                start_rad = math.radians(search_start - 180)
                end_rad = math.radians(search_end - 180)
                return start_rad <= angle_rad <= end_rad
            else:
                start_rad = math.radians(search_start - 180)
                end_rad = math.radians(search_end - 180)
                return angle_rad >= start_rad or angle_rad <= end_rad

        valid_points = [p for p in scan.points if angle_in_range(p.angle_rad)]

        if not valid_points:
            logger.debug(f"angle_to_distance: 范围[{search_start}°, {search_end}°]内无有效点")
            return 3000, 40000

        min_point = min(valid_points, key=lambda p: p.distance_m)
        dist_cm = int(min_point.distance_m * 100)

        angle_deg = math.degrees(min_point.angle_rad) + 180
        angle_centideg = int(angle_deg * 100)

        return dist_cm, angle_centideg

    async def site_to_distance(self, u: float, v: float,
                               camera_params: Tuple) -> Tuple[int, int]:
        """
        指定像素坐标 → 融合相机-雷达外参后测距

        等价于 radar5.py 的 site_to_distance()。
        核心算法：
        1. 像素坐标 → 归一化相机坐标系
        2. 应用俯仰角旋转
        3. 计算射线与地面交点
        4. 转换到雷达坐标系
        5. 在雷达数据中匹配对应角度的点

        Args:
            u: 像素 x 坐标
            v: 像素 y 坐标
            camera_params: 相机参数元组
                (fx, fy, cx, cy, delta_x, delta_y, delta_z,
                 camera_pitch_deg, angle_tolerance_rad, camera_height)

        Returns:
            (距离_cm, 角度_centidegree)，无匹配时返回 (3000, 40000)
        """
        scan = await self._get_scan()
        if scan is None or not scan.points:
            logger.warning("site_to_distance: 无雷达数据")
            return 3000, 40000

        (fx, fy, cx, cy,
         delta_x, delta_y, delta_z,
         camera_pitch_deg, angle_tolerance_rad,
         camera_height) = camera_params

        x = (u - cx) / fx
        y = (cy - v) / fy
        z = 1.0

        if abs(v - cy) < 1e-3 or abs(y) < 1e-3:
            logger.info("正前方无穷远")

        pitch = math.radians(camera_pitch_deg)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        rotation_matrix = np.array([
            [cos_p, 0, -sin_p],
            [0, 1, 0],
            [sin_p, 0, cos_p]
        ])
        dir_vec = rotation_matrix @ np.array([x, y, z])

        if dir_vec[1] < 0:
            dir_vec[1] = -dir_vec[1]

        vertical_component = dir_vec[1]
        if abs(vertical_component) < 1e-6:
            sign = 1 if vertical_component >= 0 else -1
            vertical_component = sign * 1e-6
            logger.warning("校正近零垂直分量")

        t = -camera_height / vertical_component
        x_ground = dir_vec[0] * t
        z_ground = dir_vec[2] * t

        x_relative = x_ground - delta_x
        z_relative = z_ground - delta_y

        radar_angle = math.atan2(x_relative, z_relative)

        best_point = None
        best_diff = float('inf')
        for point in scan.points:
            diff = abs(point.angle_rad - radar_angle)
            diff = min(diff, 2 * math.pi - diff)
            if diff < best_diff:
                best_diff = diff
                best_point = point

        if best_point is None or best_diff >= angle_tolerance_rad:
            logger.warning(
                f"site_to_distance: 像素({u},{v})无匹配，diff={math.degrees(best_diff):.2f}° "
                f"> tolerance={math.degrees(angle_tolerance_rad):.2f}°"
            )
            return 3000, 40000

        best_angle_deg = math.degrees(best_point.angle_rad)
        if best_angle_deg < 0:
            best_angle_deg += 360
        if 90 < best_angle_deg < 270:
            best_angle_deg = 180 + best_angle_deg
            if best_angle_deg > 360:
                best_angle_deg -= 360

        if best_point.distance_m > 30.0:
            return 3000, 40000

        return int(best_point.distance_m * 100), int(best_angle_deg * 100)

    async def get_obstacle(self, timeout: float = 3.0) -> List[Tuple[int, int]]:
        """
        获取所有检测到的障碍物（点群 + 孤立点）

        等价于 radar5.py 的 get_obstacle()。
        障碍物判定逻辑：
        1. 点群：连续 3 个以上距离差 ≤3cm 的点聚成一组
        2. 孤立点：比周围点/点群近 3cm 以上

        Returns:
            List[(距离_cm, 角度_centidegree)]，无障碍物时返回 [(3000, 40000)]
        """
        scan = await self._get_scan(timeout)
        if scan is None or not scan.points:
            return [(3000, 40000)]

        MAX_DISTANCE = 10.0
        CLUSTER_THRESHOLD = 0.03

        ranges = np.array([p.distance_m for p in scan.points])
        angles = np.array([p.angle_rad for p in scan.points])

        obstacles = self._detect_obstacles(ranges, angles, MAX_DISTANCE, CLUSTER_THRESHOLD)

        if not obstacles:
            return [(3000, 40000)]

        result = []
        for angle_rad, dist_m in obstacles:
            angle_deg = math.degrees(angle_rad + math.pi)
            if angle_deg < 0:
                angle_deg += 360
            result.append((int(dist_m * 100), int(angle_deg * 100)))

        return result

    def _detect_obstacles(self, ranges: np.ndarray, angles: np.ndarray,
                          max_dist: float, threshold: float) -> List[Tuple[float, float]]:
        """检测点群障碍物和孤立障碍点"""
        obstacles = []
        n = len(ranges)

        clusters = []
        current_cluster = []

        for i in range(n):
            dist = ranges[i]
            if dist > max_dist:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = []
                continue

            if not current_cluster:
                current_cluster.append(i)
                continue

            last_index = current_cluster[-1]
            diff = abs(dist - ranges[last_index])

            if diff <= threshold:
                current_cluster.append(i)
            else:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = [i]

        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        clustered_points = set()
        for cluster in clusters:
            cluster_dists = ranges[cluster]
            cluster_angles = angles[cluster]
            median_angle = float(np.median(cluster_angles))
            mean_distance = float(np.mean(cluster_dists))
            obstacles.append((median_angle, mean_distance))
            clustered_points.update(cluster)

        for i in range(n):
            if i in clustered_points or ranges[i] > max_dist:
                continue
            if self._is_valid_isolated_point(i, ranges, angles, clusters, max_dist, threshold):
                obstacles.append((float(angles[i]), float(ranges[i])))

        return obstacles

    def _is_valid_isolated_point(self, idx: int, ranges: np.ndarray, angles: np.ndarray,
                                  clusters: List[List[int]], max_dist: float,
                                  threshold: float) -> bool:
        """判断是否为有效孤立障碍点"""
        left_diff = float('inf')
        right_diff = float('inf')

        for i in range(idx - 1, -1, -1):
            if ranges[i] <= max_dist:
                left_diff = abs(ranges[idx] - ranges[i])
                break

        for i in range(idx + 1, len(ranges)):
            if ranges[i] <= max_dist:
                right_diff = abs(ranges[idx] - ranges[i])
                break

        nearest_cluster_dist = float('inf')
        for cluster in clusters:
            cluster_dist = float(np.mean(ranges[cluster]))
            dist_diff = abs(ranges[idx] - cluster_dist)
            if dist_diff < nearest_cluster_dist:
                nearest_cluster_dist = dist_diff

        return (left_diff > threshold and
                right_diff > threshold and
                ranges[idx] < nearest_cluster_dist - 0.03)
