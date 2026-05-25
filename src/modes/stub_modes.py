"""
Stub 模式处理器 (0x01~0x06) — Phase 3 T3.4 实现

补全各模式检测逻辑：
- 0x01 CircleMode: 白色色块内椭圆检测
- 0x02 SoundMode: 雷达听声辩位（纯雷达测角，无相机）
- 0x03 IdleModeAlt: 绿色色块 + 雷达测距（0x00 变体）
- 0x04 AprilTagMode: AprilTag 检测 + 位姿估计
- 0x05 ColorBlockMode: 蓝色色块检测
- 0x06 BarcodeMode: 条形码/二维码检测
"""

from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.colorblob as colorblob
import src.outsite as outsite
import src.other as other
import src.config.hardware as hardware
from .base import ModeHandler


# AprilTag 检测器（全局单例，与 qr_mode.py 保持一致）
_apriltag_detector = None


def _get_apriltag_detector():
    global _apriltag_detector
    if _apriltag_detector is None:
        import apriltag
        options = apriltag.DetectorOptions(
            families=hardware.AprilTagConfig.TAG_FAMILY
        )
        _apriltag_detector = apriltag.Detector(options)
    return _apriltag_detector


# ============================================================
# 0x01 圆形检测模式 — 检测白色色块中的椭圆
# ============================================================
class CircleMode(ModeHandler):
    """
    圆形检测模式 (0x01)：
    白色色块 → 取最大色块 ROI → 椭圆检测。
    """

    MODE_ID = 0x01

    async def process(self, frame) -> List:
        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        # 白色色块检测
        cnt_efc = colorblob.detect_color_to_rect(frame, "white")
        if not cnt_efc:
            return []

        result_img = cnt_efc[0]["result"]
        center = cnt_efc[0]["center"]

        # 在 ROI 内检测椭圆（取第二大椭圆，排除圆形 ROI 边框）
        _, _, _, ellipse_center, r_max = outsite.detect_ellipses(result_img)

        if ellipse_center == (0, 0) or r_max == 0:
            return []

        self.target.flag = 1
        self.target.x = ellipse_center[0]
        self.target.y = ellipse_center[1]
        self.target.pixel = r_max

        return cnt_efc


# ============================================================
# 0x02 听声辩位模式 — 纯雷达扫描，无相机检测
# ============================================================
class SoundMode(ModeHandler):
    """
    听声辩位雷达模式 (0x02)：
    不依赖相机，纯用雷达扫描最近障碍物角度。
    """

    MODE_ID = 0x02

    async def process(self, frame) -> List:
        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        # 获取最近障碍物角度（纯雷达）
        obstacles = await self.radar.get_obstacle()
        if obstacles:
            dist, angle = obstacles[0]
            if dist < 3000:
                self.target.flag = 1
                self.target.distance = dist
                self.target.angle = angle

        return []


# ============================================================
# 0x03 空闲模式备用 — 绿色色块 + 雷达测距（0x00 变体）
# ============================================================
class IdleModeAlt(ModeHandler):
    """
    空闲模式备用 (0x03)：
    检测绿色色块 → 取最大色块图送雷达测距 → 显示测距结果。
    等价于 IdleMode，但颜色为绿色。
    """

    MODE_ID = 0x03

    async def process(self, frame) -> List:
        import asyncio
        import cv2

        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        cnt_efc = colorblob.detect_color_to_rect(frame, "green")
        if not cnt_efc:
            return []

        result_img = cnt_efc[0]["result"]
        center = cnt_efc[0]["center"]

        await self._queue_frame_for_radar(result_img)

        loop = asyncio.get_running_loop()
        loop.create_task(self._measure_async(center))

        try:
            radar_draw = self.queue_radar_draw.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
            return []

        if radar_draw is not None:
            cv2.imshow("radar_draw_alt", radar_draw)

        return cnt_efc

    async def _measure_async(self, center):
        x, y = center[0], center[1]
        pixel_center = (int(x), int(y))
        pixel_center2 = (int(x), int(y) + 20)

        try:
            img = self.queue_radar_test.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
            img = None

        if img is None:
            return

        camera_params = hardware.camera_params
        distance, angle = await self.radar.site_to_distance(x, y, camera_params)

        cv2.circle(img, pixel_center, 2, (0, 255, 0), thickness=1, lineType=8)
        cv2.putText(
            img,
            f"dist: {distance:.2f}cm, angle={angle:.3f}",
            pixel_center,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        after_deal = await self.radar.get_obstacle()

        angles = [item[1] for item in after_deal]
        candidate = [(distance, angle)]
        left_diffs = []
        right_diffs = []

        for i, item in enumerate(angles):
            angle_diff = abs(angle - item)
            if angle_diff <= 18000:
                left_diffs.append((angle_diff, i))
            else:
                right_diffs.append((36000 - angle_diff, i))

        left_diffs = sorted(left_diffs, key=lambda x: x[0])
        right_diffs = sorted(right_diffs, key=lambda x: x[0])

        for left in left_diffs[:2]:
            if left[0] < hardware.angle_tolerance_rad:
                candidate.append(after_deal[left[1]])

        for right in right_diffs[:2]:
            if right[0] < hardware.angle_tolerance_rad:
                candidate.append(after_deal[right[1]])

        closest_obstacle = min(candidate, key=lambda x: x[0])
        closest_angle = closest_obstacle[1]
        closest_range = closest_obstacle[0]

        cv2.putText(
            img,
            f"near: {closest_range:.2f}cm, angle:{closest_angle}",
            pixel_center2,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        if self.queue_radar_draw.full():
            self.queue_radar_draw.get_nowait()
        await self.queue_radar_draw.put(img)


# ============================================================
# 0x04 AprilTag 模式 — 检测 AprilTag 并计算位姿
# ============================================================
class AprilTagMode(ModeHandler):
    """
    AprilTag 检测模式 (0x04)：
    识别 AprilTag，计算相机坐标系下的位姿，
    写入 target.apriltag_id / x / y / pixel / distance。
    """

    MODE_ID = 0x04

    async def process(self, frame) -> List:
        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        detector = _get_apriltag_detector()

        x, y, tag_id, side_len, flag, px, py, pz = other.opencv_find_april_tag(
            frame, hardware.cam_info, detector
        )

        if flag == 0 or tag_id == 0:
            return []

        self.target.flag = 1
        self.target.apriltag_id = tag_id
        self.target.x = x
        self.target.y = y
        self.target.pixel = side_len * side_len
        self.target.distance = int(pz)

        return [{"tag_id": tag_id, "x": x, "y": y, "pz": pz}]


# ============================================================
# 0x05 色块模式 — 检测蓝色色块并发送结果
# ============================================================
class ColorBlockMode(ModeHandler):
    """
    色块检测模式 (0x05)：
    检测蓝色色块，取最大色块中心坐标和面积，写入 target。
    """

    MODE_ID = 0x05

    async def process(self, frame) -> List:
        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        cnt_efc = colorblob.detect_color_to_rect(frame, "blue")
        if not cnt_efc:
            return []

        # 取最大色块
        best = cnt_efc[0]
        center = best["center"]
        pixels_max = best["pixels_max"]

        self.target.flag = 1
        self.target.x = center[0]
        self.target.y = center[1]
        self.target.pixel = min(pixels_max, 60000)

        return cnt_efc


# ============================================================
# 0x06 条形码模式 — 检测条形码/二维码并解析内容
# ============================================================
class BarcodeMode(ModeHandler):
    """
    条形码检测模式 (0x06)：
    使用 pyzbar 检测条形码和二维码，
    写入 target.apriltag_id（编码内容）/ x / y / pixel。
    """

    MODE_ID = 0x06

    async def process(self, frame) -> List:
        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        # decodeDisplay 返回：result_img, x, y, apriltag_id, flag
        result_img, x, y, apriltag_id, flag = other.decodeDisplay(frame)

        if flag == 0:
            return []

        self.target.flag = 1
        self.target.apriltag_id = apriltag_id
        self.target.x = x
        self.target.y = y

        return [{"x": x, "y": y, "barcode_id": apriltag_id}]
