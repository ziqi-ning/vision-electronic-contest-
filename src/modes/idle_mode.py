"""
空闲模式处理器 (0x00)
Phase 2 T2.3：拆分 main.py → src/modes/
"""

import asyncio
import cv2
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.colorblob as colorblob
import src.config.hardware as hardware
from .base import ModeHandler


class IdleMode(ModeHandler):
    """
    空闲模式 (0x00)：
    检测红色色块 → 取最大色块图送雷达测距 → 显示测距结果。
    """

    MODE_ID = 0x00

    async def process(self, frame) -> List:
        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        cnt_efc = colorblob.detect_color_to_rect(frame, "red")
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
            cv2.imshow("radar_draw", radar_draw)

        return cnt_efc

    async def _measure_async(self, center):
        """
        异步执行雷达测距逻辑，等价于 main.py 的 measure()。
        """
        import src.config.hardware as hardware

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

        cv2.circle(img, pixel_center, 2, (0, 0, 255), thickness=1, lineType=8)
        cv2.putText(
            img,
            f"distance: {distance:.2f}cm, angle={angle:.3f}",
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
