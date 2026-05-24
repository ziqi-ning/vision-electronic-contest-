"""
QR 模式处理器 (0x07)
Phase 2 T2.3：拆分 main.py → src/modes/
"""

import cv2
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import ModeHandler


# QR 检测器（全局单例，与 main.py 一致）
QR_detector = cv2.QRCodeDetector()


class QRMode(ModeHandler):
    """
    QR 模式 (0x07) — 24年真题模式：
    识别二维码和 AprilTag，写入 target.apriltag_id / x / y / pixel。
    """

    MODE_ID = 0x07

    async def process(self, frame) -> List:
        import src.other as other

        self.target.flag = 0
        self.target.img_width = 640
        self.target.img_height = 480

        result = other.QR_detect(QR_detector, frame)

        if len(result) < 5:
            return []

        _, flag, data, x, y, pixel = result
        self.target.flag = flag
        self.target.apriltag_id = data
        self.target.x = x
        self.target.y = y
        self.target.pixel = pixel

        return [result]
