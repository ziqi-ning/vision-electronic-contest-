"""pytest 配置和共享 fixtures"""

# 必须在所有 src 模块导入前 mock pyzbar（src/other.py 顶层 import pyzbar）
import sys as _sys, os as _os
_mock_dir = _os.path.dirname(_os.path.abspath(__file__))
if _mock_dir not in _sys.path:
    _sys.path.insert(0, _mock_dir)
from _mock_pyzbar import *   # 注册 pyzbar mocks

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2


@pytest.fixture
def blank_frame():
    """640x480 空白 BGR 图像"""
    return np.full((480, 640, 3), 240, dtype=np.uint8)


@pytest.fixture
def red_rect_frame():
    """含红色矩形区域的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    frame[150:330, 250:390] = (0, 0, 255)
    return frame


@pytest.fixture
def green_rect_frame():
    """含绿色矩形区域的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    frame[100:300, 200:400] = (0, 200, 0)
    return frame


@pytest.fixture
def multi_color_frame():
    """含红+绿两个色块的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    frame[100:300, 50:200]   = (0, 0, 255)   # 红色
    frame[100:300, 440:590]  = (0, 200, 0)   # 绿色
    return frame


@pytest.fixture
def trapezoid_frame():
    """含红色梯形区域的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    pts = np.array([[180, 150], [460, 150], [420, 330], [220, 330]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 255))
    return frame


@pytest.fixture
def triangle_frame():
    """含红色三角形区域的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    pts = np.array([[320, 120], [180, 360], [460, 360]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 255))
    return frame


@pytest.fixture
def ellipse_frame():
    """含红色椭圆区域的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (150, 100), 0, 0, 360, (0, 0, 255), -1)
    return frame


@pytest.fixture
def pole_frame():
    """含红色平行双线的测试帧（杆子）"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.line(frame, (308, 50),  (308, 430), (0, 0, 255), 3)
    cv2.line(frame, (332, 50),  (332, 430), (0, 0, 255), 3)
    return frame


@pytest.fixture
def qr_frame():
    """含模拟 QR 码区域的测试帧"""
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.rectangle(frame, (250, 190), (390, 330), (0, 0, 0), -1)
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                cv2.rectangle(frame,
                              (250 + i * 14, 190 + j * 14),
                              (250 + i * 14 + 14, 190 + j * 14 + 14),
                              (255, 255, 255), -1)
    return frame
