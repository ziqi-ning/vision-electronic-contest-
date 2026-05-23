"""
工作模式枚举
"""

from enum import IntEnum


class WorkMode(IntEnum):
    IDLE = 0x00       # 空闲模式（红色检测 + 雷达测距）
    CIRCLE = 0x01     # 圆形检测模式
    SOUND = 0x02      # 听声辩位雷达模式
    IDLE2 = 0x03      # 空闲模式（备用）
    APRILTAG = 0x04   # AprilTag 模式
    COLOR_BLOB = 0x05 # 色块模式
    BARCODE = 0x06    # 条形码模式
    QR_2024 = 0x07    # 24年真题（QR 码）


MODE_NAMES = {
    WorkMode.IDLE: "idle",
    WorkMode.CIRCLE: "white",
    WorkMode.SOUND: "yellow",
    WorkMode.IDLE2: "green",
    WorkMode.APRILTAG: "indigo",
    WorkMode.COLOR_BLOB: "blue",
    WorkMode.BARCODE: "purple",
    WorkMode.QR_2024: "empty",
}
