"""
src.modes — 模式处理器模块
Phase 2 T2.3：拆分 main.py
"""

from .base import ModeHandler, TargetData
from .idle_mode import IdleMode
from .qr_mode import QRMode
from .stub_modes import (
    CircleMode,
    SoundMode,
    IdleModeAlt,
    AprilTagMode,
    ColorBlockMode,
    BarcodeMode,
)

MODES = {
    0x00: IdleMode,
    0x01: CircleMode,
    0x02: SoundMode,
    0x03: IdleModeAlt,
    0x04: AprilTagMode,
    0x05: ColorBlockMode,
    0x06: BarcodeMode,
    0x07: QRMode,
}

__all__ = [
    "ModeHandler",
    "TargetData",
    "IdleMode",
    "QRMode",
    "CircleMode",
    "SoundMode",
    "IdleModeAlt",
    "AprilTagMode",
    "ColorBlockMode",
    "BarcodeMode",
    "MODES",
]
