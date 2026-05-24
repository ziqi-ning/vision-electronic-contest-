"""
Stub 模式处理器 (0x01~0x06)
尚未实现，各占位保留。
"""

from typing import List
from .base import ModeHandler


class CircleMode(ModeHandler):
    MODE_ID = 0x01
    async def process(self, frame) -> List:
        return []


class SoundMode(ModeHandler):
    MODE_ID = 0x02
    async def process(self, frame) -> List:
        return []


class IdleModeAlt(ModeHandler):
    MODE_ID = 0x03
    async def process(self, frame) -> List:
        return []


class AprilTagMode(ModeHandler):
    MODE_ID = 0x04
    async def process(self, frame) -> List:
        return []


class ColorBlockMode(ModeHandler):
    MODE_ID = 0x05
    async def process(self, frame) -> List:
        return []


class BarcodeMode(ModeHandler):
    MODE_ID = 0x06
    async def process(self, frame) -> List:
        return []
