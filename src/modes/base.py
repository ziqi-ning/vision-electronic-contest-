"""
模式处理器抽象基类
Phase 2 T2.3：拆分 main.py → src/modes/
Phase 2 T2.4：增加 SerialClient 支持
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Union
import numpy as np

if TYPE_CHECKING:
    from src.pipeline.orchestrator import DetectionPipeline
    from src.comm.serial_client import SerialClient


class TargetData:
    """
    兼容 uartuse.TargetCheck 的数据结构，
    存储检测结果并由 SerialClient 打包发送。
    """

    def __init__(self):
        self.flag: int = 0
        self.apriltag_id: int = 0
        self.x: int = 0
        self.y: int = 0
        self.pixel: int = 0
        self.img_width: int = 640
        self.img_height: int = 480
        self.fps: int = 0
        self.state: int = 0
        self.angle: int = 0
        self.distance: int = 0
        self.reserved1: int = 0
        self.reserved2: int = 0
        self.reserved3: int = 0
        self.reserved4: int = 0
        self.range_sensor1: int = 0
        self.range_sensor2: int = 0
        self.range_sensor3: int = 0
        self.range_sensor4: int = 0
        self.camera_id: int = 0x02


class ModeHandler(ABC):
    """
    模式处理抽象基类。

    main.py 的 deal_data() 中 8 种工作模式 (0x00~0x07) 各自对应一个 ModeHandler 子类。
    所有子类共享：
        - cap: 相机 VideoCapture 对象
        - serial: SerialClient 串口客户端
        - radar: 雷达管理器
        - pipeline: 检测编排器
        - target: TargetData 结果容器
        - ctr: ModeCtrl 模式控制器
        - queue_radar_test: 雷达测距用帧队列
        - queue_radar_draw: 雷达绘图结果队列
    """

    MODE_ID: int = 0x00

    def __init__(self,
                 cap,
                 serial: "SerialClient",
                 radar,
                 pipeline: Optional["DetectionPipeline"],
                 target: "TargetData",
                 ctr,
                 queue_radar_test,
                 queue_radar_draw):
        self.cap = cap
        self.serial = serial
        self.radar = radar
        self.pipeline = pipeline
        self.target = target
        self.ctr = ctr
        self.queue_radar_test = queue_radar_test
        self.queue_radar_draw = queue_radar_draw

    @abstractmethod
    async def process(self, frame: np.ndarray) -> List:
        """
        处理一帧图像。

        Args:
            frame: BGR 格式输入帧

        Returns:
            List: 检测结果列表
        """
        pass

    async def _queue_frame_for_radar(self, result_img: np.ndarray):
        """将检测结果图放入雷达测距队列。"""
        try:
            if self.queue_radar_test.full():
                self.queue_radar_test.get_nowait()
            await self.queue_radar_test.put(result_img)
        except Exception:
            pass

    def _send_result(self):
        """将 target 数据通过 SerialClient 发送（兼容旧接口）。"""
        if self.serial and self.serial.is_open:
            self.serial.send(self.ctr.work_mode, self.target)

    def _update_fps(self, pre_tick: float, cur_tick: float, last_fps: int) -> int:
        """更新 FPS 记录，返回当前帧的 FPS。"""
        import cv2
        dt_ms = 1000.0 * (cur_tick - pre_tick) / cv2.getTickFrequency()
        fps = int(1000.0 / dt_ms) if dt_ms > 0 else 0
        self.target.fps = min(fps, 255)
        return fps
