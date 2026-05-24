"""
串口通信客户端
封装 uartuse.py 中的模式解析和串口打包逻辑，供 ModeHandler 使用。
Phase 2 T2.4：简化 main.py
"""

import asyncio
from typing import Optional

import serial

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.uartuse as uartuse


class SerialClient:
    """
    串口通信客户端。

    封装：
    - 串口初始化与读写
    - 模式码解析（uart_data_prase / uart_data_read）
    - 数据打包发送（package_blobs_data）
    - 模式码获取（get_mode）
    """

    BAUDRATE = 256000
    TIMEOUT = 0.5
    PORT = "/dev/ttyAMA4"

    def __init__(self, port: str = PORT, baudrate: int = BAUDRATE, timeout: float = TIMEOUT):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

        self._ser: Optional[serial.Serial] = None
        self._ctr = uartuse.ModeCtrl()
        self._R = uartuse.UartBufParse()

    def open(self) -> bool:
        """打开串口，返回是否成功。"""
        try:
            self._ser = serial.Serial(
                self._port, self._baudrate, timeout=self._timeout
            )
            self._ser.close()
            self._ser.open()
            return self._ser.is_open
        except Exception:
            return False

    def close(self):
        """关闭串口。"""
        if self._ser and self._ser.is_open:
            self._ser.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @property
    def port(self) -> str:
        return self._port

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @property
    def work_mode(self) -> int:
        return self._ctr.work_mode

    @property
    def ctr(self):
        """ModeCtrl 控制器（模式码由此解析）。"""
        return self._ctr

    def get_mode(self) -> int:
        """获取当前工作模式码。"""
        return self._ctr.work_mode

    async def serial_get(self):
        """
        协程：持续从串口读取字节，解析模式码。
        等价于原 main.py 的 serial_get()。
        """
        print("serial has been ready")
        while self.is_open:
            try:
                if self._ser.in_waiting > 0:
                    buf = self._ser.read(self._ser.in_waiting)
                    for byte in buf:
                        uartuse.uart_data_prase(self._R, byte, self._ctr)
            except Exception:
                pass
            await asyncio.sleep(0)

    def read_mode(self):
        """
        同步读取串口并解析模式码。
        供主循环每帧调用。
        """
        if not self.is_open:
            return
        try:
            if self._ser.in_waiting > 0:
                buf = self._ser.read(self._ser.in_waiting)
                for byte in buf:
                    uartuse.uart_data_prase(self._R, byte, self._ctr)
        except Exception:
            pass

    def write(self, data: bytes):
        """向串口写入字节数据。"""
        if self.is_open:
            self._ser.write(data)

    def send(self, mode: int, target) -> bool:
        """
        打包并发送目标数据。

        Args:
            mode: 当前工作模式码
            target: TargetData / TargetCheck 对象

        Returns:
            bool: 是否发送成功
        """
        if not self.is_open:
            return False
        try:
            package = uartuse.package_blobs_data(mode, target)
            self._ser.write(package)
            return True
        except Exception:
            return False
