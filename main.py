#!/usr/bin/env python3
"""
主程序 — Phase 2 T2.4 简化版
相机异步取帧 → 模式处理器执行检测 → 串口发送 + LED 指示

对比原版 380 行 main.py：
- 去除硬编码的 8 种分支逻辑，全部迁移到 src/modes/
- 串口通信封装到 src/comm/SerialClient
- 目标：逻辑清晰，main.py 只负责协调整体流程
"""

import asyncio
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comm import SerialClient
from src.modes import MODES
from src.modes.base import TargetData
from src.config import hardware
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 协程队列（模块级，跨协程共享）
# ============================================================
queue_pictures = asyncio.Queue(maxsize=2)    # 原始帧队列
queue_radar_test = asyncio.Queue(maxsize=1)  # 雷达测距用帧队列
queue_radar_draw = asyncio.Queue(maxsize=1)  # 雷达绘图结果队列


# ============================================================
# 协程 1：异步从相机取帧，放入队列
# ============================================================
async def video_get(cam):
    """
    等价于原 main.py 的 video_get()。
    持续从 VideoCapture 读帧，翻转后放入 queue_pictures。
    """
    while True:
        ret, frame = cam.read()
        if ret:
            frame = cv2.flip(frame, 0, None)
            if queue_pictures.full():
                queue_pictures.get_nowait()
            await queue_pictures.put(frame)
        await asyncio.sleep(0)


# ============================================================
# 协程 2：主数据处理循环
# ============================================================
async def deal_data(serial, radar, light):
    """
    等价于原 main.py 的 deal_data()。

    职责：
    - 从 queue_pictures 取帧
    - 读取串口模式码
    - 调用对应 ModeHandler.process()
    - 发送串口结果 + 更新 LED + 统计 FPS
    """
    ctr = serial.ctr          # ModeCtrl，模式码由 SerialClient 内部解析
    target = TargetData()
    target.img_width = hardware.CameraConfig.IMAGE_WIDTH
    target.img_height = hardware.CameraConfig.IMAGE_HEIGHT

    # 初始化 8 种模式的处理器（按 Phase 2 T2.3 的 MODES 字典）
    handlers = {
        mode_id: cls(
            cap=None,
            serial=serial,
            radar=radar,
            pipeline=None,
            target=target,
            ctr=ctr,
            queue_radar_test=queue_radar_test,
            queue_radar_draw=queue_radar_draw,
        )
        for mode_id, cls in MODES.items()
    }

    # FPS 计时
    pre_tick = cv2.getTickCount()
    count = 0
    colors = ["empty", "white", "red", "yellow", "green",
              "indigo", "blue", "purple"]

    while True:
        # 取帧
        try:
            frame = queue_pictures.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
            continue

        # 读取串口模式码
        serial.read_mode()

        # 模式路由：找到对应处理器，执行检测
        handler = handlers.get(ctr.work_mode)
        if handler:
            await handler.process(frame)

        # LED 指示灯（与原逻辑一致）
        if target.flag == 1:
            count += 1
            if count > 10:
                count = 0
                try:
                    await light.setColor("indigo")
                except Exception:
                    pass
        else:
            try:
                await light.setColor(colors[ctr.work_mode & 0x07])
            except Exception:
                pass

        # 发送串口数据
        serial.send(ctr.work_mode, target)

        # FPS 统计（与原逻辑一致）
        cur_tick = cv2.getTickCount()
        dt_ms = 1000.0 * (cur_tick - pre_tick) / cv2.getTickFrequency()
        target.fps = min(int(1000.0 / dt_ms), 255) if dt_ms > 0 else 0
        pre_tick = cur_tick

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ============================================================
# 主入口：初始化资源，启动协程
# ============================================================
async def main():
    # ---- 相机 ----
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, hardware.CameraConfig.IMAGE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, hardware.CameraConfig.IMAGE_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, hardware.CameraConfig.FPS)
    if not cam.isOpened():
        logger.error("Cannot open camera")
        return

    # ---- 串口 ----
    serial = SerialClient()
    if not serial.open():
        logger.error("serialport init fail")
        return
        logger.info("serialport init: %s @ %s", serial.port, serial.baudrate)

    # ---- 雷达（Phase 3 T3.1：使用 RadarFusion 自动检测数据源） ----
    from src.radar import RadarFusion
    radar = RadarFusion()

    # ---- LED ----
    try:
        import src.facility2 as facility2
        light = facility2.light()
    except Exception:
        light = None

    # 启动协程
    await asyncio.gather(
        video_get(cam),
        deal_data(serial, radar, light),
    )

    # 清理
    cam.release()
    cv2.destroyAllWindows()
    serial.close()


if __name__ == "__main__":
    asyncio.run(main())
