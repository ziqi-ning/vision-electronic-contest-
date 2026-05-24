"""
日志工具模块
Phase 3 T3.2：统一日志系统

提供全项目统一的日志接口，按以下级别分类：
- DEBUG:   每帧 FPS、检测数量、中间变量
- INFO:    模式切换、串口收发、心跳
- WARNING: 降级操作（如雷达断线切换 SimRadarSource）
- ERROR:   异常堆栈
"""

import logging
import sys
from pathlib import Path

_LOGGERS: dict = {}

# 日志格式模板
_DEFAULT_FORMAT = "[%(asctime)s] %(name)s:%(levelname)s - %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取或创建指定名称的 Logger。

    所有模块统一使用此函数获取 Logger，避免重复创建 handler。

    Args:
        name: Logger 名称，通常传 __name__
        level: 日志级别，默认 INFO

    Returns:
        配置好的 Logger 实例
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已注册的 Logger（如未注册则用 INFO 级别自动注册）。

    便捷别名，模块内部调用示例：
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    if name not in _LOGGERS:
        return setup_logger(name)
    return _LOGGERS[name]


def set_global_level(level: int):
    """
    设置所有已注册 Logger 的全局级别。
    方便在启动时一键关闭 DEBUG 输出。
    """
    for logger in _LOGGERS.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
