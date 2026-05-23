"""
场景参数配置
从 colorblob.py 提取的 HSV 颜色阈值及形态学参数，通过 YAML 加载
"""

import yaml
from pathlib import Path

__all__ = ["SCENE", "load_scene_config"]


def load_scene_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "scene.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SCENE = load_scene_config()


# ========== 形态学参数 ==========
MORPHOLOGY_ERODE_ITER = 2
MORPHOLOGY_DILATE_ITER = 2
MYBUFFER = 10  # 坐标中心队列长度
MIN_AREA = 1200  # 最小检测面积阈值
BAIS = 20  # ROI 扩展偏移量
