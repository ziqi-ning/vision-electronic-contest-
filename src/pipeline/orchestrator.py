"""
检测编排器
将 allin.give_me_a_color_and_i_will_give_you_a_shape() 改写为 Pipeline 编排器，
等价替换原函数的全部逻辑。
"""

from typing import List, Optional, TYPE_CHECKING
from collections import Counter
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import DetectionResult

if TYPE_CHECKING:
    from .roi_extractor import ROIExtractor
    from .shape_classifier import ShapeClassifier


class DetectionPipeline:
    """
    检测编排器：颜色检测 → ROI 提取 → 形状分类 → 结果合成

    等价于 allin.give_me_a_color_and_i_will_give_you_a_shape()，
    支持注入不同的 ROI 提取器和形状分类器。
    """

    def __init__(self,
                 roi_extractor: Optional["ROIExtractor"] = None,
                 shape_classifier: Optional["ShapeClassifier"] = None):
        from .roi_extractor import CircleROIExtractor
        from .shape_classifier import ShapeClassifier

        self.roi_extractor = roi_extractor or CircleROIExtractor()
        self.shape_classifier = shape_classifier or ShapeClassifier()

    def run(self, frame: np.ndarray, color: str, bais: int = 20) -> List[DetectionResult]:
        """
        执行完整检测 Pipeline。

        Args:
            frame: 输入图像（BGR格式）
            color: 目标颜色字符串，如 "red"、"green"
            bais: 圆形 ROI 扩展半径，默认 20

        Returns:
            List[DetectionResult]：所有检测到的形状结果
            composite_img: 合成图像（BGR格式）
        """
        from .roi_extractor import composite_ROIs

        cnt_efc = self.roi_extractor.extract(frame, color, bais=bais)
        if not cnt_efc:
            return [], np.zeros(frame.shape, dtype=np.uint8)

        composite_img = np.zeros(frame.shape, dtype=np.uint8)
        filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        type_list: List[dict] = []

        for item in cnt_efc:
            result_img = item["result"]

            shape_results = self.shape_classifier.classify(result_img)

            if shape_results:
                shape_info = [
                    {"type": self._result_type_to_code(r.type), "center": r.center, "lengh": r.extra.get("lengh", 0)}
                    for r in shape_results
                ]
                type_list.extend(shape_info)

                for r in shape_results:
                    r.extra["color"] = color
                    r.extra["color_center"] = item["center"]

            gray = np.array(result_img)
            if len(gray.shape) == 3:
                gray = gray[:, :, 0]

            _, current_mask = self._threshold_like_original(gray)

            fill_mask = self._bitwise_and(
                current_mask,
                self._bitwise_not(filled_mask)
            )

            composite_img = self._blend_roi(composite_img, result_img, fill_mask)
            filled_mask = self._bitwise_or(filled_mask, fill_mask)

        return type_list, composite_img

    def run_with_adapters(self, frame: np.ndarray, color: str,
                          bais: int = 20) -> List[DetectionResult]:
        """
        使用适配器层返回统一 DetectionResult 格式。
        供后续 Phase 2 T2.3 简化 main.py 使用。
        """
        from src.core.adapters import give_me_a_color_and_i_will_give_you_a_shape_adapted

        return give_me_a_color_and_i_will_give_you_a_shape_adapted(frame, color, bais)

    @staticmethod
    def _result_type_to_code(type_name: str) -> int:
        """将类型名转换为旧代码中的数字码"""
        mapping = {"ellipse": 0, "trapezoid": 1, "triangle": 2, "pole": 3}
        return mapping.get(type_name, -1)

    @staticmethod
    def _threshold_like_original(gray: np.ndarray) -> tuple:
        """模拟原 allin.py 中的 threshold 逻辑"""
        import cv2
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return _, current_mask

    @staticmethod
    def _bitwise_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.bitwise_and(a, b)

    @staticmethod
    def _bitwise_not(a: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.bitwise_not(a)

    @staticmethod
    def _bitwise_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.bitwise_or(a, b)

    @staticmethod
    def _blend_roi(composite: np.ndarray, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """将 ROI 区域按掩码混合到合成图上"""
        import cv2
        result = composite.copy()
        result[mask == 255] = roi[mask == 255]
        return result
