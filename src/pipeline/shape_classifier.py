"""
形状分类器 — 级联 Pipeline Stage
将 allin.find_type() 的级联分类逻辑抽象为可配置的 ShapeClassifier。
优先级顺序：trapezoid → triangle → pole → ellipse
"""

from typing import List, Dict, Any, Callable, Optional
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import outsite
from src.core.types import DetectionResult


class ShapeClassifier:
    """
    级联形状分类器。

    按优先级顺序（trapezoid → triangle → pole → ellipse）级联调用各形状检测器，
    一旦某个检测器返回结果则停止后续检测。
    """

    def __init__(self, priority: Optional[List[str]] = None):
        self.priority = priority or ["trapezoid", "triangle", "pole", "ellipse"]
        self._registry: Dict[str, Callable[[np.ndarray], tuple]] = {
            "trapezoid": self._detect_trapezoid,
            "triangle": self._detect_triangle,
            "pole": self._detect_pole,
            "ellipse": self._detect_ellipse,
        }

    def classify(self, roi: np.ndarray) -> List[DetectionResult]:
        """
        对单个 ROI 图像进行级联形状分类。

        Returns:
            List[DetectionResult]，若未检测到任何形状则返回空列表。
        """
        results: List[DetectionResult] = []

        for shape_name in self.priority:
            detector = self._registry.get(shape_name)
            if detector is None:
                continue

            flag, img, type_info = detector(roi)
            if flag == 1 and type_info:
                shape_type_map = {
                    "trapezoid": 1,
                    "triangle": 2,
                    "pole": 3,
                    "ellipse": 0,
                }
                for item in type_info:
                    results.append(DetectionResult(
                        type=shape_name,
                        confidence=0.8,
                        center=item["center"],
                        extra={"lengh": item.get("lengh", 0)}
                    ))
                break

        return results

    def _detect_trapezoid(self, img: np.ndarray) -> tuple:
        flag, img, trapezoid_info, center_max, width_max = outsite.detect_trapezoids(img)
        return flag, img, [{"type": 1, "center": center_max, "lengh": width_max}]

    def _detect_triangle(self, img: np.ndarray) -> tuple:
        flag, img, triangles_info, center_max, radius_max = outsite.detect_triangle(img)
        return flag, img, [{"type": 2, "center": center_max, "lengh": radius_max}]

    def _detect_pole(self, img: np.ndarray) -> tuple:
        flag, img, pole_groups, center = outsite.find_longest_straight_line(img)
        if pole_groups:
            return flag, img, [{"type": 3, "center": center, "lengh": 0}]
        return flag, img, []

    def _detect_ellipse(self, img: np.ndarray) -> tuple:
        flag, img, ellipse_info, center, r_max = outsite.detect_ellipses(img)
        return flag, img, [{"type": 0, "center": center, "lengh": r_max}]
