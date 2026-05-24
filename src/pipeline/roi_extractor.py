"""
ROI 提取器 — 策略模式
将 allin.py 中的 5 个 color_bonous_* 变体抽象为可替换的提取策略。
"""

from abc import ABC, abstractmethod
from typing import List
import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colorblob
import outsite


class ROIExtractor(ABC):
    """ROI 提取策略抽象基类"""

    @abstractmethod
    def extract(self, frame: np.ndarray, color: str, bais: int,
                min_area: int) -> List[dict]:
        """
        从帧中提取指定颜色的 ROI 区域列表。

        Returns:
            List[dict]，每个 dict 包含：
                result: ROI 裁剪图像
                center: (x, y) 中心点
                pixels_max: 面积（像素数）
        """
        pass


class CircleROIExtractor(ROIExtractor):
    """
    圆形 ROI 提取器 — 对应 allin.color_bonous_usual / give_me_a_color_and_i_will_give_you_a_shape
    使用圆形包围区域裁剪颜色区域。
    """

    def extract(self, frame: np.ndarray, color: str, bais: int = 20,
                min_area: int = 1200) -> List[dict]:
        cnt_efc = colorblob.detect_color(frame, color, bais=bais, min_area=min_area)
        return cnt_efc


class RectROIExtractor(ROIExtractor):
    """
    矩形 ROI 提取器 — 对应 allin.color_bonous_usual (detect_color_to_rect 版本)
    使用矩形（最小包围盒）裁剪颜色区域。
    """

    def extract(self, frame: np.ndarray, color: str, bais: int = 20,
                min_area: int = 1200) -> List[dict]:
        cnt_efc = colorblob.detect_color_to_rect(frame, color, bais=bais, min_area=min_area)
        return cnt_efc


class ORBROIExtractor(ROIExtractor):
    """
    ORB 专用 ROI 提取器 — 对应 allin.color_bonous_ORB
    圆形裁剪 + 椭圆边缘模糊，防止角点误检测。
    """

    def __init__(self, dilate_radius: int = 5, blur_kernel: tuple = (25, 25)):
        self.dilate_radius = dilate_radius
        self.blur_kernel = blur_kernel

    def extract(self, frame: np.ndarray, color: str, bais: int = 0,
                min_area: int = 1200) -> List[dict]:
        cnt_efc = colorblob.detect_color(frame, color, bais=bais, min_area=min_area)

        processed = []
        for item in cnt_efc:
            result_img = item["result"].copy()

            test_img, contour = outsite.detect_ellipse_max_one(result_img)
            if contour is not None:
                result_img = _blur_contour_only(
                    result_img, contour,
                    dilate_radius=self.dilate_radius,
                    blur_kernel=self.blur_kernel
                )

            processed.append({
                "result": result_img,
                "center": item["center"],
                "pixels_max": item["pixels_max"],
            })
        return processed


class LineROIExtractor(ROIExtractor):
    """
    直线检测专用 ROI 提取器 — 对应 allin.color_bonous_for_line
    矩形裁剪，并在 ROI 内检测直线。
    """

    def extract(self, frame: np.ndarray, color: str, bais: int = 20,
                min_area: int = 1200) -> List[dict]:
        cnt_efc = colorblob.detect_color_to_rect(frame, color, bais=bais, min_area=min_area)

        processed = []
        for item in cnt_efc:
            result_img = item["result"].copy()
            flag, result_img, pole_groups, center = outsite.find_longest_straight_line(result_img)

            processed.append({
                "result": result_img,
                "center": item["center"],
                "pixels_max": item["pixels_max"],
                "pole_groups": pole_groups,
            })
        return processed


class MultiColorROIExtractor(ROIExtractor):
    """
    多颜色 ROI 提取器 — 对应 allin.color_bonous_multi_color
    同时检测两种颜色的交叉区域。
    """

    def extract(self, frame: np.ndarray, color: str, bais: int = 20,
                min_area: int = 1200) -> List[dict]:
        color1, color2 = _parse_two_colors(color)
        cnt_efc = colorblob.detect_multi_color(
            frame, color1, color2, bais=bais, min_area=min_area
        )
        return cnt_efc


class LaserROIExtractor(ROIExtractor):
    """
    激光检测专用 ROI 提取器 — 对应 allin.color_bonous_laser_small_area
    使用更小的面积阈值检测激光点。
    """

    def __init__(self, min_area: int = 500):
        self.min_area = min_area

    def extract(self, frame: np.ndarray, color: str, bais: int = 5,
                min_area: int = None) -> List[dict]:
        effective_min_area = min_area if min_area is not None else self.min_area
        cnt_efc = colorblob.detect_color(
            frame, color, bais=bais, min_area=effective_min_area
        )
        return cnt_efc


# ========== 内部工具函数 ==========

def _blur_contour_only(src_img: np.ndarray, contour, dilate_radius: int = 5,
                       blur_kernel: tuple = (25, 25)) -> np.ndarray:
    """仅在轮廓线周围生成带状模糊区域 — 来自 allin.blur_contour_only"""
    mask = np.zeros_like(src_img[:, :, 0])
    cv2.drawContours(mask, [contour], -1, 255, thickness=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1,) * 2)
    expanded_mask = cv2.dilate(mask, kernel)

    blurred = cv2.GaussianBlur(src_img, blur_kernel, 0)

    condition = expanded_mask[:, :, None].astype(bool)
    result = np.where(condition, blurred, src_img)

    return result.astype(np.uint8)


def _parse_two_colors(color: str) -> tuple:
    """解析双颜色字符串，格式：'color1+color2'"""
    if "+" in color:
        parts = color.split("+")
        return parts[0].strip(), parts[1].strip()
    return color, color


# ========== ROI 合成工具（对应 allin 中的 composite 逻辑）==========

def composite_ROIs(frame: np.ndarray, roi_list: List[dict]) -> np.ndarray:
    """
    将多个 ROI 区域合成到一张黑色幕布上，防止重叠区域被重复填充。
    对应 allin.py 中各 color_bonous_* 函数末尾的合成逻辑。
    """
    composite_img = np.zeros(frame.shape, dtype=np.uint8)
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for item in roi_list:
        result_img = item["result"]

        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))

        composite_img[fill_mask == 255] = result_img[fill_mask == 255]

        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

    return composite_img
