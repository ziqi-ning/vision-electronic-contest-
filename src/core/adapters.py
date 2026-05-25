"""
检测器适配器层
将旧检测器的返回值转换为 DetectionResult 统一类型，保持向后兼容。
"""

from typing import List, Tuple, Any, Dict
import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colorblob
import outsite
import other

from src.core.types import DetectionResult


# ========== 颜色检测适配器 ==========

def detect_color_adapted(frame: np.ndarray, color_name: str, bais: int = 20,
                          min_area: int = 1200) -> List[DetectionResult]:
    """
    适配 colorblob.detect_color
    返回 List[DetectionResult]
    """
    raw_results = colorblob.detect_color(frame, color_name, bais=bais, min_area=min_area)
    return [
        DetectionResult(
            type="color",
            confidence=min(r["pixels_max"] / 5000.0, 1.0),
            center=r["center"],
            extra={"color": color_name, "pixels_max": r["pixels_max"], "result": r["result"]}
        )
        for r in raw_results
    ]


def detect_color_to_rect_adapted(frame: np.ndarray, color_name: str, bais: int = 20,
                                  min_area: int = 1200) -> List[DetectionResult]:
    """
    适配 colorblob.detect_color_to_rect
    返回 List[DetectionResult]
    """
    raw_results = colorblob.detect_color_to_rect(frame, color_name, bais=bais, min_area=min_area)
    return [
        DetectionResult(
            type="color_rect",
            confidence=min(r["pixels_max"] / 5000.0, 1.0),
            center=r["center"],
            extra={"color": color_name, "pixels_max": r["pixels_max"], "result": r["result"]}
        )
        for r in raw_results
    ]


def detect_multi_color_adapted(frame: np.ndarray, color_name1: str, color_name2: str,
                               bais: int = 20, min_area: int = 1200) -> List[DetectionResult]:
    """
    适配 colorblob.detect_multi_color
    返回 List[DetectionResult]
    """
    raw_results = colorblob.detect_multi_color(frame, color_name1, color_name2,
                                                bais=bais, min_area=min_area)
    return [
        DetectionResult(
            type="multi_color",
            confidence=min(r["pixels_max"] / 5000.0, 1.0),
            center=r["center"],
            extra={"colors": [color_name1, color_name2], "pixels_max": r["pixels_max"],
                   "result": r["result"]}
        )
        for r in raw_results
    ]


# ========== 形状检测适配器 ==========

def detect_ellipses_adapted(image: np.ndarray) -> List[DetectionResult]:
    """
    适配 outsite.detect_ellipses
    返回 List[DetectionResult]
    """
    flag, result_img, ellipse_info, center, r_max = outsite.detect_ellipses(image)
    results = []
    for info in ellipse_info:
        results.append(DetectionResult(
            type="ellipse",
            confidence=info["area"] / (info["area"] + 1),
            center=(int(info["center"][0]), int(info["center"][1])),
            extra={
                "axes": info["axes"],
                "angle": info["angle"],
                "r": info["r"],
                "area": info["area"],
            }
        ))
    return results


def detect_ellipse_max_one_adapted(image: np.ndarray) -> Tuple[np.ndarray, Any, List[DetectionResult]]:
    """
    适配 outsite.detect_ellipse_max_one
    返回 (result_img, contour_max, List[DetectionResult])
    """
    result_img, contour_max = outsite.detect_ellipse_max_one(image)
    results = []
    if contour_max is not None:
        ellipse = outsite.detect_ellipses(result_img.copy())
        if ellipse[0] == 1 and ellipse[3] != (0, 0):
            results.append(DetectionResult(
                type="ellipse_max",
                confidence=0.8,
                center=ellipse[3],
                extra={"contour": contour_max}
            ))
    return result_img, contour_max, results


def detect_trapezoids_adapted(img: np.ndarray) -> List[DetectionResult]:
    """
    适配 outsite.detect_trapezoids
    返回 List[DetectionResult]
    """
    flag, img, trapezoid_info, center_max, width_max = outsite.detect_trapezoids(img)
    results = []
    for info in trapezoid_info:
        results.append(DetectionResult(
            type="trapezoid",
            confidence=0.8,
            center=info["center"],
            extra={"width": info["width"], "contour": info["contour"]}
        ))
    return results


def detect_triangle_adapted(img: np.ndarray) -> List[DetectionResult]:
    """
    适配 outsite.detect_triangle
    返回 List[DetectionResult]
    """
    flag, img, triangles_info, center_max, radius_max = outsite.detect_triangle(img)
    results = []
    for info in triangles_info:
        center = (int(info["center"][0]), int(info["center"][1]))
        results.append(DetectionResult(
            type="triangle",
            confidence=0.8,
            center=center,
            extra={"radius": info["radius"], "contour": info["contour"]}
        ))
    return results


def find_longest_straight_line_adapted(image: np.ndarray) -> List[DetectionResult]:
    """
    适配 outsite.find_longest_straight_line
    返回 List[DetectionResult]
    """
    flag, image, pole_groups, center = outsite.find_longest_straight_line(image)
    results = []
    for pole in pole_groups:
        cx = pole["center"][0] if pole["center"] else 0
        results.append(DetectionResult(
            type="pole",
            confidence=0.8,
            center=(cx, 0),
            extra={
                "angle": pole["angle"],
                "lines": pole["lines"],
                "avg_distance": pole["avg_distance"]
            }
        ))
    return results


# ========== 特殊标记适配器 ==========

def QR_detect_adapted(detector: cv2.QRCodeDetector, img: np.ndarray) -> List[DetectionResult]:
    """
    适配 other.QR_detect
    返回 List[DetectionResult]

    注意：other.QR_detect 返回 (img, flag, data, x, y, pixel)
    即 flag==1 时返回 5 值元组，flag==0 时返回 4 值元组
    适配后统一为 5 值，并在 flag==0 时返回空列表
    """
    result = other.QR_detect(detector, img)
    if len(result) < 5:
        return []

    img, flag, data, x, y, pixel = result
    if flag == 0:
        return []

    return [
        DetectionResult(
            type="qr",
            confidence=1.0,
            center=(int(x), int(y)),
            extra={
                "qr_data": data,
                "pixel": pixel,
                "result_img": img
            }
        )
    ]


def decode_barcode_adapted(_img: np.ndarray) -> List[DetectionResult]:
    """
    适配 other.decodeDisplay（条码检测）
    返回 List[DetectionResult]
    """
    result = other.decodeDisplay(_img)
    if len(result) < 5:
        return []
    img, x, y, apriltag_id, flag = result
    if flag == 0:
        return []

    return [
        DetectionResult(
            type="barcode",
            confidence=1.0,
            center=(int(x), int(y)),
            extra={
                "barcode_id": apriltag_id,
                "result_img": img
            }
        )
    ]


# ========== 级联形状分类器适配器 ==========

def find_type_adapted(imgsrc: np.ndarray) -> List[DetectionResult]:
    """
    适配 allin.find_type，将所有形状检测器级联
    返回 List[DetectionResult]
    """
    img, type_info = outsite.find_type(imgsrc)
    results = []
    for item in type_info:
        shape_type_map = {0: "ellipse", 1: "trapezoid", 2: "triangle", 3: "pole"}
        det_type = shape_type_map.get(item["type"], "unknown")
        results.append(DetectionResult(
            type=det_type,
            confidence=0.8,
            center=item["center"],
            extra={"lengh": item["lengh"]}
        ))
    return results


# ========== 完整 Pipeline 适配器 ==========

def give_me_a_color_and_i_will_give_you_a_shape_adapted(
        frame: np.ndarray, color: str, bais: int = 20) -> List[DetectionResult]:
    """
    适配 allin.give_me_a_color_and_i_will_give_you_a_shape
    返回 List[DetectionResult]，等价于原函数的 composite_img + type_list
    """
    raw_results = colorblob.detect_color(frame, color, bais)
    results = []

    for item in raw_results:
        result_img = item["result"]
        _, type_info = outsite.find_type(result_img)

        for t in type_info:
            shape_type_map = {0: "ellipse", 1: "trapezoid", 2: "triangle", 3: "pole"}
            det_type = shape_type_map.get(t["type"], "unknown")
            results.append(DetectionResult(
                type=det_type,
                confidence=0.8,
                center=t["center"],
                extra={
                    "color": color,
                    "color_center": item["center"],
                    "lengh": t["lengh"]
                }
            ))
    return results
