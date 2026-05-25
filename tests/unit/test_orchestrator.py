"""test_orchestrator.py — 检测编排器单元测试"""

import pytest
import numpy as np

from src.pipeline.orchestrator import DetectionPipeline
from src.pipeline.roi_extractor import CircleROIExtractor


class TestDetectionPipeline:
    def test_default_extractors_created(self):
        """默认构造不传参数时自动创建提取器和分类器"""
        pipeline = DetectionPipeline()
        assert pipeline.roi_extractor is not None
        assert pipeline.shape_classifier is not None

    def test_custom_extractor_injection(self):
        """可注入自定义 ROI 提取器"""
        extractor = CircleROIExtractor()
        pipeline = DetectionPipeline(roi_extractor=extractor)
        assert pipeline.roi_extractor is extractor

    def test_run_red_rectangle(self, red_rect_frame):
        """红色矩形帧 → 应检测到某种形状"""
        pipeline = DetectionPipeline()
        type_list, composite_img = pipeline.run(red_rect_frame, "red", bais=20)
        assert isinstance(type_list, list)
        assert composite_img.shape == red_rect_frame.shape
        assert composite_img.dtype == np.uint8

    def test_run_no_color(self, blank_frame):
        """空白帧无颜色 → 返回空列表"""
        pipeline = DetectionPipeline()
        type_list, composite_img = pipeline.run(blank_frame, "red", bais=20)
        assert type_list == []
        assert composite_img.shape == blank_frame.shape

    def test_run_returns_composite_img_type(self, red_rect_frame):
        pipeline = DetectionPipeline()
        _, composite_img = pipeline.run(red_rect_frame, "red")
        assert isinstance(composite_img, np.ndarray)

    def test_result_type_to_code(self):
        """类型码映射正确"""
        assert DetectionPipeline._result_type_to_code("ellipse") == 0
        assert DetectionPipeline._result_type_to_code("trapezoid") == 1
        assert DetectionPipeline._result_type_to_code("triangle") == 2
        assert DetectionPipeline._result_type_to_code("pole") == 3
        assert DetectionPipeline._result_type_to_code("unknown") == -1

    def test_bitwise_not(self):
        """位运算工具正确"""
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:8, 2:8] = 255
        result = DetectionPipeline._bitwise_not(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 255
        assert result[5, 5] == 0

    def test_bitwise_and(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        a[2:8, 2:8] = 255
        b = np.zeros((10, 10), dtype=np.uint8)
        b[4:6, 4:6] = 255
        result = DetectionPipeline._bitwise_and(a, b)
        assert result[5, 5] == 255
        assert result[1, 1] == 0

    def test_blend_roi(self):
        composite = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        result = DetectionPipeline._blend_roi(composite, roi, mask)
        assert result[50, 50, 0] == 200
        assert result[0, 0, 0] == 0
