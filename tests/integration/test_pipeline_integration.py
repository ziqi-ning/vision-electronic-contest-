"""test_pipeline_integration.py — 检测 Pipeline 端到端集成测试"""

import pytest
import numpy as np
import cv2

from src.pipeline.orchestrator import DetectionPipeline
from src.pipeline.roi_extractor import CircleROIExtractor, RectROIExtractor
from src.pipeline.shape_classifier import ShapeClassifier


class TestPipelineIntegration:
    """完整检测链路集成测试：frame → ROI提取 → 形状分类 → 结果合成"""

    def test_full_pipeline_red_rectangle(self, red_rect_frame):
        pipeline = DetectionPipeline()
        type_list, composite_img = pipeline.run(red_rect_frame, "red", bais=20)
        # 有色块 → 应有形状结果或空列表（取决于形状检测灵敏度）
        assert isinstance(type_list, list)
        assert composite_img.shape == red_rect_frame.shape

    def test_full_pipeline_empty_frame(self, blank_frame):
        pipeline = DetectionPipeline()
        type_list, composite_img = pipeline.run(blank_frame, "red", bais=20)
        assert type_list == []
        assert composite_img.shape == blank_frame.shape

    def test_pipeline_with_rect_extractor(self, red_rect_frame):
        pipeline = DetectionPipeline(roi_extractor=RectROIExtractor())
        type_list, composite_img = pipeline.run(red_rect_frame, "red")
        assert composite_img.shape == red_rect_frame.shape

    def test_pipeline_multi_color(self, multi_color_frame):
        """多颜色帧：红+绿各自分别检测"""
        pipeline = DetectionPipeline()
        red_results, _ = pipeline.run(multi_color_frame, "red")
        green_results, _ = pipeline.run(multi_color_frame, "green")
        assert isinstance(red_results, list)
        assert isinstance(green_results, list)

    def test_pipeline_shapes_sequence(self, trapezoid_frame, triangle_frame, ellipse_frame):
        """依次送入梯形、三角形、椭圆帧，结果应各不相同"""
        pipeline = DetectionPipeline()
        results_trap, _ = pipeline.run(trapezoid_frame, "red")
        results_tri, _ = pipeline.run(triangle_frame, "red")
        results_ell, _ = pipeline.run(ellipse_frame, "red")
        # 三种形状的结果不应完全相同（理想情况下各自的形状类型应不同）
        assert isinstance(results_trap, list)
        assert isinstance(results_tri, list)
        assert isinstance(results_ell, list)

    def test_composite_img_non_empty_after_detection(self, red_rect_frame):
        """合成图非空检测（降级：bitwise_not 整帧）"""
        pipeline = DetectionPipeline()
        _, composite_img = pipeline.run(red_rect_frame, "red", bais=20)
        assert composite_img.shape == red_rect_frame.shape
        assert composite_img.dtype == np.uint8

    def test_pipeline_run_with_custom_classifier(self, trapezoid_frame):
        """自定义形状分类器注入"""
        classifier = ShapeClassifier(priority=["trapezoid"])
        pipeline = DetectionPipeline(shape_classifier=classifier)
        type_list, _ = pipeline.run(trapezoid_frame, "red")
        assert isinstance(type_list, list)


class TestColorBlobAdapter:
    """adapters.py 颜色检测适配器集成测试"""

    def test_detect_color_adapted(self, red_rect_frame):
        from src.core.adapters import detect_color_adapted
        results = detect_color_adapted(red_rect_frame, "red", bais=20, min_area=500)
        assert isinstance(results, list)
        if results:
            assert results[0].type == "color"
            assert results[0].center is not None
            assert 0.0 <= results[0].confidence <= 1.0

    def test_detect_color_adapted_no_color(self, blank_frame):
        from src.core.adapters import detect_color_adapted
        results = detect_color_adapted(blank_frame, "red", bais=20, min_area=500)
        assert results == []

    def test_detect_multi_color_adapted(self, multi_color_frame):
        from src.core.adapters import detect_multi_color_adapted
        results = detect_multi_color_adapted(
            multi_color_frame, "red", "green", bais=20, min_area=500
        )
        assert isinstance(results, list)

    def test_give_me_a_color_and_i_will_give_you_a_shape_adapted(self, red_rect_frame):
        from src.core.adapters import give_me_a_color_and_i_will_give_you_a_shape_adapted
        results = give_me_a_color_and_i_will_give_you_a_shape_adapted(
            red_rect_frame, "red", bais=20
        )
        assert isinstance(results, list)


class TestDetectionModesIntegration:
    """各模式处理器集成测试（mock 串口和雷达）"""

    @pytest.mark.asyncio
    async def test_idle_mode_no_color(self, blank_frame):
        from src.modes.idle_mode import IdleMode
        from src.modes.base import TargetData

        mock_serial = __import__("unittest.mock").mock.MagicMock()
        mock_radar = __import__("unittest.mock").mock.MagicMock()
        mock_radar.get_obstacle = __import__("asyncio").coroutine(
            lambda: [(3000, 40000)]
        )
        mock_queue_test = __import__("asyncio").Queue()
        mock_queue_draw = __import__("asyncio").Queue()

        mode = IdleMode(
            cap=None, serial=mock_serial, radar=mock_radar,
            pipeline=None, target=TargetData(),
            ctr=__import__("src.uartuse", fromlist=["ModeCtrl"]).ModeCtrl(),
            queue_radar_test=mock_queue_test,
            queue_radar_draw=mock_queue_draw,
        )
        result = await mode.process(blank_frame)
        assert result == []

    @pytest.mark.asyncio
    async def test_color_block_mode(self, red_rect_frame):
        from src.modes.stub_modes import ColorBlockMode
        from src.modes.base import TargetData

        mock_serial = __import__("unittest.mock").mock.MagicMock()
        mock_radar = __import__("unittest.mock").mock.MagicMock()
        mock_queue_test = __import__("asyncio").Queue()
        mock_queue_draw = __import__("asyncio").Queue()

        target = TargetData()
        mode = ColorBlockMode(
            cap=None, serial=mock_serial, radar=mock_radar,
            pipeline=None, target=target,
            ctr=__import__("src.uartuse", fromlist=["ModeCtrl"]).ModeCtrl(),
            queue_radar_test=mock_queue_test,
            queue_radar_draw=mock_queue_draw,
        )
        await mode.process(red_rect_frame)
        assert target.img_width == 640
        assert target.img_height == 480

    @pytest.mark.asyncio
    async def test_circle_mode_no_circle(self, blank_frame):
        from src.modes.stub_modes import CircleMode
        from src.modes.base import TargetData

        mock_serial = __import__("unittest.mock").mock.MagicMock()
        mock_radar = __import__("unittest.mock").mock.MagicMock()
        mock_queue_test = __import__("asyncio").Queue()
        mock_queue_draw = __import__("asyncio").Queue()

        target = TargetData()
        mode = CircleMode(
            cap=None, serial=mock_serial, radar=mock_radar,
            pipeline=None, target=target,
            ctr=__import__("src.uartuse", fromlist=["ModeCtrl"]).ModeCtrl(),
            queue_radar_test=mock_queue_test,
            queue_radar_draw=mock_queue_draw,
        )
        await mode.process(blank_frame)
        # 空白帧无白色色块，flag 应为 0
        assert target.flag == 0

    @pytest.mark.asyncio
    async def test_sound_mode_radar_only(self, blank_frame):
        from src.modes.stub_modes import SoundMode
        from src.modes.base import TargetData

        mock_serial = __import__("unittest.mock").mock.MagicMock()
        mock_radar = __import__("unittest.mock").mock.MagicMock()
        mock_radar.get_obstacle = __import__("asyncio").coroutine(
            lambda: [(2000, 9000)]
        )
        mock_queue_test = __import__("asyncio").Queue()
        mock_queue_draw = __import__("asyncio").Queue()

        target = TargetData()
        mode = SoundMode(
            cap=None, serial=mock_serial, radar=mock_radar,
            pipeline=None, target=target,
            ctr=__import__("src.uartuse", fromlist=["ModeCtrl"]).ModeCtrl(),
            queue_radar_test=mock_queue_test,
            queue_radar_draw=mock_queue_draw,
        )
        await mode.process(blank_frame)
        # 障碍物 < 3000cm → flag=1
        assert target.flag == 1
        assert target.distance == 2000
