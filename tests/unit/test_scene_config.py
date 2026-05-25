"""test_scene_config.py — 场景配置加载测试"""

import pytest
from pathlib import Path


class TestSceneConfig:
    def test_scene_yaml_loads(self):
        from src.config.scene import SCENE, load_scene_config
        config = load_scene_config()
        assert "colors" in config
        assert "detection" in config

    def test_scene_has_required_colors(self):
        from src.config.scene import SCENE
        required = ["red", "green", "blue"]
        for color in required:
            assert color in SCENE["colors"]
            assert "lower" in SCENE["colors"][color]
            assert "upper" in SCENE["colors"][color]

    def test_color_hsv_ranges_valid(self):
        from src.config.scene import SCENE
        for name, cfg in SCENE["colors"].items():
            lower = cfg["lower"]
            upper = cfg["upper"]
            assert len(lower) == 3
            assert len(upper) == 3
            assert all(0 <= v <= 255 for v in lower + upper)

    def test_detection_params(self):
        from src.config.scene import SCENE
        assert SCENE["detection"]["min_area"] > 0
        assert SCENE["detection"]["roi_bais"] >= 0

    def test_morphology_params(self):
        from src.config.scene import SCENE
        assert SCENE["morphology"]["erode_iter"] >= 0
        assert SCENE["detection"]["roi_bais"] >= 0


class TestHardwareConfig:
    def test_camera_params_tuple(self):
        from src.config.hardware import camera_params
        assert len(camera_params) == 10

    def test_image_dimensions(self):
        from src.config.hardware import CameraConfig
        assert CameraConfig.IMAGE_WIDTH == 640
        assert CameraConfig.IMAGE_HEIGHT == 480

    def test_radar_tolerance(self):
        from src.config.hardware import angle_tolerance_rad
        assert angle_tolerance_rad > 0

    def test_cam_info_populated(self):
        from src.config.hardware import cam_info
        assert cam_info.fx > 0
        assert cam_info.fy > 0
        assert cam_info.tag_size_m > 0
