"""
src.pipeline — 检测 Pipeline 模块
Phase 2 T2.2：拆分 allin.py
"""

from .roi_extractor import (
    ROIExtractor,
    CircleROIExtractor,
    RectROIExtractor,
    ORBROIExtractor,
    LineROIExtractor,
    MultiColorROIExtractor,
    LaserROIExtractor,
    composite_ROIs,
)
from .shape_classifier import ShapeClassifier
from .orchestrator import DetectionPipeline

__all__ = [
    "ROIExtractor",
    "CircleROIExtractor",
    "RectROIExtractor",
    "ORBROIExtractor",
    "LineROIExtractor",
    "MultiColorROIExtractor",
    "LaserROIExtractor",
    "composite_ROIs",
    "ShapeClassifier",
    "DetectionPipeline",
]
