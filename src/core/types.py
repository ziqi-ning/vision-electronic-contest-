"""
统一数据类型
所有检测器的返回值均使用 DetectionResult 等统一类型，消除返回值格式不一致问题。
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class BoundingBox:
    """轴对齐包围盒"""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @classmethod
    def from_rect(cls, rect: Tuple[int, int, int, int]) -> "BoundingBox":
        """从 (cx, cy, w, h) 格式创建"""
        cx, cy, w, h = rect
        return cls(x=cx - w // 2, y=cy - h // 2, width=w, height=h)


@dataclass
class DetectionResult:
    """
    所有检测器的统一返回类型

    Attributes:
        type: 检测类型，'color' | 'ellipse' | 'trapezoid' | 'triangle' | 'pole' |
              'qr' | 'april_tag' | 'barcode' | 'laser'
        confidence: 置信度 0.0~1.0
        bbox: 可选包围盒
        center: 可选中心点坐标 (x, y)
        extra: 扩展数据字典，用于携带检测器特有的附加信息
    """
    type: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    center: Optional[Tuple[int, int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RadarScan:
    """雷达扫描数据"""
    timestamp: float
    points: List[Tuple[float, float]]  # (distance_cm, angle_centidegree)
    obstacles: List[Tuple[float, float]] = field(default_factory=list)  # 障碍物


@dataclass
class FusionResult:
    """相机-雷达融合结果"""
    pixel: Tuple[int, int]
    distance: Optional[float] = None      # 米
    angle: Optional[float] = None          # 度
    obstacle_detected: bool = False


# ========== 类型别名 ==========
ColorResultList = List[Dict[str, Any]]  # 兼容 colorblob 原始返回值
ShapeResultList = List[Dict[str, Any]]   # 兼容 outsite 原始返回值
