# uav-multisensor-fusion

嵌入式平台（树莓派 / Jetson）多维感知处理库，面向电子设计竞赛场景，集成颜色识别、形状检测、特殊标记识别、激光雷达融合与串口通信功能。

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 功能特性

| 模块 | 说明 |
|------|------|
| **颜色检测** | HSV 颜色空间多颜色识别（红/绿/蓝/黄等），原创掩模过滤减轻后级算力负担 |
| **形状识别** | 椭圆、三角形、梯形、杆状线检测，支持单独调用与多区域同时识别 |
| **特殊标记** | AprilTag 位姿估计、QR 码解码、条码识别 |
| **雷达融合** | 原创点群算法 + 孤点算法实现像素坐标 → 雷达距离融合测距 |
| **串口通信** | UART 协议实现（波特率 256000），支持模式切换与结构化数据回传 |
| **外设控制** | RPi GPIO 控制（LED 调试指示灯） |

---

## 项目结构

```
FlightVersionOnRaspirryPi/
├── src/
│   ├── allin.py              # 颜色+形状综合检测（兼容旧接口）
│   ├── colorblob.py          # HSV 颜色检测模块
│   ├── outsite.py            # 形状识别模块
│   ├── other.py              # 特殊标记识别（AprilTag / QR / Barcode）
│   ├── radar5.py             # 雷达数据处理（兼容旧接口）
│   ├── uartuse.py            # 串口通信（兼容旧接口）
│   ├── facility2.py          # 外设控制（GPIO / LED）
│   │
│   ├── config/               # 配置层
│   │   ├── hardware.py       # 相机内参 / 外参标定参数
│   │   ├── protocol.py       # UART 通信协议字段定义
│   │   ├── modes.py          # 模式定义
│   │   └── scene.py          # 场景参数（从 YAML 加载）
│   │
│   ├── core/                 # 核心抽象
│   │   ├── types.py          # 统一数据类型（DetectionResult / BoundingBox 等）
│   │   └── adapters.py       # 新旧接口适配器层
│   │
│   ├── pipeline/             # 检测编排层
│   │   ├── orchestrator.py   # DetectionPipeline 编排器
│   │   ├── roi_extractor.py  # ROI 提取（Circular ROI + 掩模过滤）
│   │   └── shape_classifier.py # 形状分类器
│   │
│   ├── modes/                # 模式处理器
│   │   ├── base.py           # ModeHandler 抽象基类 + TargetData
│   │   ├── idle_mode.py      # 空闲模式
│   │   ├── qr_mode.py        # 二维码模式
│   │   └── stub_modes.py     # 各工作模式存根实现
│   │
│   ├── radar/                # 雷达数据层（ROS 解耦）
│   │   ├── base.py           # RadarSource 抽象基类
│   │   ├── ros_source.py     # ROS 数据源
│   │   ├── sim_source.py      # 模拟数据源（无 ROS 时自动回退）
│   │   └── fusion.py          # 相机-雷达融合器（RadarFusion）
│   │
│   ├── comm/                 # 通信层
│   │   └── serial_client.py  # 串口客户端（AsyncIO 化）
│   │
│   └── utils/
│       └── logger.py         # 统一日志模块
│
├── tests/                    # 测试套件
│   ├── unit/                # 单元测试
│   └── integration/         # 集成测试
│
├── util/                    # 调参工具
│   ├── 调参工具：颜色调参手动器.py
│   └── 调参工具：canny边缘调参手动器.py
│
├── docs/                    # 项目文档
├── reports/                 # 工作报告
├── main.py                  # 主程序入口（asyncio 协程模式）
├── pyproject.toml           # 项目配置（PEP 621）
├── pytest.ini               # pytest 配置
├── generate_test_video.py   # 合成测试视频生成脚本
└── 若干须知.md              # 使用注意事项
```

---

## 快速开始

### 环境要求

- Python >= 3.8
- OpenCV >= 4.8.0
- 支持 USB / CSI 接口摄像头

### 安装依赖

```bash
# 基础依赖
pip install -e .

# 可选功能
pip install -e ".[ros]"       # ROS1 雷达数据源支持
pip install -e ".[apriltag]"  # AprilTag 识别
pip install -e ".[barcode]"   # 条码识别
pip install -e ".[all]"       # 安装全部可选依赖
```

### 开发依赖

```bash
pip install -e ".[dev]"
```

包含：pytest、mypy、black、ruff。

### 运行测试

```bash
# 所有测试
pytest

# 仅单元测试
pytest tests/unit/

# 仅集成测试
pytest tests/integration/

# 带覆盖率报告
pytest --cov=src tests/
```

### 代码质量检查

```bash
ruff check src/    # ruff 检查
black src/         # black 格式化
mypy src/          # mypy 类型检查
```

---

## 核心 API 概览

### 颜色 + 形状综合检测

```python
from src.allin import give_me_a_color_and_i_will_give_you_a_shape

results, composite_img = give_me_a_color_and_i_will_give_you_a_shape(frame, "red", bais=20)
```

### 检测编排器（推荐）

```python
from src.pipeline.orchestrator import DetectionPipeline

pipeline = DetectionPipeline()
results, composite_img = pipeline.run(frame, "red", bais=20)
```

### 雷达融合

```python
from src.radar.fusion import RadarFusion

fusion = RadarFusion()  # 自动检测 ROS / 模拟数据源
dist_cm, angle_centideg = await fusion.angle_to_distance(80, 100)
obstacles = await fusion.get_obstacle()
```

### 串口通信

```python
from src.comm.serial_client import SerialClient

serial = SerialClient(port="/dev/ttyUSB0", baudrate=256000)
serial.connect()
serial.send(work_mode, target_data)
```

---

## 通信协议

### 控制指令

```
[0xFF, 0xFE, 0xA0, 0x01, mode, checksum]
```

### 数据返回

```
[0xFF, 0xFC, 0xA0+mode, length, data..., checksum]
```

完整协议字段定义见 `src/config/protocol.py`。

---

## 工程化说明

本项目已完成完整的工程化重构（Phase 1~4），包含：

- **PEP 621 标准化**：符合 Python 打包规范的 `pyproject.toml`
- **模块化架构**：配置层 / 核心抽象 / 检测编排 / 模式处理 / 雷达数据 / 通信层分层解耦
- **ROS 解耦**：雷达数据源通过 `RadarSource` 抽象接口注入，无 ROS 环境下自动回退到模拟数据源
- **类型提示**：全量类型标注（mypy 验证通过）
- **单元测试**：pytest 测试套件覆盖核心模块
- **代码质量**：ruff + black + mypy 完整工具链

---

## 调参工具

| 工具 | 用途 |
|------|------|
| `util/调参工具：颜色调参手动器.py` | 调整 HSV 颜色阈值，实时预览掩模效果 |
| `util/调参工具：canny边缘调参手动器.py` | 调整 Canny 边缘检测参数，优化形状识别效果 |

运行时将检测参数导出后保存至 `scene.yaml`，由 `src/config/scene.py` 加载生效。

---

## 注意事项

1. **硬件兼容性**：确保摄像头和雷达驱动正确安装
2. **光照条件**：颜色检测对光照敏感，建议在稳定光照环境下使用
3. **雷达标定**：相机-雷达融合功能需要精确的传感器外参标定（见 `src/config/hardware.py`）
4. **性能调优**：嵌入式平台上运行时，可适当调整图像分辨率和处理帧率

---

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。
