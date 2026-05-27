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
│   ├── colorblob.py          # HSV 颜色检测模块（底层实现）
│   ├── outsite.py            # 形状识别模块（底层实现）
│   ├── other.py              # 特殊标记识别（AprilTag / QR / Barcode，底层实现）
│   ├── uartuse.py            # 串口通信协议解析（底层实现）
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

## API 使用指南

### 1. 颜色 + 形状综合检测

这是最常用的入口，一次调用完成"找色块 → 提取 ROI → 识别形状"全流程。

```python
import cv2
from src.allin import give_me_a_color_and_i_will_give_you_a_shape

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

# 在 frame 中找所有红色区域，并识别其中的形状
composite_img, type_list = give_me_a_color_and_i_will_give_you_a_shape(frame, "red", bais=30)
```

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `frame` | `np.ndarray` | BGR 格式输入图像（来自 `cv2.VideoCapture` 或文件） |
| `color` | `str` | 目标颜色，见下方支持列表 |
| `bais` | `int` | 圆形 ROI 向外扩展的像素数，默认 20，越大越宽松 |

**支持的颜色字符串**

`"red"` / `"green"` / `"blue"` / `"black"` / `"white"` / `"red_laser"` / `"all"` / `"tiger"` / `"wolf"` / `"elephant"` / `"monkey"` / `"peacock"`

**返回值**

- `composite_img`：合成后的可视化图像（BGR），可直接 `cv2.imshow` 显示
- `type_list`：检测到的形状列表，每项是一个字典：

```python
# type_list 示例
[
    {"type": 0, "center": (320, 240), "lengh": 45},  # 椭圆/圆
    {"type": 1, "center": (150, 200), "lengh": 60},  # 梯形
]
```

**形状类型码**

| `type` 值 | 形状 |
|-----------|------|
| `0` | 椭圆 / 圆 |
| `1` | 梯形 |
| `2` | 三角形 |
| `3` | 杆状线（两条平行竖线） |

> 推荐用法：用 `DetectionPipeline`（见下节）替代直接调用此函数，接口更规范。

---

### 2. 检测编排器（推荐方式）

`DetectionPipeline` 是对上述函数的面向对象封装，行为完全等价，但支持注入自定义的 ROI 提取器和形状分类器。

```python
import cv2
from src.pipeline.orchestrator import DetectionPipeline

pipeline = DetectionPipeline()  # 使用默认提取器和分类器

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

type_list, composite_img = pipeline.run(frame, "green", bais=20)

for item in type_list:
    shape_name = {0: "椭圆", 1: "梯形", 2: "三角形", 3: "杆子"}.get(item["type"], "未知")
    print(f"检测到 {shape_name}，中心 {item['center']}，尺寸 {item['lengh']}")

cv2.imshow("result", composite_img)
cv2.waitKey(0)
```

> 注意：`pipeline.run()` 的返回顺序是 `(type_list, composite_img)`，与 `allin` 函数相反。

---

### 3. 颜色掩模（不识别形状，只提取色块区域）

如果只需要把某种颜色的区域抠出来，不需要形状识别，用以下函数：

```python
from src.allin import color_bonous_usual          # 圆形 ROI 掩模
from src.allin import color_bonous_multi_color    # 同时匹配两种颜色

# 单色掩模
composite_img = color_bonous_usual(frame, "blue")

# 双色掩模（两种颜色合并为一个掩模）
composite_img = color_bonous_multi_color(frame, "red", "green")
```

返回值均为 BGR 图像，非目标颜色区域为纯黑。

---

### 4. 激光点检测

激光点面积极小，需要用专用函数先缩小 ROI 范围，再检测高亮点：

```python
from src.allin import color_bonous_laser_small_area
from src.colorblob import detect_laser

# 第一步：提取激光颜色区域（min_area=0 允许极小色块）
composite_img = color_bonous_laser_small_area(frame, "red_laser", bais=0, min_area=0)

# 第二步：在提取区域内定位激光亮点
flag, result_img, center, radius = detect_laser(composite_img, light_bais=25, min_area=0, max_area=5000)

if flag == 1:
    print(f"激光点坐标: {center}，半径: {radius}px")
```

**`detect_laser` 返回值**

| 返回值 | 说明 |
|--------|------|
| `flag` | `1` 检测到，`0` 未检测到 |
| `result_img` | 标注了激光点的图像 |
| `center` | 激光点中心坐标 `(x, y)` |
| `radius` | 激光点半径（像素） |

---

### 5. 特殊标记识别（QR / AprilTag / 条码）

```python
import cv2
import apriltag
from src.other import QR_detect, opencv_find_april_tag, decodeDisplay

# --- QR 码 ---
detector = cv2.QRCodeDetector()
result_img, flag, data, x, y, pixel = QR_detect(detector, frame)
if flag == 1:
    print(f"QR 内容: {data}，中心: ({x}, {y})，面积: {pixel}px²")

# --- AprilTag ---
options = apriltag.DetectorOptions(families="tag36h11")
april_detector = apriltag.Detector(options)
# cam_info 需包含 fx, fy, cx, cy, tag_size_m 属性
x, y, tag_id, side_len, flag, px_mm, py_mm, pz_mm = opencv_find_april_tag(frame, cam_info, april_detector)
if flag == 1:
    print(f"Tag ID: {tag_id}，位置: ({x}, {y})，距离: {pz_mm}mm")

# --- 条码 ---
result_img, x, y, barcode_id, flag = decodeDisplay(frame)
if flag == 1:
    print(f"条码内容: {barcode_id}，中心: ({x}, {y})")
```

**`QR_detect` 返回值说明**

| 返回值 | 说明 |
|--------|------|
| `flag` | `1` 检测到，`0` 未检测到 |
| `data` | 解码内容（整数） |
| `x`, `y` | 二维码中心坐标 |
| `pixel` | 二维码面积（像素²），可用于估算距离 |

---

### 6. 雷达融合测距

`RadarFusion` 自动检测 ROS 环境，无 ROS 时回退到模拟数据源，接口不变。所有方法均为 `async`，需在协程中调用。

```python
import asyncio
from src.radar.fusion import RadarFusion

fusion = RadarFusion()  # 自动选择 ROS / 模拟数据源

async def main():
    # 方式一：按角度范围查询最近障碍物
    # 参数为搜索起始角度和结束角度（单位：度）
    dist_cm, angle_centideg = await fusion.angle_to_distance(80, 100)
    print(f"距离: {dist_cm}cm，角度: {angle_centideg / 100:.1f}°")

    # 方式二：按像素坐标查询对应距离（需要相机标定参数）
    camera_params = (fx, fy, cx, cy, delta_x, delta_y, delta_z,
                     camera_pitch_deg, angle_tolerance_rad, camera_height)
    dist_cm, angle_centideg = await fusion.site_to_distance(320, 240, camera_params)

    # 方式三：获取所有障碍物列表
    obstacles = await fusion.get_obstacle()
    # obstacles 格式：[(距离_cm, 角度_centideg), ...]
    for dist, angle in obstacles:
        print(f"障碍物: {dist}cm @ {angle / 100:.1f}°")

asyncio.run(main())
```

**返回值说明**

所有测距方法在无数据时均返回 `(3000, 40000)`，即 30m / 400°，作为"无效值"标志。

| 返回值 | 说明 |
|--------|------|
| `dist_cm` | 距离，单位厘米 |
| `angle_centideg` | 角度，单位百分之一度（除以 100 得到度数） |

**`camera_params` 元组字段顺序**

```python
camera_params = (
    fx,                   # 水平焦距（像素）
    fy,                   # 垂直焦距（像素）
    cx,                   # 光心 x（通常为图像宽度/2）
    cy,                   # 光心 y（通常为图像高度/2）
    delta_x,              # 雷达相对相机 X 偏移（米，右为正）
    delta_y,              # 雷达相对相机 Y 偏移（米，前为负）
    delta_z,              # 雷达相对相机 Z 偏移（米，上为负）
    camera_pitch_deg,     # 相机俯仰角（度）
    angle_tolerance_rad,  # 角度匹配容差（弧度）
    camera_height,        # 相机离地高度（米）
)
```

---

### 7. 串口通信

```python
from src.comm.serial_client import SerialClient

# 默认端口 /dev/ttyAMA4，波特率 256000
serial = SerialClient(port="/dev/ttyAMA4", baudrate=256000)

if serial.open():
    print("串口已连接")

# 读取当前工作模式（需在主循环中持续调用）
serial.read_mode()
mode = serial.get_mode()  # 返回整数模式码

# 发送检测结果
success = serial.send(mode, target_data)  # target_data 为 TargetData 对象

serial.close()
```

也可以用协程方式持续监听：

```python
import asyncio

async def main():
    serial = SerialClient()
    serial.open()
    # serial_get() 会持续读取串口，直到串口关闭
    await serial.serial_get()

asyncio.run(main())
```

---

### 8. ORB 模板匹配（Logo 识别）

用于在摄像头画面中匹配预存的模板图像（如特定 Logo）：

```python
import cv2
from src.allin import template_matching

# 加载模板图像
template = cv2.imread("logo.png")

# 在当前帧中匹配（会自动提取目标颜色区域再匹配）
matched, result_img, score = template_matching(template, frame, "red")

if matched:
    print(f"匹配成功，得分: {score:.3f}")  # score 越小越准（平均匹配距离模式）
```

多模板遍历时，对每个模板调用一次，取 `score` 最小的作为最佳匹配。

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

## 示例演示

### 第一步：生成测试视频

```bash
python generate_test_video.py
```

运行后在 `example/videos/` 目录下生成 8 个独立场景视频：

| 视频文件 | 场景内容 |
|---------|---------|
| `test_color_single.avi` | 红/绿/蓝色块依次移动 |
| `test_multi_color.avi` | 红+绿色块同框各自移动 |
| `test_trapezoid.avi` | 红色梯形移动 |
| `test_triangle.avi` | 红色三角形移动 |
| `test_ellipse.avi` | 红色圆/椭圆移动 |
| `test_multi_shape.avi` | 梯形+三角形+圆同框 |
| `test_pole.avi` | 两根平行红色竖线移动 |
| `test_laser.avi` | 极亮红色激光点游走 |

### 第二步：运行对应 demo

```bash
# 颜色检测
python example/demo_color.py

# 多颜色同时检测
python example/demo_multi_color.py

# 形状识别（梯形/三角形/椭圆，改脚本内 VIDEO_NAME 切换）
python example/demo_shape.py

# 多区域形状同时识别
python example/demo_multi_shape.py

# 杆子（平行竖线）检测
python example/demo_pole.py

# 激光点检测
python example/demo_laser.py
```

每个 demo 弹出左右分屏窗口：**左侧原图，右侧检测结果**，按 `Q` 退出，视频自动循环。

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
