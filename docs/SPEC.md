# FVS-Cpp 项目需求规格文档

> **文档类型**：需求规格（SPEC）\
> **版本**：1.0\
> **日期**：2026-05-27\
> **状态**：草稿，待确认

---

## 一、项目背景与目标

### 1.1 项目来源

本项目是 `uav-multisensor-fusion`（Python版）的 C++ 重写版本。

- **Python版仓库**：`https://github.com/CQUT-302/FlightVersionOnRaspirryPi`
- **Python版状态**：Phase 1-4 全部完成，功能完整，主分支
- **C++版分支**：`cpp-restruct`（从 `main` 分出）
- **C++版工作空间**：`E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version3\FVS-Cpp`

### 1.2 重写目标

1. 将核心感知处理逻辑迁移到 C++，消除 Python 运行时在 ARM 嵌入式平台上的开销
2. 交叉编译为单二进制文件，简化树莓派/Jetson 部署
3. 保持与 Python 版完全一致的检测结果输出（像素级对齐）
4. **下位机固件不做任何修改**，UART 通信协议（36字节数据包）完全照搬

### 1.3 约束条件

| 约束 | 说明 |
|------|------|
| 协议不变 | UART 数据包格式（帧头0xFF 0xFC、功能码0xA0+mode、36字节字段）严格对齐 |
| 功能等价 | 检测结果（坐标、面积、形状类型）与 Python 版像素级一致 |
| 目标平台 | ARM（树莓派 3B+/4 / Jetson Nano/NX/Xavier） |
| 开发平台 | Windows x86（当前）+ Linux ARM（最终） |
| ROS 兼容 | 雷达数据支持 ROS `/scan` 话题，无 ROS 时自动回退到模拟数据源 |

---

## 二、硬件规格（来自 Python 版 `src/config/hardware.py`）

### 2.1 相机配置

| 参数 | 值 |
|------|------|
| 分辨率 | 640 × 480 |
| 帧率 | 50 FPS |
| 水平焦距 fx | `(3.6 / 3.6736) × 640 ≈ 628.7 px` |
| 垂直焦距 fy | `(3.6 / 2.7384) × 480 ≈ 631.0 px` |
| 光心 cx | 320.0 |
| 光心 cy | 240.0 |

### 2.2 雷达外参

| 参数 | 值 |
|------|------|
| 雷达相对相机 X 偏移 | 0 m（右侧为正） |
| 雷达相对相机 Y 偏移 | +0.09 m（前方为正，雷达在相机后方9cm） |
| 雷达相对相机 Z 偏移 | -0.12 m（上方为正，实际装飞机上 -0.18 m） |
| 相机俯仰角 | 0° |
| 相机高度 | 0.03 m |
| 角度容差 | ±0.5 rad（约 ±28.6°） |

### 2.3 AprilTag 配置

| 参数 | 值 |
|------|------|
| Tag 边长 | 0.15 m |
| Tag 家族 | tag36h11 |

---

## 三、功能需求

### 3.1 检测模块

| 功能 | 描述 | 优先级 | 参照 Python 模块 |
|------|------|--------|----------------|
| HSV颜色识别 | 支持 red/green/blue/black/white/red_laser 等12种颜色预设 | 🔴 必须 | `src/colorblob.py` |
| 椭圆/圆检测 | 拟合椭圆弧，筛选最大/第二大椭圆 | 🔴 必须 | `src/outsite.py` `detect_ellipses` |
| 梯形检测 | 四边形角度余弦过滤，面积排序筛选 | 🔴 必须 | `src/outsite.py` `detect_trapezoids` |
| 三角形检测 | 三点计算边长，旋转不变性处理 | 🔴 必须 | `src/outsite.py` `detect_triangle` |
| 杆状线检测 | 霍夫变换 + 角度分桶 + 平行性判定（间距5~70px） | 🔴 必须 | `src/outsite.py` `find_longest_straight_line` |
| QR码检测 | OpenCV QRCodeDetector，返回内容、中心坐标、面积 | 🔴 必须 | `src/other.py` `QR_detect` |
| AprilTag | apriltag C++ SDK，位姿估计，tag36h11家族 | 🔴 必须 | `src/other.py` `opencv_find_april_tag` |
| 条码检测 | zbar 库，返回内容、中心坐标 | 🟡 应该 | `src/other.py` `decodeDisplay` |
| 激光点检测 | 极小ROI内高亮点定位（V通道最亮像素） | 🟡 应该 | `src/colorblob.py` `detect_laser` |

### 3.2 融合模块

| 功能 | 描述 | 优先级 | 参照 Python 模块 |
|------|------|--------|----------------|
| ROS雷达数据源 | 订阅 `/scan` 话题，解析 LaserScan | 🔴 必须 | `src/radar/ros_source.py` |
| 模拟雷达数据源 | 无ROS环境下生成/读取模拟雷达数据 | 🔴 必须 | `src/radar/sim_source.py` |
| 自动数据源切换 | 自动检测ROS环境，无ROS时回退SimRadarSource | 🔴 必须 | `src/radar/fusion.py` |
| 按角度范围测距 | 给定角度区间，返回最近障碍物距离 | 🔴 必须 | `src/radar/fusion.py` `angle_to_distance` |
| 像素→雷达距离融合 | 像素坐标经相机内参+外参转换为射线，查雷达最近点 | 🔴 必须 | `src/radar/fusion.py` `site_to_distance` |
| 点群聚类 | 连续3点距离差≤3cm归为一簇，独立点为障碍物 | 🔴 必须 | `src/radar/fusion.py` `_detect_obstacles` |

### 3.3 通信模块

| 功能 | 描述 | 优先级 | 参照 Python 模块 |
|------|------|--------|----------------|
| 串口读写 | UART 256000波特，8N1，36字节数据包 | 🔴 必须 | `src/uartuse.py` |
| 协议解析 | 状态机解析0xFF 0xFE帧头，读模式码 | 🔴 必须 | `src/comm/serial_client.py` |
| 数据发送 | 组包发送（x/y/pixel/flag/state/angle/distance/apriltag等字段） | 🔴 必须 | `src/uartuse.py` |
| 校验和计算 | 累加和取低8位，协议完全对齐 | 🔴 必须 | `src/config/protocol.py` |

### 3.4 编排模块

| 功能 | 描述 | 优先级 | 参照 Python 模块 |
|------|------|--------|----------------|
| ROI提取器 | CircleROIExtractor / RectROIExtractor / ORBROIExtractor | 🔴 必须 | `src/pipeline/roi_extractor.py` |
| 形状分类器 | 级联形状识别（ellipse→trapezoid→triangle→pole） | 🔴 必须 | `src/pipeline/shape_classifier.py` |
| 检测编排器 | Pipeline入口：取帧→颜色检测→ROI→形状→融合→返回 | 🔴 必须 | `src/pipeline/orchestrator.py` |

### 3.5 模式处理（8种工作模式）

| 模式码 | 名称 | 检测内容 | 优先级 |
|--------|------|---------|--------|
| 0x00 | IDLE | 红色检测 + 雷达测距 | 🔴 必须 |
| 0x01 | CIRCLE | 白色色块内椭圆检测 | 🟡 应该 |
| 0x02 | SOUND | 纯雷达，无视觉 | 🟡 应该 |
| 0x03 | IDLE2 | 绿色色块 + 雷达测距 | 🟡 应该 |
| 0x04 | APRILTAG | AprilTag位姿估计 | 🔴 必须 |
| 0x05 | COLOR_BLOB | 蓝色色块检测 | 🟡 应该 |
| 0x06 | BARCODE | 条码识别 | 🟡 应该 |
| 0x07 | QR_2024 | 二维码识别（24年真题） | 🔴 必须 |

### 3.6 外设模块

| 功能 | 描述 | 优先级 | 参照 Python 模块 |
|------|------|--------|----------------|
| GPIO LED控制 | RGB三色LED组合7种颜色（indigo/purple/yellow/blue/green/red/empty/white） | 🟡 应该 | `src/facility2.py` |

---

## 四、非功能需求

### 4.1 性能

| 指标 | 目标 | 说明 |
|------|------|------|
| 帧率 | ≥ 50 FPS（640×480输入） | 不低于Python版 |
| 端到端延迟 | < 20ms/帧 | 取帧→检测→融合→串口发送 |
| 串口发送周期 | ≤ 50ms | 对应50Hz刷新率 |
| 内存占用 | < 100MB（运行时峰值） | ARM平台内存敏感 |

### 4.2 可靠性

| 指标 | 目标 |
|------|------|
| 雷达断线自动恢复 | 3秒内切换到SimRadarSource |
| 串口断线自动重连 | 最多5次尝试，每次间隔1秒 |
| 无崩溃运行时间 | ≥ 24小时连续运行 |

### 4.3 可移植性

| 平台 | 支持状态 |
|------|---------|
| Windows x86 | 开发/调试 |
| Linux x86 | 开发/测试 |
| Linux ARM (Raspberry Pi 3B+/4) | 生产部署 |
| Linux ARM (Jetson Nano/NX/Xavier) | 生产部署（优先CUDA版OpenCV） |

### 4.4 代码质量

- 编译通过无警告（`-Wall -Wextra -Wpedantic`）
- 通过 GoogleTest 单元测试，覆盖率 ≥ 70%（按模块区分）
- clang-tidy 静态分析无严重问题
- 单元测试：每个模块独立测试
- 集成测试：基于合成视频逐帧对比 Python vs C++ 输出

---

## 五、测试验收标准

### 5.1 功能验收

| 测试项 | 通过条件 |
|--------|---------|
| 颜色检测 | 在合成视频上，C++版检测到的红色块中心坐标与Python版偏差 ≤ 3像素 |
| 形状识别 | 形状类型（ellipse/trapezoid/triangle/pole）识别一致率 = 100% |
| QR码 | 检测标志（flag）和中心坐标与Python版一致 |
| AprilTag | tag_id检测一致，位置偏差 ≤ 5像素 |
| 雷达融合 | `site_to_distance()` 在相同输入像素坐标下，距离差 ≤ 5cm |
| 串口协议 | 36字节数据包逐字节与Python版发送内容一致 |

### 5.2 性能验收

| 测试项 | 通过条件 |
|--------|---------|
| 帧率 | 在Jetson Nano上 ≥ 40 FPS（生产部署基准） |
| 延迟 | 单帧处理时间（检测+融合+串口） ≤ 20ms |
| 内存 | 连续运行10分钟，内存不持续增长（无内存泄漏） |

### 5.3 回归测试数据集

使用与 Python 版相同的合成视频测试数据：

```
generate_test_video.py 生成的以下视频（需在 C++ 版中复现生成）：
  test_color_single.avi   — 红/绿/蓝/黄 单色块漂移
  test_multi_color.avi    — 红+绿 双色块同框
  test_trapezoid.avi      — 红色梯形漂移
  test_triangle.avi       — 红色三角形漂移
  test_ellipse.avi        — 红色圆/椭圆漂移
  test_multi_shape.avi    — 梯形+三角形+圆 同框
  test_pole.avi           — 平行竖线漂移
  test_laser.avi          — 极亮激光点游走
```

C++版需要有一个等效的 `generate_test_video.cpp`（或Python脚本调用OpenCV C++接口）来生成一致的测试视频。

---

## 六、技术选型

| 组件 | 选型 | 版本 |
|------|------|------|
| 编译器 | GCC / Clang | C++17 minimum |
| 构建系统 | CMake | ≥ 3.20 |
| OpenCV | libopencv-dev | ≥ 4.8.0 |
| 数学库 | Eigen3（可选） | ≥ 3.4 |
| YAML配置 | yaml-cpp | ≥ 0.8 |
| 串口通信 | Boost.Asio | header-only |
| AprilTag | apriltag C++ SDK | 3.x |
| 条码 | zbar | 0.10+ |
| 日志 | spdlog | ≥ 1.11 |
| 单元测试 | GoogleTest | ≥ 1.12 |
| 依赖管理 | Conan v2 | — |

---

## 七、接口兼容性要求

### 7.1 对外接口（与 Python 版的等价性）

C++版的所有检测/融合函数，其行为必须与 `src/` 下的 Python 模块完全等价。

关键接口对应关系：

| Python 函数/类 | C++ 函数/类 |
|---------------|------------|
| `colorblob.detect_color()` | `ColorDetector::detect()` |
| `outsite.detect_ellipses()` | `ShapeRecognizer::detectEllipses()` |
| `outsite.detect_trapezoids()` | `ShapeRecognizer::detectTrapezoids()` |
| `outsite.detect_triangle()` | `ShapeRecognizer::detectTriangle()` |
| `outsite.find_longest_straight_line()` | `ShapeRecognizer::findPoleLines()` |
| `other.QR_detect()` | `MarkerDetector::detectQR()` |
| `other.opencv_find_april_tag()` | `MarkerDetector::detectAprilTag()` |
| `other.decodeDisplay()` | `MarkerDetector::detectBarcode()` |
| `radar.fusion.RadarFusion.site_to_distance()` | `RadarFusion::siteToDistance()` |
| `radar.fusion.RadarFusion.angle_to_distance()` | `RadarFusion::angleToDistance()` |
| `radar.fusion.RadarFusion.get_obstacle()` | `RadarFusion::getObstacles()` |
| `pipeline.orchestrator.DetectionPipeline.run()` | `DetectionPipeline::run()` |

### 7.2 数据类型对应

| Python 类型 | C++ 类型 |
|------------|---------|
| `dict` | `struct` / `class` |
| `List[dict]` | `std::vector<Struct>` |
| `Tuple[int,int]` | `cv::Point2i` / `std::pair<int,int>` |
| `Optional[T]` | `std::optional<T>` |
| `bytes` | `std::vector<uint8_t>` |

---

## 八、项目外约束

1. **不得修改下位机固件**：协议字段严格对齐
2. **不得改变检测算法逻辑**：与Python版行为完全等价
3. **不得在ARM平台引入额外运行时依赖**：所有依赖需静态链接或系统库
4. **文档和代码必须同步更新**：每完成一个子任务，同步更新 `docs/AI工作文档.md` 的进度状态

---

*文档版本记录：*\
*v1.0 — 2026-05-27 — 初始版本*
