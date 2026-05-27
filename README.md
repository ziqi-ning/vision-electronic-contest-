# FVS-Cpp

> C++ 重写版：`uav-multisensor-fusion`（Python版）

**状态**：规划中，尚未开始开发\
**分支**：`cpp-restruct`\
**目标平台**：ARM嵌入式（树莓派 / Jetson）

---

## 项目简介

`uav-multisensor-fusion` 的 C++ 重写版本，目标是将核心感知处理逻辑从 Python 迁移到 C++，在 ARM 嵌入式平台上获得更高性能和更简化的部署。

## 核心功能

| 模块 | 功能 |
|------|------|
| **检测层** | HSV颜色识别、椭圆/梯形/三角形/杆状线检测、QR码/AprilTag/条码 |
| **融合层** | 相机-雷达几何融合、点群聚类障碍物检测 |
| **通信层** | UART串口协议（256000波特，36字节数据包） |
| **编排层** | Pipeline检测编排、8种工作模式 |
| **外设层** | GPIO LED控制 |

## 技术栈

| 组件 | 选型 |
|------|------|
| C++标准 | C++17 |
| 构建系统 | CMake 3.20+ |
| 视觉库 | OpenCV 4.8+ |
| 串口 | Boost.Asio |
| 日志 | spdlog |
| 测试 | GoogleTest |
| 依赖管理 | Conan v2 |

## 文档索引

| 文档 | 说明 |
|------|------|
| `docs/SPEC.md` | 需求规格文档（必读） |
| `docs/PLAN.md` | 施工计划，分Phase执行（必读） |
| `docs/AGENT.md` | AI Agent工作文档，接手指南（必读） |

## 快速开始

**文档尚未编写**，请先阅读 `docs/AGENT.md` 了解如何开始工作。

## 与 Python 版的关系

- **Python版**（`main`分支）：当前主分支，功能完整
- **C++版**（`cpp-restruct`分支）：重写版本，功能等价于Python版
- 两个版本使用完全相同的测试数据（合成视频）

## 参考：Python 版

- 仓库：`https://github.com/CQUT-302/FlightVersionOnRaspirryPi`
- C++版开发时请始终以 Python 版源码作为功能等价性的唯一权威参考
