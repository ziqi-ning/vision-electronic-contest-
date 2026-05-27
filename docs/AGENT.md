# FVS-Cpp — AI Agent 工作文档

> **维护者**：项目Owner\
> **用途**：每次启动新的 AI Agent 时，只需让 Agent 读本文件即可，无需额外说明\
> **远程仓库**：`https://github.com/ziqi-ning/uav-multisensor-fusion.git`\
> **分支**：`cpp-restruct`（所有代码改动必须 commit 到本分支，然后 push）\
> **工作空间**：`E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version3\FVS-Cpp`\
> **⚠️ 重要**：每次完成任务后务必 commit + push 到 `cpp-restruct`，并更新本文档"当前工作状态"再 commit + push 一次

---

## 一、项目概述

本项目是 `uav-multisensor-fusion`（Python版）的 C++ 重写版本。

- **Python版仓库**：`https://github.com/CQUT-302/FlightVersionOnRaspirryPi`
- **Python版状态**：主分支，功能完整
- **C++版分支**：`cpp-restruct`（从 `main` 分出）
- **目标平台**：ARM嵌入式（树莓派 / Jetson）
- **核心功能**：HSV颜色识别、形状检测（椭圆/梯形/三角形/杆线）、QR/AprilTag/条码、雷达融合测距、UART串口通信
- **总工期**：Phase 0-5，约 7~12 周

---

## 二、权威来源说明

```
GitHub仓库（C++版代码）  ←  push 到这里
https://github.com/ziqi-ning/uav-multisensor-fusion.git
分支：cpp-restruct

GitHub仓库（Python版参考）  ←  功能等价性的唯一权威参考
https://github.com/ziqi-ning/uav-multisensor-fusion.git
（Python版在 main 分支）

FVS-Cpp本地仓库（C++版）
E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version3\FVS-Cpp
```

**规则**：
- **Python版**：始终以 GitHub 仓库的 main 分支源码作为功能等价性的唯一权威参考
- **C++版**：在本地仓库工作，**每完成一个子任务必须 commit 并 push 到 cpp-restruct 分支**
- **计划文档**：以本目录下的 `docs/SPEC.md` 和 `docs/PLAN.md` 为准
- **工作报告**：写入 `reports/` 目录，完成后 commit 并 push

---

## 三、整体计划索引

详细计划在 `docs/PLAN.md`，分为 6 个 Phase：

| Phase | 名称 | 预计工时 | 核心目标 |
|-------|------|---------|---------|
| Phase 0 | 项目骨架 | 1周 | CMake结构 + 类型系统 + 配置层 + 日志 + 工具 |
| Phase 1 | 通信层 | 1周 | 串口（Boost.Asio）+ LED控制（GPIO） |
| Phase 2 | 检测层 | 2~3周 | 颜色/形状/特殊标记识别（最高风险） |
| Phase 3 | 融合层 | 1~2周 | 雷达数据获取 + 相机-雷达融合 |
| Phase 4 | 编排层+模式层 | 1~2周 | Pipeline编排 + 8种工作模式 |
| Phase 5 | 测试+优化 | 1~2周 | 单元测试 + 集成测试 + 性能优化 |

每个 Phase 内部又分为若干子任务（如 T0.1、T2.1 等）。

---

## 四、当前工作状态

> **【重要】每次交接时更新此部分**
> **最后更新**：2026-05-27
> **当前进度**：Phase 0 进行中，T0.3 已完成，下一个任务 T0.4

```
总体进度：
  Phase 0：◐ 进行中（T0.1 √  T0.2 √  T0.3 √  | T0.4 → T0.5 ○）
  Phase 1：○ 未开始（T1.1 ~ T1.3）
  Phase 2：○ 未开始（T2.1 ~ T2.4）
  Phase 3：○ 未开始（T3.1 ~ T3.3）
  Phase 4：○ 未开始（T4.1 ~ T4.4）
  Phase 5：○ 未开始（T5.1 ~ T5.4）

当前阻塞：
  无

最近完成：
  2026-05-27：Phase 0 T0.3 — 建立配置加载层
    - 完成：config/scene.yaml、HardwareConfig.h、SceneConfig.h/.cpp、Protocol.h
    - commit: c0dcc01（报告）/ 7f815a9（T0.2）
  2026-05-27：Phase 0 T0.2 — 建立统一类型系统
    - 完成：include/Types.h（11个核心结构体）、include/Version.h（枚举+常量）、src/Types.cpp（方法实现）
    - commit: 3582044（T0.2） / 5c7986d（报告）
  2026-05-27：Phase 0 T0.1 — 建立 CMake 项目结构
    - 完成：目录骨架、CMakeLists.txt、conanfile.txt、src/main.cpp、.gitignore
    - commit: 6bbb8f2
```

---

## 五、任务领取与工作流程

### 5.1 如何领取任务

按 `docs/PLAN.md` 中的 Phase 顺序推进。每次工作时：

1. 确认当前进度（见上方"当前工作状态"）
2. 从下一个未开始的子任务开始
3. 在本文件的"当前工作状态"栏注明"由 [名字/Agent ID] 领取"
4. 按下方"单次任务工作流程"执行

### 5.2 单次任务工作流程

```
步骤 1 → 确认当前工作目录是 FVS-Cpp
          工作空间：E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version3\FVS-Cpp

步骤 2 → 读取 docs/PLAN.md 中对应 Phase 的具体任务描述
          确认子任务编号（如 T2.1）、目标、验收标准

步骤 3 → 读取任务直接涉及的 Python 版源代码文件
          作为功能等价性的唯一参考
          通常需要读：src/colorblob.py / src/outsite.py / src/other.py / src/radar/fusion.py

步骤 4 → 对要改动的文件，先 git add + git commit 记录当前状态

步骤 5 → 按任务要求创建/修改 C++ 文件
          严格遵循 docs/SPEC.md 中的接口对应表
          保持与 Python 版完全等价的行为

步骤 6 → 运行相关测试，确认功能等价于 Python 版
          Phase 2+：用合成视频逐帧对比 Python vs C++ 输出

步骤 7 → 按下方"工作报告模板"填写，写入 reports/[任务编号]-[日期].md

步骤 8 → git add + git commit（格式：`[Phase-X] {简短描述}`）并 **git push 到 cpp-restruct 分支**

步骤 9 → 更新本文件"当前工作状态"，commit **并 push** 本文件的更新
```

---

## 六、工作报告模板

每完成一个子任务，必须填写此模板，写入 `reports/` 目录。

**文件命名**：`[任务编号]-[日期].md`，例：`T2.1-20260601.md`

```markdown
# 工作报告：[任务编号] [任务名称]

## 基本信息

| 字段 | 内容 |
|------|------|
| 任务编号 | T{x}.{y} |
| 任务名称 | {从PLAN.md抄过来} |
| 执行者 | {你的名字或Agent ID} |
| 开始时间 | YYYY-MM-DD HH:MM |
| 结束时间 | YYYY-MM-DD HH:MM |
| 耗时 | X 小时 |

## 任务目标

{从PLAN.md抄过来}

## 完成情况

- [x] {具体完成项1}
- [x] {具体完成项2}
- [ ] {未完成项，写明原因}

## 新建/修改的文件清单

| 文件路径 | 操作类型 | 改动说明 |
|---------|---------|---------|
| src/detection/ColorDetector.h | 新建 | ColorDetector类头文件 |
| src/detection/ColorDetector.cpp | 新建 | ColorDetector类实现 |
| CMakeLists.txt | 修改 | 添加ColorDetector编译目标 |

## 核心代码改动说明

{描述关键代码逻辑的改动，如果是新建文件，描述其核心职责}

## 功能等价性验证

{描述如何验证与Python版的行为一致，用了什么测试数据，结果如何}

## 与Python版的关键差异记录

{如果有任何与Python版行为不一致的地方（即使很小），必须记录在此}

## 遗留问题

无

## 下一步建议

{基于你的工作，对下一个任务有什么提示或注意事项}
```

---

## 七、测试数据说明

### 7.1 合成视频（用于回归测试）

项目使用与 Python 版相同的合成视频作为测试数据：

```
generate_test_video.py（Python版仓库中）生成的以下视频：
  test_color_single.avi   — 红/绿/蓝/黄 单色块漂移
  test_multi_color.avi    — 红+绿 双色块同框
  test_trapezoid.avi      — 红色梯形漂移
  test_triangle.avi       — 红色三角形漂移
  test_ellipse.avi        — 红色圆/椭圆漂移
  test_multi_shape.avi     — 梯形+三角形+圆 同框
  test_pole.avi           — 平行竖线漂移
  test_laser.avi          — 极亮激光点游走

视频不提交到版本库（.gitignore 中配置 *.avi）
需要测试时，在 Python 版仓库中运行 python generate_test_video.py 重新生成
```

### 7.2 逐帧对比脚本

Phase 2 和 Phase 3 完成后，需要运行逐帧对比脚本验证等价性：

```
tests/integration/compare_with_python.py
  - 输入：合成视频路径
  - 输出：Python版结果 vs C++版结果，逐帧差异报告
  - 通过标准：100帧中偏差超过阈值的不超过3帧
```

---

## 八、代码风格规范

### 8.1 必须遵守

1. **功能等价优先**：任何实现不得改变检测结果。改动前后用同一个测试视频验证，结果偏差不超过 SPEC.md 中的阈值
2. **不过度设计**：只实现任务要求的功能，不做超出任务范围的"优化"
3. **保留原注释风格**：Python 版的中文注释说明的算法逻辑，在 C++ 版中用相同的逻辑注释
4. **文件名遵循规范**：Phase 0 的文件见 PLAN.md；后续按模块组织
5. **命名遵循规范**：
   - 类名：`PascalCase`（如 `ColorDetector`）
   - 函数名：`PascalCase`（如 `detectEllipses`，与 Python 版 `detect_ellipses` 对应驼峰）
   - 常量：`kCamelCase`（如 `kDefaultBais`）
   - 枚举值：`PascalCase`（如 `WorkMode::IDLE`）
6. **编译无警告**：`clang-tidy` 无严重问题，`-Wall -Wextra -Wpedantic` 无警告

### 8.2 推荐做法

1. **先读代码再动手**：至少完整读一遍对应的 Python 版源文件，理解后再开始写 C++
2. **增量提交**：每完成一个小模块就提交一次，不要等到最后一次性提交
3. **提交信息格式**：`[Phase-X] {简短描述}`，如 `[Phase-2] ColorDetector C++ implementation`
4. **差异记录**：如果有任何与 Python 版行为不一致的地方（即使很小），必须记录在工作报告的"与Python版的关键差异记录"章节

### 8.3 禁止事项

1. 不删除任何检测逻辑代码（除非任务明确要求）
2. 不改变任何检测算法的核心逻辑
3. 不在 commit message 中写 `fix` / `fix:` 以外的描述性前缀（保留 `[Phase-X]` 格式）
4. 不提交 `.avi` 视频文件到版本库
5. 不在 `reports/` 目录以外写临时文件

---

## 九、快速启动（Agent 首次读取时）

如果你是一个新的 Agent，第一次参与本项目，按以下顺序阅读：

```
1. 本文件（docs/AGENT.md）← 你现在在这里，搞清楚项目是什么
2. docs/SPEC.md           ← 搞清楚需求和约束
3. docs/PLAN.md 的"七、执行顺序" ← 搞清楚总体执行路径
4. docs/PLAN.md 的对应 Phase ← 了解你要做的那个 Phase 的具体任务
5. Python 版对应源文件    ← 开始工作（功能等价性参考）
```

**不需要读全部源代码**，除非任务明确要求。

### 9.1 Agent 运行环境约束

> 新 Agent 必须遵守以下约束，否则会卡死或行为异常。

1. **切换盘符**：工作空间在 E 盘，每次操作前先执行 `E:` 切换到 E 盘
2. **读文件**：用 `Read` 工具直接读文件，禁止用 WebFetch 等外部工具
3. **搜索**：
   - 需要找文件时，直接用 Glob / Grep（比 Shell grep 快）
   - 禁止用 `ls` / `dir` 遍历大目录，会卡死
4. **Git 操作**：
   - C++ 版在本地 FVS-Cpp 仓库工作，commit 后按需 push
   - Python 版在 GitHub，不需要在本地 clone
5. **工作流程**：必须严格按第五节"单次任务工作流程"执行，不得跳过步骤 7（写报告）和步骤 9（更新本文档）

### 9.2 Python 版源代码索引

按任务需要读取：

| 任务 | Python 版必读文件 |
|------|-----------------|
| T0.2 类型系统 | `src/core/types.py` |
| T0.3 配置层 | `src/config/hardware.py` + `scene.py` + `protocol.py` + `modes.py` |
| T1.1 串口 | `src/uartuse.py` + `src/comm/serial_client.py` |
| T2.1 颜色检测 | `src/colorblob.py` + `src/allin.py` |
| T2.2 形状识别 | `src/outsite.py` |
| T2.3 特殊标记 | `src/other.py` |
| T3.1~3.2 雷达融合 | `src/radar/fusion.py` + `base.py` + `ros_source.py` + `sim_source.py` |
| T4.1~4.3 编排+模式 | `src/pipeline/orchestrator.py` + `roi_extractor.py` + `shape_classifier.py` + `src/modes/*.py` |

---

## 十、参考资料索引

| 文件 | 位置 | 何时读 |
|------|------|--------|
| SPEC.md | `docs/SPEC.md` | 任何时候 |
| PLAN.md | `docs/PLAN.md` | 任何时候 |
| AGENT.md | `docs/AGENT.md` | 首次/交接时 |
| Python版 types.py | GitHub: `src/core/types.py` | T0.2 |
| Python版 hardware.py | GitHub: `src/config/hardware.py` | T0.3, T3.2 |
| Python版 colorblob.py | GitHub: `src/colorblob.py` | T2.1 |
| Python版 outsite.py | GitHub: `src/outsite.py` | T2.2 |
| Python版 other.py | GitHub: `src/other.py` | T2.3 |
| Python版 fusion.py | GitHub: `src/radar/fusion.py` | T3.2 |
| Python版 serial_client.py | GitHub: `src/comm/serial_client.py` | T1.1 |
| Python版 orchestrator.py | GitHub: `src/pipeline/orchestrator.py` | T4.3 |
| Python版 modes/*.py | GitHub: `src/modes/` | T4.1~4.2 |
| Python版 pyproject.toml | GitHub: `pyproject.toml` | 项目配置参考 |

---

## 十一、C++ 项目目录结构（目标状态）

```
FVS-Cpp/
├── docs/
│   ├── SPEC.md            ← 需求规格（必读）
│   ├── PLAN.md            ← 施工计划（必读）
│   └── AGENT.md           ← 本文件（必读）
│
├── reports/                ← 工作报告目录
│   ├── T0.1-20260601.md
│   └── ...
│
├── src/
│   ├── main.cpp           ← 主程序入口
│   │
│   ├── config/            ← Phase 0 T0.3
│   │   ├── HardwareConfig.h
│   │   └── SceneConfig.h/.cpp
│   │
│   ├── detection/         ← Phase 2 T2.1~2.3
│   │   ├── ColorDetector.h/.cpp
│   │   ├── ShapeRecognizer.h/.cpp
│   │   └── MarkerDetector.h/.cpp
│   │
│   ├── fusion/             ← Phase 3 T3.1~3.2
│   │   ├── RadarSource.h
│   │   ├── ROSRadarSource.h/.cpp
│   │   ├── SimRadarSource.h/.cpp
│   │   └── RadarFusion.h/.cpp
│   │
│   ├── comm/               ← Phase 1 T1.1
│   │   └── SerialPort.h/.cpp
│   │
│   ├── modes/              ← Phase 4 T4.1~4.2
│   │   ├── ModeHandler.h
│   │   ├── IdleMode.h/.cpp
│   │   ├── QRMode.h/.cpp
│   │   └── ...
│   │
│   ├── pipeline/           ← Phase 4 T4.3
│   │   ├── ROIExtractor.h
│   │   ├── CircleROIExtractor.h/.cpp
│   │   ├── ShapeClassifier.h/.cpp
│   │   └── DetectionPipeline.h/.cpp
│   │
│   ├── hardware/           ← Phase 1 T1.2
│   │   └── LEDController.h/.cpp
│   │
│   └── utils/              ← Phase 0 T0.4~0.5
│       ├── Logger.h
│       ├── RingBuffer.h
│       └── MathUtils.h
│
├── include/                ← 公共头文件
│   ├── Types.h            ← Phase 0 T0.2（核心类型）
│   └── Version.h           ← Phase 0 T0.2（枚举/常量）
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── config/                ← 运行时配置
│   └── scene.yaml
│
├── CMakeLists.txt         ← Phase 0 T0.1
├── conanfile.txt          ← Phase 0 T0.1
└── README.md
```

---

*本文件由项目 Owner 维护，每次任务交接后更新"第四部分：当前工作状态"。*
