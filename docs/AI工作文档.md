# AI 工作文档 — 视觉竞赛库重构项目

> **维护者**：项目Owner
> **用途**：每次启动新的 AI Agent 时，只需让 Agent 读本文件即可，无需额外说明
> **代码仓库**：`https://github.com/CQUT-302/FlightVersionOnRaspirryPi.git`

---

## 一、项目概述

本项目是一个**视觉+雷达融合的竞赛处理库**，目标平台为 Raspberry Pi / Jetson 等嵌入式设备，主要功能：

- HSV 颜色识别（red/green/blue/black 等多种颜色预设）
- 形状检测：椭圆、梯形、三角形、杆状线
- ORB 模板匹配（logo 识别）
- 特殊标记：AprilTag + QR码 + 条码
- LiDAR 点云处理 + 相机-雷达几何融合
- 串口 UART 协议通信（波特率 256000）
- RPi GPIO 外设控制（LED）
- asyncio 主程序编排（协程模式）

**核心代码量**：约 3000+ 行 Python，分布在 `src/` 下的 7 个模块中。

**项目当前状态**：功能完整，已完成比赛，代码处于"竞赛原型"阶段，需要工程化重构。

---

## 二、整体计划索引

详细的重构计划在同级的 `整改计划.md` 中，分为 4 个 Phase：

| Phase | 名称 | 预计工时 | 核心目标 |
|-------|------|---------|---------|
| Phase 1 | 奠基 | 8-12h | 建立配置层 + 基础工程保障，不改核心逻辑 |
| Phase 2 | 结构重塑 | 20-30h | 拆分 allin.py + main.py，建立统一数据流 |
| Phase 3 | 深度优化 | 16-24h | 解耦 ROS + 统一日志 + 调参工具闭环 |
| Phase 4 | 工程保障 | 8-12h | 单元测试 + 集成测试 + 类型提示 + CI |

每个 Phase 内部又分为若干子任务（如 T1.1、T1.2、T1.3...）。

---

## 三、当前工作状态

> **【重要】每次交接时更新此部分**
> **最后更新**：2026-05-24
> **当前进度**：Phase 3 进行中（T3.2 已完成，push 待网络恢复）

```
总体进度：
  Phase 1：■ 已完成（T1.1-T1.6）
  Phase 2：■ 已完成（T2.1、T2.2、T2.3、T2.4 全部完成）
  Phase 3：□ 未开始
  Phase 3 T3.1：■ 已完成
  Phase 3 T3.2：■ 已完成（commit 7fa6e14，push 待网络恢复）
  Phase 4：□ 未开始

最近完成：
  2026-05-24：Phase 3 T3.2（统一日志系统）
    - 分支：23-nzq
    - 提交：7fa6e14（push 待网络恢复）
    - 内容：src/utils/logger.py + __init__.py + main.py 等 8 个文件
    - 改动：建立统一日志模块，替换全项目 14 处 print 为结构化日志
  2026-05-24：Phase 3 T3.1（解耦 ROS 依赖）
    - 分支：23-nzq
    - 提交：0c620c9
    - 内容：src/radar/base.py + ros_source.py + sim_source.py + fusion.py + __init__.py
    - 改动：建立 RadarSource 抽象基类，RadarFusion 自动检测数据源（ROS/模拟），无 ROS 时自动回退到 SimRadarSource
  2026-05-24：Phase 2 T2.4（简化 main.py）
    - 内容：src/comm/serial_client.py + __init__.py + main.py 重写（380行→140行）
  2026-05-24：Phase 2 T2.3（拆分 main.py → src/modes/）
    - 分支：23-nzq
    - 提交：e07d288
    - 内容：src/modes/base.py + idle_mode.py + qr_mode.py + stub_modes.py + __init__.py
  2026-05-24：Phase 2 T2.2（拆分 allin.py → src/pipeline/）
    - 分支：23-nzq
    - 提交：38c1df4（待 push，网络中断）
    - 内容：src/pipeline/roi_extractor.py + shape_classifier.py + orchestrator.py
  2026-05-24：Phase 2 T2.1（建立统一数据类型）
    - 分支：23-nzq
    - 提交：b79d0b7
    - 内容：src/core/types.py + src/core/adapters.py + src/core/__init__.py
  2026-05-23：Phase 1（T1.1-T1.6）建立配置层和项目骨架
    - 分支：23-nzq
    - 提交：490222c
    - 内容：src/config/ 配置层 + requirements.txt + .gitignore + scene.yaml.example

当前阻塞：
  无
```

---

## 四、权威来源说明

```
GitHub仓库  ← 唯一的权威真相来源
https://github.com/CQUT-302/FlightVersionOnRaspirryPi

代码、计划、报告全部在这里
        ↑ pull
        │
       ┌┴──────────────┐
       ↓               ↓
  [Agent A 工作区]  [Agent B 工作区]
       │               │
       └─── 工作完成后 push ──┘
```

**规则**：
- **代码**：始终从 GitHub 拉取，工作后推送回去
- **计划**：以 GitHub 仓库里的 `docs/整改计划.md` 和 `docs/AI工作文档.md` 为准
- **报告**：写入 `reports/` 目录，完成后随代码一起推送

---

## 五、任务领取与工作流程

### 5.1 如何领取任务

本项目按 `docs/整改计划.md` 中的 Phase 顺序推进。每次工作时：

1. 从 GitHub 拉取最新代码：`git pull`
2. 查看上方"当前工作状态"，确认从哪个子任务开始
3. 在本文件的"当前工作状态"栏更新进度（注明"由 [名字/Agent ID] 领取"）
4. 按下方"单次任务工作流程"执行

### 5.2 单次任务工作流程

```
步骤 1 → git pull 拉取最新，确认无冲突

步骤 2 → 读取 docs/整改计划.md 中对应 Phase 的具体任务描述（T1.x 等）

步骤 3 → 读取任务直接涉及的源代码文件（通常 1-3 个），确认改动范围

步骤 4 → 对要改动的文件，先 git add + git commit 记录改动前状态

步骤 5 → 按任务要求创建/修改文件

步骤 6 → 运行相关示例脚本，确认功能等价于改动前

步骤 7 → 按下方"六、工作报告模板"填写，写入 reports/[任务编号]-[日期].md

步骤 8 → git add + git commit（格式：[T{x}.{y}] {简短描述}）+ git push

步骤 9 → 更新本文件"当前工作状态"，提交并推送本文件的更新
```

---

## 六、工作报告模板

每完成一个子任务，必须填写此模板，写入 `reports/` 目录下。

**文件命名**：`[任务编号]-[日期].md`，例：`T1.3-20250523.md`

```markdown
# 工作报告：{任务编号} {任务名称}

## 基本信息

| 字段 | 内容 |
|------|------|
| 任务编号 | T{x}.{y} |
| 任务名称 | {从整改计划.md抄过来} |
| 执行者 | {你的名字或Agent ID} |
| 开始时间 | YYYY-MM-DD HH:MM |
| 结束时间 | YYYY-MM-DD HH:MM |
| 耗时 | X 小时 |

## 任务目标

{从整改计划.md 抄过来}

## 完成情况

- [x] {具体完成项1}
- [x] {具体完成项2}
- [ ] {未完成项，写明原因}

## 新建/修改的文件清单

| 文件路径 | 操作类型 | 改动说明 |
|----------|---------|---------|
| src/config/hardware.py | 新建 | 从main.py提取相机内参 |
| src/__init__.py | 新建 | 建立src包结构 |
| main.py | 修改 | 引用src.config.hardware替代原hardcode |

## 核心代码改动说明

{描述关键代码逻辑的改动，如果是新建文件，描述其核心职责}

## 验证方式与结果

{运行了什么脚本/测试，验证了什么，结果如何}

## 遗留问题

无

## 下一步建议

{基于你的工作，对下一个任务有什么提示或注意事项}
```

---

## 七、测试数据说明

**测试视频**：项目使用 `generate_test_video.py` 生成的合成视频（`test_video.avi`）作为回归测试数据。

- 视频**不提交**到版本库（`.gitignore` 已配置 `*.avi`）
- 需要测试时，在本地运行 `python generate_test_video.py` 重新生成即可
- 合成视频约 8 个场景，覆盖红色检测、形状识别、杆检测、激光检测等场景

---

## 八、代码风格规范

### 8.1 必须遵守

1. **功能等价优先**：任何重构不得改变检测结果。改动前后用同一个测试视频验证，召回率不得下降。
2. **不过度设计**：只实现任务要求的功能，不做超出任务范围的"优化"。
3. **保留原注释**：原代码中的中文注释应保留。
4. **不改文件名**：除非任务明确要求，否则不重命名现有文件。

### 8.2 推荐做法

1. **先读代码再动手**：至少完整读一遍要改动的文件，理解后再改。
2. **增量提交**：每完成一个小改动就提交一次，不要等到最后一次性提交。
3. **简洁提交信息**：格式 `[T{x}.{y}] {简短描述}`
   - 例：`[T1.2] 提取相机内参到 hardware.py`
4. **不写冗余注释**：代码本身能说明的不要加注释。

### 8.3 禁止事项

1. 不删除任何现有功能代码（除非任务明确要求）
2. 不改变任何检测算法的核心逻辑
3. 不提交 `.avi` 视频文件、`.pyc` 缓存文件到版本库
4. 不提交 `reports/` 目录以外的临时文件

---

## 九、快速启动（Agent 首次读取时）

如果你是一个新的 Agent，第一次参与本项目，按以下顺序阅读：

```
1. 本文件（docs/AI工作文档.md）← 你现在在这里，搞清楚项目是什么
2. docs/整改计划.md 的"六、重构顺序" ← 了解总体执行路径
3. docs/整改计划.md 的对应 Phase   ← 了解你要做的那个 Phase 的具体任务
4. 相关源代码文件              ← 开始工作
```

**不需要读全部源代码**，除非任务明确要求。

### 9.1 Agent 运行环境约束

> 新 Agent 必须遵守以下约束，否则会卡死或行为异常。

1. **切换盘符**：工作空间在 E 盘，每次操作前先执行 `E:` 切换到 E 盘
2. **读文件**：用 `Read` 工具直接读文件，禁止用 WebFetch 等外部工具
3. **禁止搜索**：
   - 禁止用 `Grep` / `Glob` / `Shell` 中的 find/grep 搜索代码，会卡死
   - 禁止用 `ls` / `dir` 遍历目录，会卡死
   - 需要找文件时，直接用已知路径读文件，或在文档中已有明确路径时直接读
4. **Git 操作**：push 失败后不要反复重试，先在本地完成所有工作，网络恢复后一次性 push
5. **工作流程**：必须严格按第五节"单次任务工作流程"执行，不得跳过步骤 7（写报告）和步骤 9（更新本文档）

---

## 十、参考资料索引

| 文件 | 仓库内路径 | 何时读 |
|------|-----------|--------|
| 整改计划.md | `docs/整改计划.md` | 任何时候 |
| AI工作文档.md | `docs/AI工作文档.md` | 首次/交接时 |
| 诊断1-5.md | `docs/诊断1.md` 等 | 开始工作前 |
| src/colorblob.py | `src/colorblob.py` | Phase 1 T1.3、Phase 2 T2.1 |
| src/outsite.py | `src/outsite.py` | Phase 2 T2.2 |
| src/allin.py | `src/allin.py` | Phase 2 T2.2 |
| src/radar5.py | `src/radar5.py` | Phase 3 T3.1 |
| src/uartuse.py | `src/uartuse.py` | Phase 1 T1.4 |
| src/other.py | `src/other.py` | Phase 2 T2.1 |
| main.py | `main.py` | Phase 1 T1.2、Phase 2 T2.3 |
| generate_test_video.py | `generate_test_video.py` | 生成测试数据时 |
| test_video.avi | 本地生成，不入版本库 | 回归测试时 |

---

## 十一、仓库目录结构（重构后）

```
FlightVersionOnRaspirryPi/
├── docs/                        # 文档目录（Phase 1 新增）
│   ├── 整改计划.md
│   ├── AI工作文档.md
│   ├── 诊断1.md ~ 诊断5.md
│   └── README.md
│
├── reports/                      # 工作报告目录（Phase 1 新增）
│   ├── T1.1-20250601.md
│   ├── T1.2-20250603.md
│   └── ...
│
├── src/
│   ├── __init__.py              # Phase 1
│   └── config/                   # Phase 1
│       ├── __init__.py
│       ├── hardware.py
│       ├── scene.py
│       ├── protocol.py
│       └── modes.py
│
├── tests/                       # Phase 4
├── main.py
├── generate_test_video.py        # 测试数据生成脚本（应提交）
├── requirements.txt              # Phase 1
├── .gitignore                    # Phase 1
└── ...
```

> **注意**：请勿在本地创建与仓库平行的"第二套"文档体系。所有工作基于 GitHub 仓库进行。

---

*本文件由项目 Owner 维护，每次任务交接后更新"第三部分：当前工作状态"。*
