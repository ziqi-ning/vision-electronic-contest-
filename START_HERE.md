# FVS-Cpp — Agent 启动入口

> **此文件是唯一的入口**。只需让 Agent 读这个文件，即可自动完成后续所有工作。

---

## 第一步：读文档

按以下顺序读取（不要跳顺序）：

```
1. docs/AGENT.md       ← 必读，搞清楚项目结构、工作流程、代码规范
2. docs/SPEC.md        ← 必读，搞清楚需求、约束、测试标准
3. docs/PLAN.md        ← 必读，搞清楚 Phase 顺序和当前状态
```

读取完成后，根据 "docs/AGENT.md → 第四部分：当前工作状态" 确认下一个要做的子任务。

---

## 第二步：开始工作

按 `docs/AGENT.md` 第五部分"单次任务工作流程"执行：

```
1. 确认当前工作目录是 FVS-Cpp（E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version3\FVS-Cpp）
2. 读取 docs/PLAN.md 中对应子任务的具体要求
3. 读取 Python 版对应源文件（作为功能等价性参考）
   - Python版仓库：https://github.com/CQUT-302/FlightVersionOnRaspirryPi
4. git add + commit 记录当前状态
5. 实现代码
6. 测试验证（合成视频对比 / 单元测试）
7. 写报告到 reports/[任务编号]-[日期].md
8. git commit + push
9. 更新 docs/AGENT.md 的"当前工作状态"
10. git commit + push AGENT.md 的更新
```

---

## 第三步：约束

- 工作空间在 **E 盘**，每次操作前先执行 `E:`
- **不要跳步骤**，特别是写报告（步骤7）和更新进度（步骤9）
- commit 格式：`[Phase-X] {简短描述}`
- 视频文件（`.avi`）不提交到版本库
- **功能等价性第一**：任何改动必须与 Python 版行为一致

---

## 当前状态

> **最后更新**：2026-05-27
> 尚未开始开发。所有 Phase 均为未开始状态。
> 下一个任务：Phase 0 / T0.1 — 建立 CMake 项目结构

详见 `docs/AGENT.md` → 第四部分：当前工作状态
