# 🤖 智能体设计模式（Agentic Design Patterns）

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![GitHub stars](https://img.shields.io/github/stars/gtesei/agentic_design_patterns?style=social)](https://github.com/gtesei/agentic_design_patterns/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/gtesei/agentic_design_patterns?style=social)](https://github.com/gtesei/agentic_design_patterns/network)
[![GitHub release](https://img.shields.io/github/v/release/gtesei/agentic_design_patterns?include_prereleases&sort=semver)](https://github.com/gtesei/agentic_design_patterns/releases)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[English](README.md)

> **把你的 AI 应用从简单提示词进化为成熟的智能系统。**

## 🧭 原则

本仓库围绕四条承诺构建。它们决定了什么留下、什么被删除。

1. **精炼，不求大而全。** 这**不是**一个包含成百上千个、没人会真正读懂或记得住的模式的目录。它是一份**面向人类理解与记忆的精炼目录**。冗长的模式清单不是优点，而是噪音。
2. **硬上限：最多 28 个模式。** 如果某个模式并非必需，就把它移除。新增一个，就要删掉另一个。这个上限是一种强制约束，而不是远期目标。
3. **示例必须真的能跑起来。** 每个示例都会在**自动化的每周冒烟测试**中运行，同时覆盖 **Python 与 TypeScript**，并且必须通过。（见 [`.github/workflows/weekly-smoke.yml`](./.github/workflows/weekly-smoke.yml)。）跑不起来的教学代码就是在说谎的教学代码。
4. **要看具体相关性，而不只是论文。** 对每一个模式，我们都会衡量它在**真实编码 Agent**中的呈现方式 —— 目前对标的是 [Pi](https://github.com/earendil-works/pi) —— 并把分析结果写入每个模式目录下的 `pi.md`，同时提供一份[包含覆盖度热力图的仓库级汇总](./pi.md)。我们还维护一份 [`diff.md`](./diff.md) 来澄清那些常被混淆的模式对：**如果两个模式无法被清晰地区分开来，那么其中一个大概率不该留在这里**。一个只存在于某篇论文里的模式，同样是它可能不属于这里的信号。

---

AI 演进的速度太快，传统书籍很难保持时效，尤其是在智能体（agentic）系统这种快速变化的领域。这正是本仓库被定位为关于智能体 AI 最好的“活书（living book）”之一的原因：它是一个全面、动手实践的设计模式合集，用于构建健壮的 AI 智能体，并持续更新真实世界的实现、可运行的示例，以及为可扩展、可维护 AI 应用提供的详细架构指引。

## 目录

- [📚 学术基础](#-学术基础)
- [🏗️ 仓库结构](#-仓库结构)
- [🗂️ 模式速览](#-模式速览)
- [📚 基础模式](#-基础模式foundational-patterns)
- [🧠 进阶推理模式](#-进阶推理模式)
- [🛡️ 可靠性模式](#-可靠性模式)
- [🎯 编排模式](#-编排模式)
- [📊 可观测性模式](#-可观测性模式)
- [🧩 记忆模式](#-记忆模式)
- [🎓 学习模式](#-学习模式)
- [🚀 快速开始](#-快速开始)
- [🗺️ 模式选择指南](#-模式选择指南)
- [🎓 学习路径](#-学习路径)
- [🛠️ 技术栈](#-技术栈)
- [📖 资源](#-资源)
- [🏛️ 标准与合规](#-标准与合规)
- [📌 如何引用](#-如何引用)
- [🙏 致谢](#-致谢)

> **新增：Pi 实现分析**
>
> 本仓库现在包含面向实现层面的分析，说明这些模式如何映射到 [Pi](https://github.com/earendil-works/pi)。这些分析基于 Pi 的真实代码库，附带 package/module 引用、带行号的代码片段，以及对架构权衡或局限性的说明。
>
> 在各模式目录下查找 `pi.md`。这些文字保持保守的态度：如果 Pi 并没有有意义地实现某个模式，分析会直接说明这一点，而不是强行套用。
>
> 想看 28 个模式的整体覆盖情况？参见仓库根目录的 [**Pi 汇总（`pi.md`）**](./pi.md)：覆盖度热力图、按分数排序的总表，以及每个模式的一句话结论与最具代表性的代码引用。

**当前已提供的 Pi 分析：**
- 基础（Foundational）：[Prompt Chaining](./foundational_design_patterns/1_prompt_chain/pi.md)、[Routing](./foundational_design_patterns/2_routing/pi.md)、[Parallelization](./foundational_design_patterns/3_parallelization/pi.md)、[Reflection](./foundational_design_patterns/4_reflection/pi.md)、[Tool Use](./foundational_design_patterns/5_tool_use/pi.md)、[Planning](./foundational_design_patterns/6_planning/pi.md)、[Multi-Agent Collaboration](./foundational_design_patterns/7_multi_agent_collaboration/pi.md)、[ReAct](./foundational_design_patterns/8_react/pi.md)、[HITL](./foundational_design_patterns/10_hitl/pi.md)、[Structured Outputs](./foundational_design_patterns/11_structured_outputs/pi.md)、[Computer Use](./foundational_design_patterns/12_computer_use/pi.md)
- 推理（Reasoning）：[Tree of Thoughts](./reasoning/tree_of_thoughts/pi.md)、[Graph of Thoughts](./reasoning/graph_of_thoughts/pi.md)、[Deep Research](./reasoning/deep_research/pi.md)
- 可靠性（Reliability）：[Error Recovery](./reliability/error_recovery/pi.md)、[Guardrails](./reliability/guardrails/pi.md)
- 编排（Orchestration）：[Goal Management](./orchestration/goal_management/pi.md)、[Subagents](./orchestration/subagents/pi.md)、[Skills](./orchestration/skills/pi.md)、[Agent Communication](./orchestration/agent_communication/pi.md)、[MCP](./orchestration/mcp/pi.md)、[Prioritization](./orchestration/prioritization/pi.md)
- 可观测性（Observability）：[Evaluation & Monitoring](./observability/evaluation_monitoring/pi.md)、[Resource Optimization](./observability/resource_optimization/pi.md)
- 记忆（Memory）：[Memory Management](./memory/memory_management/pi.md)、[Context Management](./memory/context_management/pi.md)
- 学习（Learning）：[Adaptive Learning](./learning/adaptive_learning/pi.md)

---

## 📚 学术基础

本仓库实现的设计模式根植于经过同行评议的研究和行业最佳实践。关键的学术贡献包括：

### 核心研究论文

#### **推理与行动（Reasoning and Acting）**
- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)**（Yao 等，2022，ICLR 2023）
  奠基性论文，证明将推理轨迹（reasoning traces）与行动交织在一起所产生的协同效应，优于分别进行推理或行动的做法。直接在我们的 [ReAct 模式](./foundational_design_patterns/8_react/) 中实现。

#### **进阶推理框架**
- **[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)**（Yao 等，2023，NeurIPS 2023）
  把 Chain-of-Thought 推广到连贯推理路径的探索，支持自评估与回溯。在我们的 [Tree of Thoughts 模式](./reasoning/tree_of_thoughts/) 中实现。

- **[Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)**（Besta 等，2023）
  将推理扩展到任意图结构，支持思考合并和非层级化连接。在排序任务上质量提升 62%，同时成本降低 31%。在我们的 [Graph of Thoughts 模式](./reasoning/graph_of_thoughts/) 中实现。

#### **Agentic RAG**
- **[Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136)**（Singh 等，2025）
  一份系统性综述，讨论自主 AI 智能体如何通过反思、规划、工具使用和多智能体协作来管理检索策略——超越静态 RAG 的局限。指导我们的 [RAG](./foundational_design_patterns/9_rag/) 和 [Multi-Agent](./foundational_design_patterns/7_multi_agent_collaboration/) 模式。

### 研究影响

这些模式代表了如下演化：
- **Chain-of-Thought**（线性推理） → **Tree of Thoughts**（分支探索） → **Graph of Thoughts**（网络化推理）
- **静态 RAG**（固定检索） → **Agentic RAG**（自主、自适应的检索）
- **单智能体系统** → **多智能体协作**，具备专业化角色与通信协议

---

## 🏗️ 仓库结构
```
agentic_design_patterns/
├── foundational_design_patterns/
│   ├── 1_prompt_chain/         # 顺序式任务分解
│   ├── 2_routing/              # 智能查询路由
│   ├── 3_parallelization/      # 并发执行
│   ├── 4_reflection/           # 迭代式改进
│   ├── 5_tool_use/             # 外部系统集成
│   ├── 6_planning/             # 战略性任务规划
│   ├── 7_multi_agent_collaboration/  # 协同智能体
│   ├── 8_react/                # 推理与行动
│   ├── 9_rag/                  # 检索增强生成
│   ├── 10_hitl/                # 人类在环
│   ├── 11_structured_outputs/  # 受 schema 约束的输出
│   └── 12_computer_use/        # 浏览器 / UI 自动化
│
├── reasoning/                  # 进阶推理模式
│   ├── tree_of_thoughts/       # 系统化探索
│   ├── graph_of_thoughts/      # 非层级化推理
│   └── deep_research/          # 迭代研究循环
│
├── reliability/                # 安全与韧性
│   ├── error_recovery/         # 故障处理
│   └── guardrails/             # 安全约束
│
├── orchestration/              # 多智能体协调
│   ├── goal_management/        # 目标分解
│   ├── subagents/              # Orchestrator-Worker 拓扑
│   ├── skills/                 # 可加载的能力包
│   ├── agent_communication/    # 智能体间消息传递
│   ├── mcp/                    # Model Context Protocol
│   └── prioritization/         # 任务排序
│
├── observability/              # 监控与优化
│   ├── evaluation_monitoring/  # 指标与质量
│   └── resource_optimization/  # 成本与性能
│
├── memory/                     # 上下文与历史
│   ├── memory_management/      # 长期记忆
│   └── context_management/     # 上下文优化
│
├── learning/                   # 持续改进
│   └── adaptive_learning/      # 从反馈中学习
│
├── tests/                      # 仓库级可靠性 smoke 测试
├── .github/workflows/          # CI 工作流
├── repo_support.py             # 共享运行时 / 引导脚手架
├── .env                        # 环境变量
├── LICENSE                     # MIT 协议
└── README.md                   # 本文件
```

---

## 🗂️ 模式速览

按行索引、便于快速定位：每一行指向对应模式的目录；Pi 一列指向 `pi.md` 实现分析（如有）；TypeScript 一列标记是否在 `<pattern>/typescript/` 下提供 Bun/TypeScript 移植。

| # | 模式 | 分类 | Pi 分析 | TypeScript |
|---|---|---|---|---|
| 1 | [Prompt Chaining](./foundational_design_patterns/1_prompt_chain/) | 基础 | [pi](./foundational_design_patterns/1_prompt_chain/pi.md) | ✓ |
| 2 | [Routing](./foundational_design_patterns/2_routing/) | 基础 | [pi](./foundational_design_patterns/2_routing/pi.md) | ✓ |
| 3 | [Parallelization](./foundational_design_patterns/3_parallelization/) | 基础 | [pi](./foundational_design_patterns/3_parallelization/pi.md) | ✓ |
| 4 | [Reflection](./foundational_design_patterns/4_reflection/) | 基础 | [pi](./foundational_design_patterns/4_reflection/pi.md) | ✓ |
| 5 | [Tool Use](./foundational_design_patterns/5_tool_use/) | 基础 | [pi](./foundational_design_patterns/5_tool_use/pi.md) | ✓ |
| 6 | [Planning](./foundational_design_patterns/6_planning/) | 基础 | [pi](./foundational_design_patterns/6_planning/pi.md) | ✓ |
| 7 | [Multi-Agent Collaboration](./foundational_design_patterns/7_multi_agent_collaboration/) | 基础 | [pi](./foundational_design_patterns/7_multi_agent_collaboration/pi.md) | ✓ |
| 8 | [ReAct](./foundational_design_patterns/8_react/) | 基础 | [pi](./foundational_design_patterns/8_react/pi.md) | ✓ |
| 9 | [RAG](./foundational_design_patterns/9_rag/) | 基础 | — | ✓ |
| 10 | [Human-in-the-Loop (HITL)](./foundational_design_patterns/10_hitl/) | 基础 | [pi](./foundational_design_patterns/10_hitl/pi.md) | ✓ |
| 11 | [Structured Outputs](./foundational_design_patterns/11_structured_outputs/) | 基础 | [pi](./foundational_design_patterns/11_structured_outputs/pi.md) | ✓ |
| 12 | [Computer Use](./foundational_design_patterns/12_computer_use/) | 基础 | [pi](./foundational_design_patterns/12_computer_use/pi.md) | ✓ |
| 13 | [Tree of Thoughts](./reasoning/tree_of_thoughts/) | 推理 | [pi](./reasoning/tree_of_thoughts/pi.md) | — |
| 14 | [Graph of Thoughts](./reasoning/graph_of_thoughts/) | 推理 | [pi](./reasoning/graph_of_thoughts/pi.md) | — |
| 15 | [Deep Research](./reasoning/deep_research/) | 推理 | [pi](./reasoning/deep_research/pi.md) | — |
| 16 | [Error Recovery](./reliability/error_recovery/) | 可靠性 | [pi](./reliability/error_recovery/pi.md) | — |
| 17 | [Guardrails](./reliability/guardrails/) | 可靠性 | [pi](./reliability/guardrails/pi.md) | — |
| 18 | [Goal Management](./orchestration/goal_management/) | 编排 | [pi](./orchestration/goal_management/pi.md) | — |
| 19 | [Subagents](./orchestration/subagents/) | 编排 | [pi](./orchestration/subagents/pi.md) | — |
| 20 | [Skills](./orchestration/skills/) | 编排 | [pi](./orchestration/skills/pi.md) | — |
| 21 | [Agent Communication](./orchestration/agent_communication/) | 编排 | [pi](./orchestration/agent_communication/pi.md) | — |
| 22 | [MCP](./orchestration/mcp/) | 编排 | [pi](./orchestration/mcp/pi.md) | — |
| 23 | [Prioritization](./orchestration/prioritization/) | 编排 | [pi](./orchestration/prioritization/pi.md) | — |
| 24 | [Evaluation & Monitoring](./observability/evaluation_monitoring/) | 可观测性 | [pi](./observability/evaluation_monitoring/pi.md) | — |
| 25 | [Resource Optimization](./observability/resource_optimization/) | 可观测性 | [pi](./observability/resource_optimization/pi.md) | — |
| 26 | [Memory Management](./memory/memory_management/) | 记忆 | [pi](./memory/memory_management/pi.md) | — |
| 27 | [Context Management](./memory/context_management/) | 记忆 | [pi](./memory/context_management/pi.md) | — |
| 28 | [Adaptive Learning](./learning/adaptive_learning/) | 学习 | [pi](./learning/adaptive_learning/pi.md) | — |

---

## 📚 基础模式（Foundational Patterns）

### 1️⃣ [Prompt Chaining（提示链）](./foundational_design_patterns/1_prompt_chain/)
**将复杂任务拆分为顺序、可管理的步骤**
```python
# 把一个庞大的提示词转化为一条由专门提示组成的链
input → extract_data → transform → validate → final_output
```

**何时使用：**
- 多步转换（数据抽取 → 分析 → 格式化）
- 需要中间校验的任务
- 通过分解能受益的复杂工作流

**主要收益：**
- 🎯 通过聚焦化提示提升准确率
- 🔍 中间步骤可见，更易调试
- 🔄 跨工作流的可复用组件

[**📖 了解更多 →**](./foundational_design_patterns/1_prompt_chain/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/1_prompt_chain/pi.md)

---

### 2️⃣ [Routing（路由）](./foundational_design_patterns/2_routing/)
**智能地把查询导向专门的处理器**
```python
# 基于查询分类的动态路由
user_query → classifier → [technical_expert | sales_agent | support_bot]
```

**何时使用：**
- 多领域应用（客服、销售、技术）
- 专门化模型选择（快/便宜 vs. 慢/精准）
- 需要不同处理路径的意图驱动工作流

**主要收益：**
- 💰 成本优化（只在必要时使用昂贵模型）
- ⚡ 性能提升（把简单查询交给快处理器）
- 🎨 专门化处理（领域问题交给领域专家）

[**📖 了解更多 →**](./foundational_design_patterns/2_routing/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/2_routing/pi.md)

---

### 3️⃣ [Parallelization（并行化）](./foundational_design_patterns/3_parallelization/)
**并发执行独立操作，实现显著加速**
```python
# 顺序：15 秒                       # 并行：5 秒
task_a(5s) →                      task_a(5s) ↘
task_b(5s) →          vs.         task_b(5s) → combine → output
task_c(5s) → output               task_c(5s) ↗
```

**何时使用：**
- 多次 API 调用（搜索引擎、数据库、外部服务）
- 并行的数据处理（同时分析多个文档）
- 多源研究或内容生成

**主要收益：**
- ⚡ I/O 密集型任务可获 2–10 倍加速
- 📈 更好地利用资源
- 🚀 通过降低延迟改善用户体验

[**📖 了解更多 →**](./foundational_design_patterns/3_parallelization/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/3_parallelization/pi.md)

---

### 4️⃣ [Reflection（反思）](./foundational_design_patterns/4_reflection/)
**通过系统性的自我批评与改写来迭代提升输出质量**

四大核心 agentic 设计模式之一（Ng, 2024），反思使 AI 能够系统性地审视并改进自身输出。
```python
# 一次性生成：5/10 质量              # 加入反思：8.5/10 质量
input → generate → done            input → generate → critique →
                                          refine → critique → final
```

**何时使用：**
- 高风险内容（代码、法律文书、待发表文章）
- 复杂推理任务（逻辑谜题、战略规划）
- “够用就行”不够的高质量场景

**主要收益：**
- 🎯 质量分提升 50%–70%
- 🔍 系统化的错误检测与纠正
- 🧠 无需人工干预即可自我提升

**权衡：**
- ⚠️ Token 成本提升 3–5 倍
- ⏱️ 执行时间延长 4–8 倍

[**📖 了解更多 →**](./foundational_design_patterns/4_reflection/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/4_reflection/pi.md)

---

### 5️⃣ [Tool Use（工具使用）](./foundational_design_patterns/5_tool_use/)
**让 LLM 能与外部系统和 API 交互**

工具使用是把 LLM 的输出锚定到真实世界数据与行动的关键，也是现代智能体系统的基础能力（Ng, 2024；Yao 等，2022）。
```python
# 不用工具：只能依赖训练数据
# 使用工具：可获取实时数据并采取行动
user_query → LLM decides → call_weather_api(location) → integrate_result → response
```

**何时使用：**
- 实时数据检索（天气、股价、新闻）
- 访问私有/专有数据（数据库、CRM 系统）
- 精确计算或代码执行
- 外部行动（发邮件、更新记录、控制设备）

**主要收益：**
- 🌐 获取实时、动态信息
- 🎯 精确计算与数据校验
- 🔧 与现有企业系统集成
- 💰 降低 Token 成本（按需获取 vs. 嵌入提示词）

**权衡：**
- ⚠️ 每次工具调用都会增加延迟
- 🔒 安全考量（认证、校验）

[**📖 了解更多 →**](./foundational_design_patterns/5_tool_use/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/5_tool_use/pi.md)

---

### 6️⃣ [Planning（规划）](./foundational_design_patterns/6_planning/)
**把复杂目标拆解为结构化、可执行的行动计划**

智能体系统的基础能力之一（Ng, 2024），让 AI 能战略性地分解复杂目标，而不是被动反应。
```python
# 不规划：被动、不完整的执行
# 有规划：战略化拆解 + 系统化执行
complex_goal → analyze → decompose → plan_steps → execute_sequentially → final_result
```

**何时使用：**
- 需要编排的多步工作流（研究报告、数据管道）
- 操作之间相互依赖的任务
- 需要战略思考的复杂问题求解
- 工作流自动化（入职、采购、项目搭建）

**主要收益：**
- 🎯 面向复杂目标的结构化方法
- 🧠 战略思考而非反应式响应
- 🔄 通过动态重规划获得适应性
- 📊 执行策略的可观测性

**权衡：**
- ⚠️ 规划开销（Token 多 20%–40%，延迟 5–15 秒）
- 🛠️ 需要较成熟的状态管理

[**📖 了解更多 →**](./foundational_design_patterns/6_planning/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/6_planning/pi.md)

---

### 7️⃣ [Multi-Agent Collaboration（多智能体协作）](./foundational_design_patterns/7_multi_agent_collaboration/)
**协调多个专业化智能体共同求解复杂任务**

多智能体系统在 Agentic RAG 综述（Singh 等，2025）和生产部署中（LangChain，2024）都被重点提及，能够实现复杂的任务分工和专业化能力组合。
```python
# 像一个团队：角色专业化 + 协调通信
user_goal → manager/planner → [researcher | coder | designer | writer | reviewer] → synthesize → final_output
```

**何时使用：**
- 需要多样化专业能力的复杂任务（研究 + 写作 + QA）
- 阶段明确的工作流（研究 → 起草 → 编辑 → 打包）
- 工具专业化的角色（网页搜索、代码执行、图像生成）
- 质量要求高的流水线（批评者/审查者循环）

**主要收益：**
- 🧩 模块化：可以一个角色一个角色地构建与优化
- 🛡️ 鲁棒性：审查者捕捉错误 / 降低幻觉
- ⚡ 并行性：独立工作流拆分以提高速度
- ♻️ 复用：同一智能体可服务于多个产品

**常见协作模型：**
- 顺序交接（线性流水线）
- 主管 / 经理式编排（层级化）
- 并行工作流（结果合并）
- 辩论 / 共识（评估候选方案）
- 评论者—审查者（质量把关）
- 全连接 / 网络拓扑（探索性，可预测性较低）
- 自定义混合（贴合领域约束）

[**📖 了解更多 →**](./foundational_design_patterns/7_multi_agent_collaboration/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/7_multi_agent_collaboration/pi.md)

---

### 8️⃣ [ReAct（推理 + 行动）](./foundational_design_patterns/8_react/)（Yao 等，2022）
**将推理轨迹与工具执行交织在一起，实现自适应问题求解**

由 Yao 等（2022）在论文 "ReAct: Synergizing Reasoning and Acting in Language Models" 中首次提出。该模式表明，把推理轨迹与具体任务行动交织起来所产生的协同效应，优于把推理和行动视为彼此分离的能力。

```python
# 传统：直接行动，没有显式推理
user_query → tool_call → response

# ReAct：显式推理 + 受真实世界锚定的行动
user_query → Thought (reason) → Action (tool) → Observation (result) →
             Thought (adapt) → Action → Observation → Final Answer
```

**何时使用：**
- 需要信息查询与验证的多步研究
- 解决路径不预先确定的复杂问题求解
- 需要根据中间结果调整策略的任务
- 调试与探索性分析
- 需要可解释推理以提升可读性

**主要收益：**
- 🧠 显式的推理轨迹提升决策质量
- 🎯 基于事实的行动降低幻觉
- 🔄 根据观测结果动态调整
- 🔍 决策过程透明、可调试
- ✓ 自我纠错与错误恢复

**权衡：**
- ⚠️ 延迟更高（多轮推理 + 行动循环）
- 💰 Token 成本上升（推理轨迹 + 工具调用）
- 🔁 若没有迭代上限，可能陷入无效循环

[**📖 了解更多 →**](./foundational_design_patterns/8_react/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/8_react/pi.md)

---

### 9️⃣ [RAG（检索增强生成）](./foundational_design_patterns/9_rag/)
**用相关外部知识为 LLM 的回答提供锚点**

正如 Singh 等（2025）最新综述所述，这一方法在加入智能体能力后能让自主 AI 智能体通过反思、规划和工具使用动态地管理检索策略——突破静态 RAG 工作流的局限。

```python
# 不用 RAG：仅依赖训练数据
user_query → LLM → response (可能产生幻觉)

# 使用 RAG：以知识为锚的回答
user_query → retrieve_relevant_docs → augment_context → LLM → grounded_response
```

**何时使用：**
- 动态或频繁更新的信息（文档、产品目录）
- 私有 / 专有知识库
- 超出 LLM 训练范围的领域专业知识
- 通过事实锚定降低幻觉

**主要收益：**
- 📚 获取最新且专有的信息
- 🎯 通过锚定降低幻觉
- 💰 知识更新无需重新训练
- 🔍 来源可溯、过程透明

[**📖 了解更多 →**](./foundational_design_patterns/9_rag/README.md)

---

### 🔟 [Human-in-the-Loop（人类在环，HITL）](./foundational_design_patterns/10_hitl/)
**把人工监督与审批整合进 AI 工作流**

HITL 模式是智能体 AI 系统中的关键策略，刻意地把人类认知的独特优势——判断力、创造力、细腻的理解——与 AI 的计算能力和效率结合起来。这种战略性整合可确保 AI 在伦理边界内运行、遵守安全规程，并以最佳方式达成目标。

```python
# 不用 HITL：完全自动化
agent_action → execute → result

# 使用 HITL：人工检查点
agent_proposal → human_review → [approve|reject|modify] → execute → result
```

**何时使用：**
- 高风险决策（金融交易、法律行动、量刑）
- 质量要求高的内容（出版物、客户沟通）
- 合规与监管要求
- 需要细致判断的复杂场景
- 从人类专业知识中学习以持续改进
- 含糊性超出 LLM 可靠能力范围的任务

**关键方面：**
- **人工监督**：通过仪表盘/日志监控 AI 表现，确保符合规范
- **干预与纠正**：人工操作员纠正错误或在含糊场景下提供指导
- **学习反馈**：人类偏好用于指导智能体学习（如 RLHF）
- **决策增强**：AI 提供分析/建议，最终决策由人类做出
- **人-智能体协作**：发挥各自优势的合作式互动
- **升级策略**：定义何时把任务升级给人类的协议

**主要收益：**
- 🛡️ 在关键领域提供安全与风险缓解
- ✅ 质量保障与合规
- 🎓 从人类反馈中持续学习
- 🤝 通过透明度建立用户信任
- 🎯 在复杂场景中做出细致判断
- 🔄 持续改进的反馈循环

**实际应用：**
- **内容审核**：AI 大规模过滤，人工复核含糊案例
- **自动驾驶**：AI 处理大部分任务，复杂情况由人接管
- **金融反欺诈**：AI 标记可疑模式，人工分析师调查高风险预警
- **法律文书审阅**：AI 扫描/分类，律师审阅准确性与影响
- **客户支持**：聊天机器人处理常规问题，复杂/情绪化案例升级至人工
- **数据标注**：人类提供用于训练数据集的真实标签
- **生成式 AI 精修**：人类编辑审阅/打磨 LLM 输出，保证质量与品牌一致
- **自治网络**：AI 分析 KPI，人工审批关键网络变更

**权衡与注意事项：**
- ⚠️ **可扩展性限制**：人工监督无法处理百万级任务
- 👥 **专家依赖**：效果取决于熟练的领域专家
- 🔒 **隐私担忧**：敏感信息需要匿名化
- 💰 **成本考虑**：人工审核增加运营开销

**“Human-on-the-loop”变体：**
在该方式中，人类专家定义宏观策略，AI 处理即时行动以确保合规（例如：在人类设定的规则下进行自动交易、按经理设定的策略进行呼叫中心路由）。

[**📖 了解更多 →**](./foundational_design_patterns/10_hitl/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/10_hitl/pi.md)

---

### 1️⃣1️⃣ [Structured Outputs（结构化输出）](./foundational_design_patterns/11_structured_outputs/)
**强制 LLM 输出符合 schema，以便可靠地驱动下游自动化**
```python
# 朴素解析（脆弱）
text → prompt_json_request → parse_string_json → runtime_fail

# 结构化输出（可靠）
text → response_schema(Pydantic/JSON Schema) → validated_object → safe_automation
```

**主要收益：** Schema 保证、解析失败率更低、更安全的智能体循环

[**📖 了解更多 →**](./foundational_design_patterns/11_structured_outputs/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/11_structured_outputs/pi.md)

---

### 1️⃣2️⃣ [Computer Use（电脑操作）](./foundational_design_patterns/12_computer_use/)
**带有显式安全控制的浏览器 / UI 工作流自动化**
```python
# 面向 UI 任务的“观察 → 思考 → 行动”循环
screenshot/state → reasoning → ui_action(click/type/navigate) → observation → iterate
```

**主要收益：** 遗留系统自动化、UI QA 工作流、覆盖无 API 的任务

[**📖 了解更多 →**](./foundational_design_patterns/12_computer_use/README.md) · [**🔎 Pi 分析 →**](./foundational_design_patterns/12_computer_use/pi.md)

---

## 🧠 进阶推理模式

### [Tree of Thoughts](./reasoning/tree_of_thoughts/)（Yao 等，2023）
**系统化地探索多条推理路径**

发表于 NeurIPS 2023 的 Tree of Thoughts（ToT）把 Chain-of-Thought 提示推广为：让 LLM 探索多条推理路径、对各选择进行自评估，必要时回溯——为复杂任务实现有意识的“审慎”问题求解。

```python
# Chain of Thought：线性推理
input → step1 → step2 → step3 → answer

# Tree of Thoughts：分支式探索
input → [thought1, thought2, thought3] → evaluate → expand_best →
        [refined_thoughts] → evaluate → solution
```

**主要收益：** 通过系统化探索得到更优解、可回溯、决策树透明可见

[**📖 了解更多 →**](./reasoning/tree_of_thoughts/README.md) · [**🔎 Pi 分析 →**](./reasoning/tree_of_thoughts/pi.md)

---

### [Graph of Thoughts](./reasoning/graph_of_thoughts/)（Besta 等，2023）
**允许非层级化的思考连接与合并**

在 ToT 的基础上，Graph of Thoughts 把推理范式从层级树扩展到任意图，支持非线性的思考连接与聚合——在排序任务上获得 62% 的质量提升，同时降低 31% 的成本（Besta 等，2023）。

```python
# 思考可以引用、并基于“任何”其他思考构建（不仅是父子关系）
input → generate_perspectives → connect_thoughts → aggregate → synthesis
```

**主要收益：** 多视角分析、思考合并、灵活的推理路径

[**📖 了解更多 →**](./reasoning/graph_of_thoughts/README.md) · [**🔎 Pi 分析 →**](./reasoning/graph_of_thoughts/pi.md)

---

### [Deep Research（深度研究）](./reasoning/deep_research/)
**通过差距驱动的追问，运行迭代式研究循环**
```python
# 规划 → 搜索 → 阅读 → 反思 → 追问 → 综合
question → sub_queries → retrieve_sources → identify_gaps → refine_queries → cited_output
```

**主要收益：** 更全面的覆盖、更少的盲点、更强的引用质量

[**📖 了解更多 →**](./reasoning/deep_research/README.md) · [**🔎 Pi 分析 →**](./reasoning/deep_research/pi.md)

---

## 🛡️ 可靠性模式

### [Error Recovery（错误恢复）](./reliability/error_recovery/)
**优雅地处理失败并自我纠正**
```python
# 检测 → 诊断 → 恢复 → 校验
operation → [success | failure] → classify_error → [retry | fallback | self_correct] → verify
```

**主要收益：** 韧性、平滑降级、自动自愈、减少停机

[**📖 了解更多 →**](./reliability/error_recovery/README.md) · [**🔎 Pi 分析 →**](./reliability/error_recovery/pi.md)

---

### [Guardrails（护栏）](./reliability/guardrails/)
**强制执行安全约束与合规要求**
```python
# 多层校验
input → validate → process → validate_output → [pass | block] → log
```

**主要收益：** 安全保障、合规性、品牌保护、风险降低

[**📖 了解更多 →**](./reliability/guardrails/README.md) · [**🔎 Pi 分析 →**](./reliability/guardrails/pi.md)

---

## 🎯 编排模式

### [Goal Management（目标管理）](./orchestration/goal_management/)
**分解并追踪复杂目标**
```python
# 层级化分解，伴随进度追踪
complex_goal → decompose → [subgoal1, subgoal2, subgoal3] →
              track_dependencies → execute → monitor → replan
```

**主要收益：** 结构化执行、进度可见、自适应规划、资源优化

[**📖 了解更多 →**](./orchestration/goal_management/README.md) · [**🔎 Pi 分析 →**](./orchestration/goal_management/pi.md)

---

### [Subagents（Orchestrator–Worker 子智能体）](./orchestration/subagents/)
**生成上下文隔离、有结构化摘要的聚焦子智能体**
```python
lead_agent → decompose_task → spawn_workers_parallel → structured_summaries → synthesize
```

**主要收益：** 上下文隔离、并行吞吐、综合更清晰

[**📖 了解更多 →**](./orchestration/subagents/README.md) · [**🔎 Pi 分析 →**](./orchestration/subagents/pi.md)

---

### [Skills（技能）](./orchestration/skills/)
**通过“先看元数据”的发现机制按需加载能力包**
```python
skill_catalog(metadata) → select_relevant_skill → load_SKILL_body → execute
```

**主要收益：** 工具扩展能力超过扁平列表的极限、降低提示负担、能力模块化

[**📖 了解更多 →**](./orchestration/skills/README.md) · [**🔎 Pi 分析 →**](./orchestration/skills/pi.md)

---

### [Agent Communication（A2A 智能体通信）](./orchestration/agent_communication/)
**让智能体通过消息传递相互协调**
```python
# 直接消息、发布-订阅、协商协议
agent1 → message → agent2 → response → agent1
```

**主要收益：** 松耦合、动态发现、可扩展性、容错性

[**📖 了解更多 →**](./orchestration/agent_communication/README.md) · [**🔎 Pi 分析 →**](./orchestration/agent_communication/pi.md)

---

### [Model Context Protocol（MCP）](./orchestration/mcp/)
**标准化的工具与资源集成**
```python
# AI 的“USB”：工具/数据的标准接口
LLM → discover_tools → invoke_tool(params) → receive_result → integrate
```

**主要收益：** 标准化、可复用、互操作性、可组合

[**📖 了解更多 →**](./orchestration/mcp/README.md) · [**🔎 Pi 分析 →**](./orchestration/mcp/pi.md)

---

### [Prioritization（优先级排序）](./orchestration/prioritization/)
**优化任务顺序与资源分配**
```python
# 多维度评分 + 动态再平衡
tasks → score(urgency, impact, effort) → rank → schedule → execute
```

**主要收益：** 资源优化、按期完成、公平性、效率

[**📖 了解更多 →**](./orchestration/prioritization/README.md) · [**🔎 Pi 分析 →**](./orchestration/prioritization/pi.md)

---

## 📊 可观测性模式

### [Evaluation & Monitoring（评估与监控）](./observability/evaluation_monitoring/)
**跟踪性能与质量指标**
```python
# 定量 + 定性指标
operation → collect_metrics → evaluate_quality → aggregate → alert → visualize
```

**主要收益：** 可观测、早期发现、数据驱动决策、持续改进

[**📖 了解更多 →**](./observability/evaluation_monitoring/README.md) · [**🔎 Pi 分析 →**](./observability/evaluation_monitoring/pi.md)

---

### [Resource Optimization（资源优化）](./observability/resource_optimization/)
**降低成本、提升性能**
```python
# 缓存、批处理、模型路由
request → [cache_hit | cache_miss] → [cheap_model | expensive_model] → optimize
```

**主要收益：** 节省 65%–80% 成本、响应更快、体验更好

[**📖 了解更多 →**](./observability/resource_optimization/README.md) · [**🔎 Pi 分析 →**](./observability/resource_optimization/pi.md)

---

## 🧩 记忆模式

### [Memory Management（记忆管理）](./memory/memory_management/)
**维护对话历史与长期记忆**
```python
# 缓冲 + 语义记忆
interaction → store → [buffer_memory | vector_memory] → retrieve_relevant → use
```

**主要收益：** 上下文保留、个性化、从历史中学习

[**📖 了解更多 →**](./memory/memory_management/README.md) · [**🔎 Pi 分析 →**](./memory/memory_management/pi.md)

---

### [Context Management（上下文管理）](./memory/context_management/)
**优化上下文窗口使用**
```python
# 动态选择与压缩
content → score_relevance → compress → fit_window → optimize
```

**主要收益：** 节省 70%–90% 成本、回答更聚焦、性能更好

[**📖 了解更多 →**](./memory/context_management/README.md) · [**🔎 Pi 分析 →**](./memory/context_management/pi.md)

---

## 🎓 学习模式

### [Adaptive Learning（自适应学习）](./learning/adaptive_learning/)
**通过反馈与持续学习不断改进**
```python
# 从结果中学习
action → feedback → analyze_patterns → adapt_strategy → improve
```

**主要收益：** 持续改进、个性化、领域自适应

[**📖 了解更多 →**](./learning/adaptive_learning/README.md) · [**🔎 Pi 分析 →**](./learning/adaptive_learning/pi.md)

---

## 🚀 快速开始

### 先决条件
```bash
# Python 3.11 或更高
python --version

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装
```bash
# 克隆仓库
git clone https://github.com/gtesei/agentic_design_patterns.git
cd agentic_design_patterns

# 设置共享环境
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 仓库运行时说明

- 仓库要求 **Python 3.11+**。
- 每个模式目录都有自己的 `pyproject.toml`，请进入你要运行的模式目录后执行 `uv sync`。
- 示例脚本现在使用 `repo_support.py` 中的共享引导脚手架来：
  - 定位仓库根目录
  - 加载根目录下的 `.env`
  - 让仓库可以从任意模式目录中被导入
- 如果希望覆盖默认示例模型，请在环境中设置 `OPENAI_MODEL`：

```bash
export OPENAI_MODEL=gpt-4o-mini
```

- 如果你处于企业 SSL 拦截代理后方，SSL 旁路现在为 **opt-in**：

```bash
export AGENTIC_DISABLE_SSL=1
```

### 运行你的第一个模式

Python 仍是首要支持的语言轨道。所有当前的基础模式也提供位于 `<pattern>/typescript/` 下的 Bun/TypeScript 移植。

```bash
# Python
cd foundational_design_patterns/3_parallelization
[uv sync]
uv run python src/parallelization.py

# TypeScript
cd foundational_design_patterns/3_parallelization/typescript
[bun install]
bash run.sh
```

当前同时提供 Python 与 TypeScript 实现的基础模式位于：

- `foundational_design_patterns/1_prompt_chain`
- `foundational_design_patterns/2_routing`
- `foundational_design_patterns/3_parallelization`
- `foundational_design_patterns/4_reflection`
- `foundational_design_patterns/5_tool_use`
- `foundational_design_patterns/6_planning`
- `foundational_design_patterns/7_multi_agent_collaboration`
- `foundational_design_patterns/8_react`
- `foundational_design_patterns/9_rag`
- `foundational_design_patterns/10_hitl`
- `foundational_design_patterns/11_structured_outputs`
- `foundational_design_patterns/12_computer_use`

TypeScript 工作区的约定、覆盖范围和运行时细节，参见 [typescript_base/TYPESCRIPT.md](./typescript_base/TYPESCRIPT.md)。

### 可靠性 / Smoke 测试

请在仓库根目录运行：

```bash
# Python 基础模式离线 smoke
bash scripts/run_demos_smoke.sh --mode basic

# TypeScript 基础模式离线 smoke
bash scripts/run_demos_smoke_typescript.sh --mode basic
```

### CI

GitHub Actions 会在推送和 PR 上执行可靠性闸门：

- `.github/workflows/reliability-gate.yml`
- 通过 `unittest` 执行 Python 共享运行时 smoke
- 通过 `bun --bun tsc --noEmit` 对基础 TypeScript 进行类型检查
- 通过 `scripts/run_demos_smoke_typescript.sh --mode basic` 执行基础 TypeScript 离线 smoke

---

## 🗺️ 模式选择指南

### 按需求选择模式：

**需要速度？** → **Routing** + **Parallelization** + **Resource Optimization**（缓存、批处理）

**需要质量？** → **Reflection** + **RAG**（基于知识的锚定）+ **Evaluation & Monitoring**

**需要成本优化？** → **Routing** + **Resource Optimization**（节省 65%–80%）+ **Context Management**

**既要速度又要质量？** → **Parallelization** + **Prompt Chaining** + **RAG**

**复杂的多步工作流？** → **Prompt Chaining** + **Planning** + **Goal Management**

**独立的并发任务？** → **Parallelization** 能带来巨大加速

**高风险输出？** → **Reflection** + **HITL**（人工审批）+ **Guardrails**（安全）

**外部系统集成？** → **Tool Use** + **MCP**（标准化协议）

**多步自动化？** → **Planning** + **Goal Management** + **Agent Communication**

**多角色协同？** → **Multi-Agent Collaboration** + **Agent Communication**（A2A）

**探索性的多步任务？** → **ReAct**（推理 + 行动）或 **Tree of Thoughts**（探索）

**需要透明的决策过程？** → **ReAct**（显式推理）+ **Evaluation & Monitoring**

**需要严格的机器可读输出？** → **Structured Outputs** + **Guardrails**

**需要 UI / 浏览器自动化？** → **Computer Use** + **HITL**

**需要带引用的迭代综合？** → **Deep Research** + **RAG**

**需要可扩展、能力丰富的智能体？** → **Subagents** + **Skills**

**需要基于知识的回答？** → **RAG** 在生成前检索相关文档

**复杂的推理任务？** → **Tree of Thoughts**（系统化）或 **Graph of Thoughts**（多视角）

**生产可靠性？** → **Error Recovery** + **Guardrails** + **Evaluation & Monitoring**

**长对话？** → **Memory Management** + **Context Management**（优化窗口）

**持续改进？** → **Adaptive Learning** + **Evaluation & Monitoring**（反馈回路）

**资源受限？** → **Prioritization** + **Resource Optimization** + **Context Management**


---

## 🎓 学习路径

### 初学者 → 进阶 → 高级 → 专家

**第 1 阶段：基础（从这里开始）**
1. [Prompt Chaining](./foundational_design_patterns/1_prompt_chain/) - 一切的起点
2. [Routing](./foundational_design_patterns/2_routing/) - 学会优化模型选择
3. [Parallelization](./foundational_design_patterns/3_parallelization/) - 扩展你的应用
4. [Reflection](./foundational_design_patterns/4_reflection/) - 精通质量优化
5. [Tool Use](./foundational_design_patterns/5_tool_use/) - 接入外部系统

**第 2 阶段：核心模式**
6. [RAG](./foundational_design_patterns/9_rag/) - 基于知识的回答
7. [ReAct](./foundational_design_patterns/8_react/) - 推理 + 行动
8. [Planning](./foundational_design_patterns/6_planning/) - 战略性分解
9. [HITL](./foundational_design_patterns/10_hitl/) - 人工监督
10. [Multi-Agent](./foundational_design_patterns/7_multi_agent_collaboration/) - 智能体协调

**第 3 阶段：进阶推理**
11. [Tree of Thoughts](./reasoning/tree_of_thoughts/) - 系统化探索
12. [Graph of Thoughts](./reasoning/graph_of_thoughts/) - 多视角推理

**第 4 阶段：生产化模式**
13. [Error Recovery](./reliability/error_recovery/) - 韧性
14. [Guardrails](./reliability/guardrails/) - 安全
15. [Evaluation & Monitoring](./observability/evaluation_monitoring/) - 指标
16. [Resource Optimization](./observability/resource_optimization/) - 成本/性能

**第 5 阶段：编排与记忆**
17. [Goal Management](./orchestration/goal_management/) - 目标追踪
18. [Agent Communication](./orchestration/agent_communication/) - 消息传递
19. [MCP](./orchestration/mcp/) - 标准化集成
20. [Prioritization](./orchestration/prioritization/) - 任务排序
21. [Memory Management](./memory/memory_management/) - 上下文保留
22. [Context Management](./memory/context_management/) - 优化

**第 6 阶段：持续改进**
23. [Adaptive Learning](./learning/adaptive_learning/) - 从反馈中学习
24. [Structured Outputs](./foundational_design_patterns/11_structured_outputs/) - Schema 可靠性
25. [Computer Use](./foundational_design_patterns/12_computer_use/) - 浏览器/UI 自动化
26. [Subagents](./orchestration/subagents/) - Orchestrator–Worker 拓扑
27. [Skills](./orchestration/skills/) - 能力包
28. [Deep Research](./reasoning/deep_research/) - 迭代式研究循环

每个模式都建立在前面模式的概念之上。请从第 1 阶段开始，然后按需探索其他阶段。

---

## 🛠️ 技术栈

### 核心框架
- **[LangChain](https://python.langchain.com/)** —— LLM 应用的综合框架
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** —— 有状态工作流与多智能体编排
- **[LangSmith](https://smith.langchain.com/)** —— LLM 应用的监控与评估

### 模型与 API
- **[OpenAI Models](https://openai.com/)** —— 示例中始终使用的主要提供商，通过环境变量配置
- **[Anthropic Claude](https://anthropic.com/)** —— 另一类先进模型家族，长上下文支持出色
- **[其他 LLM 提供商](https://python.langchain.com/docs/integrations/llms/)** —— 通过 LangChain 抽象层完全兼容

### 开发工具
- **[Pydantic](https://docs.pydantic.dev/)** —— 数据校验与结构化输出
- **[Python 3.11+](https://www.python.org/)** —— 现代 Python 特性（`match/case`、typing）
- **[uv](https://github.com/astral-sh/uv)** —— 极速 Python 包管理器

### 可观测性与评估
- **[W&B Weave](https://wandb.ai/site/weave/)** —— 智能体评估与监控
- **[LangSmith](https://smith.langchain.com/)** —— 追踪与调试

---


## 📖 资源

### 🎓 学术论文与综述

**推理与规划：**
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)（Yao 等，2022）- ICLR 2023
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)（Yao 等，2023）- NeurIPS 2023
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)（Besta 等，2023）

**检索增强生成：**
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136)（Singh 等，2025）
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)（Lewis 等，2020）

### 📚 图书

- **[Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems](https://link.springer.com/book/10.1007/978-3-031-87617-1)** —— Antonio Gullí（Springer Nature，2024）—— 本仓库的主要灵感来源
- **[Building LLM Powered Applications](https://www.oreilly.com/library/view/building-llm-powered/9781835462317/)** —— Valentina Alto（Packt/O'Reilly，2024）
- **[Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large/9781098150952/)** —— Jay Alammar & Maarten Grootendorst（O'Reilly，2024）

### 🎓 课程与教育内容

**基础课程：**
- **[Agentic AI with Andrew Ng](https://www.deeplearning.ai/courses/agentic-ai/)**（DeepLearning.AI，2024）—— 涵盖反思、工具使用、规划与多智能体协作

**框架相关：**
- [LangChain Academy](https://academy.langchain.com/) —— 官方 LangChain 课程
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) —— 有状态智能体工作流
- [OpenAI Cookbook](https://cookbook.openai.com/) —— 函数调用与智能体模式
- [Anthropic Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)

### 🏭 行业文档与指南

**官方框架文档：**
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/stable/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Agents Platform](https://platform.openai.com/docs/guides/agents)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

**生产实践：**
- [LangChain: Top 5 LangGraph Agents in Production 2024](https://www.blog.langchain.com/top-5-langgraph-agents-in-production-2024/) —— 真实世界部署
- [Weights & Biases: Agentic RAG Guide](https://wandb.ai/byyoung3/Generative-AI/reports/Agentic-RAG-Enhancing-retrieval-augmented-generation-with-AI-agents--VmlldzoxMTcyNjQ5Ng)

### 🌐 社区资源

**精选合集：**
- [Awesome-LangGraph](https://github.com/von-development/awesome-LangGraph) —— LangGraph 生态全景索引
- [Prompt Engineering Guide](https://www.promptingguide.ai/) —— 涵盖最新论文与技巧的综合指南
- [Learn Prompting](https://learnprompting.org/) —— 免费生成式 AI 指南

**相关项目：**
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AG2（前身 AutoGen）](https://github.com/ag2ai/ag2)

### 🔬 研究合集

- [Papers with Code: Agents](https://paperswithcode.com/task/agents) —— 含实现的最新研究
- [arXiv: Artificial Intelligence](https://arxiv.org/list/cs.AI/recent) —— AI 最新论文
- [Hugging Face Papers](https://huggingface.co/papers) —— 热门 ML 研究

---

## 🏛️ 标准与合规

### NIST AI 风险管理框架

部署智能体 AI 系统的组织应考虑 NIST AI 风险管理框架及其相关指南：

#### **核心框架**
- **[NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework)**（2023 年 1 月）
  自愿性框架，基于 Govern、Map、Measure、Manage 四大核心功能来管理 AI 风险。

#### **生成式 AI 专项**
- **[NIST AI RMF: Generative AI Profile (NIST.AI.600-1)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)**（2024 年 7 月）
  针对生成式 AI 的特有风险，包括治理、内容溯源、部署前测试与事件披露。包含 400+ 项缓解措施的目录。

### 合规映射

| NIST AI RMF 功能 | 相关模式 |
|---------------------|-------------------|
| **Govern**（治理） | [Human-in-the-Loop](./foundational_design_patterns/10_hitl/)、[Guardrails](./reliability/guardrails/) |
| **Map**（映射） | [Planning](./foundational_design_patterns/6_planning/)、[Goal Management](./orchestration/goal_management/) |
| **Measure**（度量） | [Evaluation & Monitoring](./observability/evaluation_monitoring/)、[Adaptive Learning](./learning/adaptive_learning/) |
| **Manage**（管理） | [Error Recovery](./reliability/error_recovery/)、[Guardrails](./reliability/guardrails/) |

### 智能体系统的关键关注点

- **透明性：** ReAct 模式提供可审计的显式推理轨迹
- **人工监督：** HITL 模式启用审批流程
- **安全约束：** Guardrails 模式强制执行合规边界
- **评估：** 监控模式跟踪质量与偏差指标
- **错误恢复：** 优雅降级与事件响应

### 其他资源

- [AI Executive Order 14110](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/)
- [EU AI Act](https://artificialintelligenceact.eu/)
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) —— AI 管理体系标准

---

## 📄 许可证

本项目采用 MIT 许可证 —— 详见 [LICENSE](./LICENSE) 文件。

---

## 📌 如何引用

如需在学术或专业工作中引用本目录，请使用：

```bibtex
@misc{tesei_agentic_design_patterns,
  author       = {Tesei, Gino},
  title        = {Agentic Design Patterns: A Hands-On Catalog for Building Intelligent Systems},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/gtesei/agentic_design_patterns}},
  note         = {MIT License}
}
```

---

## 🙏 致谢

本仓库的结构与方法受以下工作的启发：

### 主要参考

> **Gullí, Antonio**, *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*, Springer Nature Switzerland, 2024.

> **Ng, Andrew**, *Agentic AI*, DeepLearning.AI, 2024.

### 学术基础

我们由衷感谢支撑这些模式的研究贡献：

- **Yao, Shunyu 等** —— ReAct 与 Tree of Thoughts 框架
- **Singh, Aditi 等** —— Agentic RAG 综述与分类学
- **Besta, Maciej 等** —— Graph of Thoughts 方法论
- **Lewis, Patrick 等** —— RAG 的奠基性研究

### 社区与工具

特别感谢：
- **LangChain 与 LangGraph 团队** 构建了生产级智能体框架
- **开源 AI 社区** 推动了技术前沿
- **NIST** 提供了关于可信 AI 开发的指南
- **[Claude Code](https://claude.ai/)** 在本仓库实现的开发与打磨中提供了协助
- 所有帮助改进这些模式的**贡献者**

---

## ⭐ Star 历史

如果你觉得本仓库有帮助，欢迎 Star！这能帮助更多人发现这些模式。

[![Star History Chart](https://api.star-history.com/svg?repos=gtesei/agentic_design_patterns&type=Date)](https://star-history.com/#gtesei/agentic_design_patterns&Date)
---

<div align="center">

**用 ❤️ 为 AI 开发者社区打造**

[⬆ 返回顶部](#-智能体设计模式agentic-design-patterns)

</div>
