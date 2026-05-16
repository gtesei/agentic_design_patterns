# 🤖 Agentic Design Patterns

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![GitHub stars](https://img.shields.io/github/stars/gtesei/agentic_design_patterns?style=social)](https://github.com/gtesei/agentic_design_patterns/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/gtesei/agentic_design_patterns?style=social)](https://github.com/gtesei/agentic_design_patterns/network)
[![GitHub issues](https://img.shields.io/github/issues/gtesei/agentic_design_patterns)](https://github.com/gtesei/agentic_design_patterns/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/gtesei/agentic_design_patterns)](https://github.com/gtesei/agentic_design_patterns/commits)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Transform your AI applications from simple prompts to sophisticated intelligent systems.**

A comprehensive, hands-on collection of design patterns for building robust agentic AI systems. Each pattern is implemented with real-world examples and detailed documentation to help you architect scalable, maintainable AI applications.

---

## 📚 Academic Foundations

This repository implements design patterns grounded in peer-reviewed research and industry best practices. Key academic contributions include:

### Core Research Papers

#### **Reasoning and Acting**
- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** (Yao et al., 2022, ICLR 2023)
  Foundational paper demonstrating that interleaving reasoning traces with actions creates synergy superior to separate reasoning or acting approaches. Directly implemented in our [ReAct pattern](./foundational_design_patterns/8_react/).

#### **Advanced Reasoning Frameworks**
- **[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)** (Yao et al., 2023, NeurIPS 2023)
  Generalizes Chain-of-Thought by enabling exploration over coherent reasoning paths with self-evaluation and backtracking. Implemented in our [Tree of Thoughts pattern](./reasoning/tree_of_thoughts/).

- **[Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)** (Besta et al., 2023)
  Extends reasoning to arbitrary graph structures, enabling thought merging and non-hierarchical connections. Achieves 62% quality improvement with 31% cost reduction. Implemented in our [Graph of Thoughts pattern](./reasoning/graph_of_thoughts/).

#### **Agentic RAG**
- **[Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136)** (Singh et al., 2025)
  Comprehensive survey of autonomous AI agents managing retrieval strategies through reflection, planning, tool use, and multi-agent collaboration—transcending static RAG limitations. Informs our [RAG](./foundational_design_patterns/9_rag/) and [Multi-Agent](./foundational_design_patterns/7_multi_agent_collaboration/) patterns.

### Research Impact

These patterns represent the evolution from:
- **Chain-of-Thought** (linear reasoning) → **Tree of Thoughts** (branching exploration) → **Graph of Thoughts** (networked reasoning)
- **Static RAG** (fixed retrieval) → **Agentic RAG** (autonomous, adaptive retrieval)
- **Single-agent systems** → **Multi-agent collaboration** with specialized roles and communication protocols

---

## 📚 Foundational Patterns

### 1️⃣ [Prompt Chaining](./foundational_design_patterns/1_prompt_chain/)
**Break complex tasks into sequential, manageable steps**
```python
# Transform a monolithic prompt into a chain of specialized prompts
input → extract_data → transform → validate → final_output
```

**When to use:**
- Multi-step transformations (data extraction → analysis → formatting)
- Tasks requiring intermediate validation
- Complex workflows that benefit from decomposition

**Key benefits:**
- 🎯 Better accuracy through focused prompts
- 🔍 Easier debugging with visible intermediate steps
- 🔄 Reusable components across workflows

[**📖 Learn More →**](./foundational_design_patterns/1_prompt_chain/README.md)

---

### 2️⃣ [Routing](./foundational_design_patterns/2_routing/)
**Intelligently direct queries to specialized handlers**
```python
# Dynamic routing based on query classification
user_query → classifier → [technical_expert | sales_agent | support_bot]
```

**When to use:**
- Multi-domain applications (support, sales, technical)
- Specialized model selection (fast/cheap vs. slow/accurate)
- Intent-based workflows requiring different processing paths

**Key benefits:**
- 💰 Cost optimization (use expensive models only when needed)
- ⚡ Performance gains (route simple queries to fast handlers)
- 🎨 Specialized handling (domain experts for domain queries)

[**📖 Learn More →**](./foundational_design_patterns/2_routing/README.md)

---

### 3️⃣ [Parallelization](./foundational_design_patterns/3_parallelization/)
**Execute independent operations simultaneously for dramatic speedups**
```python
# Sequential: 15 seconds          # Parallel: 5 seconds
task_a(5s) →                      task_a(5s) ↘
task_b(5s) →          vs.         task_b(5s) → combine → output
task_c(5s) → output               task_c(5s) ↗
```

**When to use:**
- Multiple API calls (search engines, databases, external services)
- Parallel data processing (analyze multiple documents)
- Multi-source research or content generation

**Key benefits:**
- ⚡ 2-10x faster execution for I/O-bound tasks
- 📈 Better resource utilization
- 🚀 Improved user experience through reduced latency

[**📖 Learn More →**](./foundational_design_patterns/3_parallelization/README.md)

---

### 4️⃣ [Reflection](./foundational_design_patterns/4_reflection/)
**Iteratively improve outputs through systematic critique and refinement**

One of the four core agentic design patterns (Ng, 2024), reflection enables AI to systematically critique and improve its own outputs.
```python
# Single-shot: 5/10 quality        # With reflection: 8.5/10 quality
input → generate → done            input → generate → critique → 
                                          refine → critique → final
```

**When to use:**
- High-stakes content (code, legal docs, published articles)
- Complex reasoning tasks (logic puzzles, strategic planning)
- Quality-critical applications where "good enough" isn't enough

**Key benefits:**
- 🎯 50-70% higher quality scores
- 🔍 Systematic error detection and correction
- 🧠 Self-improving outputs without human intervention

**Trade-offs:**
- ⚠️ 3-5x higher token costs
- ⏱️ 4-8x longer execution time

[**📖 Learn More →**](./foundational_design_patterns/4_reflection/README.md)

---

### 5️⃣ [Tool Use](./foundational_design_patterns/5_tool_use/)
**Enable LLMs to interact with external systems and APIs**

Essential for grounding LLM outputs in real-world data and actions, tool use is a foundational capability of modern agentic systems (Ng, 2024; Yao et al., 2022).
```python
# Without tools: Limited to training data
# With tools: Access real-time data and take actions
user_query → LLM decides → call_weather_api(location) → integrate_result → response
```

**When to use:**
- Real-time data retrieval (weather, stock prices, news)
- Private/proprietary data access (databases, CRM systems)
- Precise calculations or code execution
- External actions (send emails, update records, control devices)

**Key benefits:**
- 🌐 Access to live, dynamic information
- 🎯 Precise calculations and data validation
- 🔧 Integration with existing enterprise systems
- 💰 Reduced token costs (fetch vs. embed in prompts)

**Trade-offs:**
- ⚠️ Added latency per tool call
- 🔒 Security considerations (authentication, validation)

[**📖 Learn More →**](./foundational_design_patterns/5_tool_use/README.md)

---

### 6️⃣ [Planning](./foundational_design_patterns/6_planning/)
**Decompose complex goals into structured, executable action plans**

A fundamental capability for agentic systems (Ng, 2024), enabling AI to decompose complex objectives strategically rather than responding reactively.
```python
# Without planning: Reactive, incomplete execution
# With planning: Strategic breakdown and systematic execution
complex_goal → analyze → decompose → plan_steps → execute_sequentially → final_result
```

**When to use:**
- Multi-step workflows requiring orchestration (research reports, data pipelines)
- Tasks with interdependent operations
- Complex problem-solving requiring strategic thinking
- Workflow automation (onboarding, procurement, project setup)

**Key benefits:**
- 🎯 Structured approach to complex objectives
- 🧠 Strategic thinking vs. reactive responses
- 🔄 Adaptability through dynamic replanning
- 📊 Transparency into execution strategy

**Trade-offs:**
- ⚠️ Planning overhead (+20-40% tokens, 5-15s latency)
- 🛠️ Requires sophisticated state management

[**📖 Learn More →**](./foundational_design_patterns/6_planning/README.md)

---

### 7️⃣ [Multi-Agent Collaboration](./foundational_design_patterns/7_multi_agent_collaboration/)
**Coordinate multiple specialized agents to solve complex tasks**

Multi-agent systems, highlighted in both the Agentic RAG survey (Singh et al., 2025) and production deployments (LangChain, 2024), enable sophisticated task distribution and specialized expertise.
```python
# Agents as a team: specialize roles + coordinate communication
user_goal → manager/planner → [researcher | coder | designer | writer | reviewer] → synthesize → final_output
```

**When to use:**
- Complex tasks requiring diverse expertise (research + writing + QA)
- Workflows with distinct stages (research → draft → edit → package)
- Tool-specialized roles (web search, code execution, image generation)
- Quality-critical pipelines (critic/reviewer loops)

**Key benefits:**
- 🧩 Modularity: build and improve one role at a time
- 🛡️ Robustness: reviewers catch errors / reduce hallucinations
- ⚡ Parallelism: split independent workstreams for speed
- ♻️ Reuse: agents can be reused across multiple products

**Common collaboration models:**
- Sequential handoffs (linear pipeline)
- Supervisor/manager orchestration (hierarchical)
- Parallel workstreams (merge results)
- Debate/consensus (evaluate options)
- Critic–reviewer (quality enforcement)
- Network/all-to-all (exploratory, less predictable)
- Custom hybrids (fit domain constraints)

[**📖 Learn More →**](./foundational_design_patterns/7_multi_agent_collaboration/README.md)

---

### 8️⃣ [ReAct (Reasoning and Acting)](./foundational_design_patterns/8_react/) (Yao et al., 2022)
**Interleave reasoning traces with tool execution for adaptive problem-solving**

Originally introduced by Yao et al. (2022) in the paper "ReAct: Synergizing Reasoning and Acting in Language Models," this pattern demonstrates that interleaving reasoning traces with task-specific actions creates superior synergy compared to treating reasoning and acting as separate capabilities.

```python
# Traditional: Direct action without explicit reasoning
user_query → tool_call → response

# ReAct: Explicit reasoning + grounded actions
user_query → Thought (reason) → Action (tool) → Observation (result) →
             Thought (adapt) → Action → Observation → Final Answer
```

**When to use:**
- Multi-step research requiring information lookup and verification
- Complex problem-solving where the solution path isn't predetermined
- Tasks requiring adaptation based on intermediate results
- Debugging and exploratory analysis
- Need transparent reasoning for interpretability

**Key benefits:**
- 🧠 Explicit reasoning traces improve decision quality
- 🎯 Grounded actions reduce hallucinations
- 🔄 Dynamic adaptation based on observations
- 🔍 Transparent and debuggable decision-making
- ✓ Self-correction and error recovery

**Trade-offs:**
- ⚠️ Higher latency (multiple reasoning + action cycles)
- 💰 Increased token costs (reasoning traces + tool calls)
- 🔁 Risk of unproductive loops without iteration limits

[**📖 Learn More →**](./foundational_design_patterns/8_react/README.md)

---

### 9️⃣ [RAG (Retrieval-Augmented Generation)](./foundational_design_patterns/9_rag/)
**Ground LLM responses with relevant external knowledge**

This approach, enhanced by agentic capabilities as described in the recent survey by Singh et al. (2025), enables autonomous AI agents to dynamically manage retrieval strategies through reflection, planning, and tool use—transcending the limitations of static RAG workflows.

```python
# Without RAG: Limited to training data
user_query → LLM → response (may hallucinate)

# With RAG: Knowledge-grounded responses
user_query → retrieve_relevant_docs → augment_context → LLM → grounded_response
```

**When to use:**
- Dynamic or frequently updated information (documentation, product catalogs)
- Private/proprietary knowledge bases
- Domain-specific expertise beyond LLM training
- Reducing hallucinations with factual grounding

**Key benefits:**
- 📚 Access to current and proprietary information
- 🎯 Reduced hallucinations through grounding
- 💰 No retraining needed for knowledge updates
- 🔍 Source attribution and transparency

[**📖 Learn More →**](./foundational_design_patterns/9_rag/README.md)

---

### 🔟 [Human-in-the-Loop (HITL)](./foundational_design_patterns/10_hitl/)
**Integrate human oversight and approval into AI workflows**

The Human-in-the-Loop (HITL) pattern represents a pivotal strategy in agentic AI systems, deliberately interweaving the unique strengths of human cognition—such as judgment, creativity, and nuanced understanding—with the computational power and efficiency of AI. This strategic integration ensures that AI operates within ethical boundaries, adheres to safety protocols, and achieves its objectives with optimal effectiveness.

```python
# Without HITL: Fully automated
agent_action → execute → result

# With HITL: Human checkpoint
agent_proposal → human_review → [approve|reject|modify] → execute → result
```

**When to use:**
- High-stakes decisions (financial transactions, legal actions, sentencing)
- Quality-critical content (publications, customer communications)
- Compliance and regulatory requirements
- Complex scenarios requiring nuanced judgment
- Learning from human expertise for continuous improvement
- Tasks involving ambiguity beyond reliable LLM capabilities

**Key aspects:**
- **Human Oversight**: Monitoring AI performance via dashboards/logs to ensure guideline adherence
- **Intervention & Correction**: Human operators rectify errors or guide agents in ambiguous scenarios
- **Feedback for Learning**: Human preferences inform agent learning (e.g., RLHF)
- **Decision Augmentation**: AI provides analysis/recommendations; humans make final decisions
- **Human-Agent Collaboration**: Cooperative interaction leveraging respective strengths
- **Escalation Policies**: Protocols dictating when agents escalate tasks to humans

**Key benefits:**
- 🛡️ Safety and risk mitigation in critical domains
- ✅ Quality assurance and compliance
- 🎓 Continuous learning from human feedback
- 🤝 Building user trust through transparency
- 🎯 Nuanced judgment in complex scenarios
- 🔄 Feedback loops for ongoing improvement

**Practical applications:**
- **Content Moderation**: AI filters at scale; humans review ambiguous cases
- **Autonomous Driving**: AI handles most tasks; humans take control in complex situations
- **Financial Fraud Detection**: AI flags patterns; human analysts investigate high-risk alerts
- **Legal Document Review**: AI scans/categorizes; lawyers review for accuracy and implications
- **Customer Support**: Chatbots handle routine queries; complex/emotional cases escalate to humans
- **Data Labeling**: Humans provide ground truth labels for training datasets
- **Generative AI Refinement**: Human editors review/refine LLM outputs for quality and brand alignment
- **Autonomous Networks**: AI analyzes KPIs; humans approve critical network changes

**Trade-offs & caveats:**
- ⚠️ **Scalability limitations**: Human oversight cannot handle millions of tasks
- 👥 **Expertise dependency**: Effectiveness relies on skilled domain experts
- 🔒 **Privacy concerns**: Sensitive information requires anonymization
- 💰 **Cost considerations**: Human review adds operational overhead

**"Human-on-the-loop" variation:**
In this approach, human experts define overarching policies, while AI handles immediate actions to ensure compliance (e.g., automated trading within human-defined rules, call center routing based on manager-set policies).

[**📖 Learn More →**](./foundational_design_patterns/10_hitl/README.md)

---

### 1️⃣1️⃣ [Structured Outputs](./foundational_design_patterns/11_structured_outputs/)
**Enforce schema-valid LLM outputs for reliable downstream automation**
```python
# Naive parsing (brittle)
text → prompt_json_request → parse_string_json → runtime_fail

# Structured outputs (reliable)
text → response_schema(Pydantic/JSON Schema) → validated_object → safe_automation
```

**Key benefits:** Schema guarantees, lower parsing failures, safer agent loops

[**📖 Learn More →**](./foundational_design_patterns/11_structured_outputs/README.md)

---

### 1️⃣2️⃣ [Computer Use](./foundational_design_patterns/12_computer_use/)
**Automate browser/UI workflows with explicit safety controls**
```python
# Observe → Think → Act loop for UI tasks
screenshot/state → reasoning → ui_action(click/type/navigate) → observation → iterate
```

**Key benefits:** Legacy-system automation, UI QA workflows, non-API task coverage

[**📖 Learn More →**](./foundational_design_patterns/12_computer_use/README.md)

---

## 🧠 Advanced Reasoning Patterns

### [Tree of Thoughts](./reasoning/tree_of_thoughts/) (Yao et al., 2023)
**Explore multiple reasoning paths systematically**

Introduced at NeurIPS 2023, Tree of Thoughts (ToT) generalizes Chain-of-Thought prompting by enabling LLMs to explore multiple reasoning paths, self-evaluate choices, and backtrack when necessary—enabling deliberate problem solving for complex tasks.

```python
# Chain of Thought: Linear reasoning
input → step1 → step2 → step3 → answer

# Tree of Thoughts: Branching exploration
input → [thought1, thought2, thought3] → evaluate → expand_best →
        [refined_thoughts] → evaluate → solution
```

**Key benefits:** Better solutions through systematic exploration, backtracking capability, transparent decision trees

[**📖 Learn More →**](./reasoning/tree_of_thoughts/README.md)

---

### [Graph of Thoughts](./reasoning/graph_of_thoughts/) (Besta et al., 2023)
**Enable non-hierarchical thought connections and merging**

Building on ToT, Graph of Thoughts extends the reasoning paradigm from hierarchical trees to arbitrary graphs, enabling non-linear thought connections and aggregation—achieving 62% better quality on sorting tasks while reducing costs by 31% (Besta et al., 2023).

```python
# Thoughts can reference and build on ANY other thought (not just parent-child)
input → generate_perspectives → connect_thoughts → aggregate → synthesis
```

**Key benefits:** Multi-perspective analysis, thought merging, flexible reasoning paths

[**📖 Learn More →**](./reasoning/graph_of_thoughts/README.md)

---

### [Exploration & Discovery](./reasoning/exploration_discovery/)
**Discover novel solutions through guided exploration**
```python
# Epsilon-greedy: Balance exploration vs. exploitation
query → [explore_new | exploit_best] → evaluate → update_strategy → iterate
```

**Key benefits:** Novel solution discovery, avoiding premature convergence, adaptive exploration

[**📖 Learn More →**](./reasoning/exploration_discovery/README.md)

---

### [Deep Research](./reasoning/deep_research/)
**Run iterative research loops with gap-driven follow-up queries**
```python
# Plan → Search → Read → Reflect → Follow-up → Synthesize
question → sub_queries → retrieve_sources → identify_gaps → refine_queries → cited_output
```

**Key benefits:** Better coverage, fewer blind spots, stronger citation quality

[**📖 Learn More →**](./reasoning/deep_research/README.md)

---

## 🛡️ Reliability Patterns

### [Error Recovery](./reliability/error_recovery/)
**Gracefully handle failures and self-correct**
```python
# Detect → Diagnose → Recover → Verify
operation → [success | failure] → classify_error → [retry | fallback | self_correct] → verify
```

**Key benefits:** Resilience, graceful degradation, automatic self-healing, reduced downtime

[**📖 Learn More →**](./reliability/error_recovery/README.md)

---

### [Guardrails](./reliability/guardrails/)
**Enforce safety constraints and compliance**
```python
# Multi-layer validation
input → validate → process → validate_output → [pass | block] → log
```

**Key benefits:** Safety assurance, compliance, brand protection, risk reduction

[**📖 Learn More →**](./reliability/guardrails/README.md)

---

## 🎯 Orchestration Patterns

### [Goal Management](./orchestration/goal_management/)
**Decompose and track complex objectives**
```python
# Hierarchical decomposition with progress tracking
complex_goal → decompose → [subgoal1, subgoal2, subgoal3] →
              track_dependencies → execute → monitor → replan
```

**Key benefits:** Structured execution, progress visibility, adaptive planning, resource optimization

[**📖 Learn More →**](./orchestration/goal_management/README.md)

---

### [Subagents (Orchestrator-Worker)](./orchestration/subagents/)
**Spawn focused subagents with isolated context and structured summaries**
```python
lead_agent → decompose_task → spawn_workers_parallel → structured_summaries → synthesize
```

**Key benefits:** Context isolation, parallel throughput, cleaner synthesis

[**📖 Learn More →**](./orchestration/subagents/README.md)

---

### [Skills](./orchestration/skills/)
**Load capability packages on demand via metadata-first discovery**
```python
skill_catalog(metadata) → select_relevant_skill → load_SKILL_body → execute
```

**Key benefits:** Tool-scaling beyond flat lists, lower prompt load, modular capabilities

[**📖 Learn More →**](./orchestration/skills/README.md)

---

### [Agent Communication (A2A)](./orchestration/agent_communication/)
**Enable agents to coordinate through message passing**
```python
# Direct messaging, pub-sub, negotiation protocols
agent1 → message → agent2 → response → agent1
```

**Key benefits:** Loose coupling, dynamic discovery, scalability, fault tolerance

[**📖 Learn More →**](./orchestration/agent_communication/README.md)

---

### [Model Context Protocol (MCP)](./orchestration/mcp/)
**Standardized tool and resource integration**
```python
# USB for AI: Standard interface for tools/data
LLM → discover_tools → invoke_tool(params) → receive_result → integrate
```

**Key benefits:** Standardization, reusability, interoperability, composability

[**📖 Learn More →**](./orchestration/mcp/README.md)

---

### [Prioritization](./orchestration/prioritization/)
**Optimize task ordering and resource allocation**
```python
# Multi-criteria scoring with dynamic rebalancing
tasks → score(urgency, impact, effort) → rank → schedule → execute
```

**Key benefits:** Resource optimization, deadline adherence, fairness, efficiency

[**📖 Learn More →**](./orchestration/prioritization/README.md)

---

## 📊 Observability Patterns

### [Evaluation & Monitoring](./observability/evaluation_monitoring/)
**Track performance and quality metrics**
```python
# Quantitative + qualitative metrics
operation → collect_metrics → evaluate_quality → aggregate → alert → visualize
```

**Key benefits:** Visibility, early detection, data-driven decisions, continuous improvement

[**📖 Learn More →**](./observability/evaluation_monitoring/README.md)

---

### [Resource Optimization](./observability/resource_optimization/)
**Reduce costs and improve performance**
```python
# Caching, batching, model routing
request → [cache_hit | cache_miss] → [cheap_model | expensive_model] → optimize
```

**Key benefits:** 65-80% cost reduction, faster responses, better UX

[**📖 Learn More →**](./observability/resource_optimization/README.md)

---

## 🧩 Memory Patterns

### [Memory Management](./memory/memory_management/)
**Maintain conversation history and long-term memory**
```python
# Buffer + semantic memory
interaction → store → [buffer_memory | vector_memory] → retrieve_relevant → use
```

**Key benefits:** Context retention, personalization, learning from history

[**📖 Learn More →**](./memory/memory_management/README.md)

---

### [Context Management](./memory/context_management/)
**Optimize context window usage**
```python
# Dynamic selection and compression
content → score_relevance → compress → fit_window → optimize
```

**Key benefits:** 70-90% cost reduction, focused responses, better performance

[**📖 Learn More →**](./memory/context_management/README.md)

---

## 🎓 Learning Patterns

### [Adaptive Learning](./learning/adaptive_learning/)
**Improve through feedback and continuous learning**
```python
# Learn from outcomes
action → feedback → analyze_patterns → adapt_strategy → improve
```

**Key benefits:** Continuous improvement, personalization, domain adaptation

[**📖 Learn More →**](./learning/adaptive_learning/README.md)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11 or higher
python --version

# Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation
```bash
# Clone the repository
git clone https://github.com/gtesei/agentic_design_patterns.git
cd agentic_design_patterns

# Set up shared environment
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Repository Runtime Notes

- The repository requires **Python 3.11+**.
- Each pattern folder has its own `pyproject.toml`, so run `uv sync` inside the pattern you want to execute.
- Example scripts now use a shared bootstrap helper in `repo_support.py` to:
  - locate the repo root
  - load the root `.env`
  - make the repo importable from any pattern folder
- Set `OPENAI_MODEL` in your environment if you want to override the default example model:

```bash
export OPENAI_MODEL=gpt-4o-mini
```

- If you are behind a corporate SSL interception proxy, SSL bypass is now **opt-in**:

```bash
export AGENTIC_DISABLE_SSL=1
```

### Run Your First Pattern
```bash
# Try prompt chaining
cd foundational_design_patterns/1_prompt_chain
uv sync
uv run python src/chain_prompt.py

# Try routing
cd ../2_routing
uv sync
uv run python src/routing.py

# Try parallelization
cd ../3_parallelization
uv sync
uv run python src/parallelization.py

# Try reflection (stateful loops)
cd ../4_reflection
uv sync
uv run python src/reflection_stateful_loop.py

# Try ReAct (reasoning and acting)
cd ../8_react
uv sync
uv run python src/react_agent.py

# Try Structured Outputs (schema reliability)
cd ../11_structured_outputs
uv sync
uv run python src/structured_outputs_basic.py

# Try Computer Use (UI/browser automation framing)
cd ../12_computer_use
uv sync
uv run python src/computer_use_basic.py

# Try Subagents (orchestrator-worker)
cd ../../orchestration/subagents
uv sync
uv run python src/subagents_basic.py

# Try Skills (agent-loadable capability packages)
cd ../skills
uv sync
uv run python src/skills_basic.py

# Try Deep Research (iterative research loop)
cd ../../reasoning/deep_research
uv sync
uv run python src/deep_research_basic.py
```

### Run The Reliability Gate

Use this from the repo root to verify the shared runtime/bootstrap layer:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

What it checks:

- repo bootstrap and root discovery
- SSL bypass stays off unless explicitly enabled
- repo-authored Python files compile cleanly

### CI

GitHub Actions runs the same smoke gate on pushes and pull requests:

- `.github/workflows/reliability-gate.yml`

---

## 🗺️ Pattern Selection Guide

### Choose Your Pattern Based on Your Needs:

**Need speed?** → **Routing** + **Parallelization** + **Resource Optimization** (caching, batching)

**Need quality?** → **Reflection** + **RAG** (grounded knowledge) + **Evaluation & Monitoring**

**Need cost optimization?** → **Routing** + **Resource Optimization** (65-80% savings) + **Context Management**

**Need both speed AND quality?** → **Parallelization** + **Prompt Chaining** + **RAG**

**Complex multi-step workflow?** → **Prompt Chaining** + **Planning** + **Goal Management**

**Independent concurrent tasks?** → **Parallelization** will give you massive speedups

**High-stakes output?** → **Reflection** + **HITL** (human approval) + **Guardrails** (safety)

**External system integration?** → **Tool Use** + **MCP** (standardized protocols)

**Multi-step automation?** → **Planning** + **Goal Management** + **Agent Communication**

**Multiple roles working together?** → **Multi-Agent Collaboration** + **Agent Communication** (A2A)

**Exploratory multi-step tasks?** → **ReAct** (reasoning + actions) or **Tree of Thoughts** (exploration)

**Need transparent decision-making?** → **ReAct** (explicit reasoning) + **Evaluation & Monitoring**

**Need strict machine-readable outputs?** → **Structured Outputs** + **Guardrails**

**Need UI/browser automation?** → **Computer Use** + **HITL**

**Need iterative cited synthesis?** → **Deep Research** + **RAG**

**Need scalable multi-capability agents?** → **Subagents** + **Skills**

**Knowledge-grounded responses?** → **RAG** retrieves relevant documents before generation

**Complex reasoning tasks?** → **Tree of Thoughts** (systematic) or **Graph of Thoughts** (multi-perspective)

**Production reliability?** → **Error Recovery** + **Guardrails** + **Evaluation & Monitoring**

**Long conversations?** → **Memory Management** + **Context Management** (optimize windows)

**Continuous improvement?** → **Adaptive Learning** + **Evaluation & Monitoring** (feedback loops)

**Resource constraints?** → **Prioritization** + **Resource Optimization** + **Context Management**


---

## 🏗️ Repository Structure
```
agentic_design_patterns/
├── foundational_design_patterns/
│   ├── 1_prompt_chain/         # Sequential task decomposition
│   ├── 2_routing/              # Intelligent query routing
│   ├── 3_parallelization/      # Concurrent execution
│   ├── 4_reflection/           # Iterative refinement
│   ├── 5_tool_use/             # External system integration
│   ├── 6_planning/             # Strategic task planning
│   ├── 7_multi_agent_collaboration/  # Coordinated agents
│   ├── 8_react/                # Reasoning and acting
│   ├── 9_rag/                  # Retrieval-augmented generation
│   ├── 10_hitl/                # Human-in-the-loop
│   ├── 11_structured_outputs/  # Schema-constrained outputs
│   └── 12_computer_use/        # Browser/UI automation
│
├── reasoning/                  # Advanced reasoning patterns
│   ├── tree_of_thoughts/       # Systematic exploration
│   ├── graph_of_thoughts/      # Non-hierarchical reasoning
│   ├── exploration_discovery/  # Novel solution discovery
│   └── deep_research/          # Iterative research loops
│
├── reliability/                # Safety and resilience
│   ├── error_recovery/         # Failure handling
│   └── guardrails/             # Safety constraints
│
├── orchestration/              # Multi-agent coordination
│   ├── goal_management/        # Objective decomposition
│   ├── subagents/              # Orchestrator-worker topology
│   ├── skills/                 # Agent-loadable capability packages
│   ├── agent_communication/    # Inter-agent messaging
│   ├── mcp/                    # Model Context Protocol
│   └── prioritization/         # Task ranking
│
├── observability/              # Monitoring and optimization
│   ├── evaluation_monitoring/  # Metrics and quality
│   └── resource_optimization/  # Cost and performance
│
├── memory/                     # Context and history
│   ├── memory_management/      # Long-term memory
│   └── context_management/     # Context optimization
│
├── learning/                   # Continuous improvement
│   └── adaptive_learning/      # Learning from feedback
│
├── tests/                      # Repo-level reliability smoke tests
├── .github/workflows/          # CI workflows
├── repo_support.py             # Shared runtime/bootstrap helper
├── .env                        # Environment variables
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🎓 Learning Path

### Beginner → Intermediate → Advanced → Expert

**Phase 1: Foundations (Start Here)**
1. [Prompt Chaining](./foundational_design_patterns/1_prompt_chain/) - Foundation for everything
2. [Routing](./foundational_design_patterns/2_routing/) - Learn to optimize model selection
3. [Parallelization](./foundational_design_patterns/3_parallelization/) - Scale your applications
4. [Reflection](./foundational_design_patterns/4_reflection/) - Master quality optimization
5. [Tool Use](./foundational_design_patterns/5_tool_use/) - Connect to external systems

**Phase 2: Core Patterns**
6. [RAG](./foundational_design_patterns/9_rag/) - Knowledge-grounded responses
7. [ReAct](./foundational_design_patterns/8_react/) - Reasoning + acting
8. [Planning](./foundational_design_patterns/6_planning/) - Strategic decomposition
9. [HITL](./foundational_design_patterns/10_hitl/) - Human oversight
10. [Multi-Agent](./foundational_design_patterns/7_multi_agent_collaboration/) - Agent coordination

**Phase 3: Advanced Reasoning**
11. [Tree of Thoughts](./reasoning/tree_of_thoughts/) - Systematic exploration
12. [Graph of Thoughts](./reasoning/graph_of_thoughts/) - Multi-perspective reasoning
13. [Exploration & Discovery](./reasoning/exploration_discovery/) - Novel solutions

**Phase 4: Production Patterns**
14. [Error Recovery](./reliability/error_recovery/) - Resilience
15. [Guardrails](./reliability/guardrails/) - Safety
16. [Evaluation & Monitoring](./observability/evaluation_monitoring/) - Metrics
17. [Resource Optimization](./observability/resource_optimization/) - Cost/performance

**Phase 5: Orchestration & Memory**
18. [Goal Management](./orchestration/goal_management/) - Objective tracking
19. [Agent Communication](./orchestration/agent_communication/) - Messaging
20. [MCP](./orchestration/mcp/) - Standardized integration
21. [Prioritization](./orchestration/prioritization/) - Task ranking
22. [Memory Management](./memory/memory_management/) - Context retention
23. [Context Management](./memory/context_management/) - Optimization

**Phase 6: Continuous Improvement**
24. [Adaptive Learning](./learning/adaptive_learning/) - Learning from feedback
25. [Structured Outputs](./foundational_design_patterns/11_structured_outputs/) - Schema reliability
26. [Computer Use](./foundational_design_patterns/12_computer_use/) - Browser/UI automation
27. [Subagents](./orchestration/subagents/) - Orchestrator-worker topology
28. [Skills](./orchestration/skills/) - Capability packages
29. [Deep Research](./reasoning/deep_research/) - Iterative research loops

Each pattern builds on concepts from previous ones. Start with Phase 1, then explore other phases based on your needs.

---

## 🛠️ Tech Stack

### Core Frameworks
- **[LangChain](https://python.langchain.com/)** - Comprehensive framework for LLM applications
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Stateful workflows and multi-agent orchestration
- **[LangSmith](https://smith.langchain.com/)** - LLM application monitoring and evaluation

### Models & APIs
- **[OpenAI Models](https://openai.com/)** - Primary provider used throughout the examples, configured via environment variables
- **[Anthropic Claude](https://anthropic.com/)** - Alternative frontier-model family with strong long-context support
- **[Other LLM Providers](https://python.langchain.com/docs/integrations/llms/)** - Fully compatible through LangChain abstractions

### Development Tools
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation and structured outputs
- **[Python 3.11+](https://www.python.org/)** - Modern Python features (match/case, typing)
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager

### Observability & Evaluation
- **[W&B Weave](https://wandb.ai/site/weave/)** - Agent evaluation and monitoring
- **[LangSmith](https://smith.langchain.com/)** - Tracing and debugging

---


## 📖 Resources

### 🎓 Academic Papers & Surveys

**Reasoning & Planning:**
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) - ICLR 2023
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) (Yao et al., 2023) - NeurIPS 2023
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687) (Besta et al., 2023)

**Retrieval-Augmented Generation:**
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) (Singh et al., 2025)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)

### 📚 Books

- **[Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems](https://link.springer.com/book/10.1007/978-3-031-87617-1)** - Antonio Gullí (Springer Nature, 2024) - Primary inspiration for this repository
- **[Building LLM Powered Applications](https://www.oreilly.com/library/view/building-llm-powered/9781835462317/)** - Valentina Alto (Packt/O'Reilly, 2024)
- **[Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large/9781098150952/)** - Jay Alammar & Maarten Grootendorst (O'Reilly, 2024)

### 🎓 Courses & Educational Content

**Foundational Courses:**
- **[Agentic AI with Andrew Ng](https://www.deeplearning.ai/courses/agentic-ai/)** (DeepLearning.AI, 2024) - Covers reflection, tool use, planning, and multi-agent collaboration

**Framework-Specific:**
- [LangChain Academy](https://academy.langchain.com/) - Official LangChain courses
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) - Stateful agent workflows
- [OpenAI Cookbook](https://cookbook.openai.com/) - Function calling and agent patterns
- [Anthropic Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)

### 🏭 Industry Documentation & Guides

**Official Framework Documentation:**
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/stable/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Agents Platform](https://platform.openai.com/docs/guides/agents)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

**Production Best Practices:**
- [LangChain: Top 5 LangGraph Agents in Production 2024](https://www.blog.langchain.com/top-5-langgraph-agents-in-production-2024/) - Real-world deployments
- [Weights & Biases: Agentic RAG Guide](https://wandb.ai/byyoung3/Generative-AI/reports/Agentic-RAG-Enhancing-retrieval-augmented-generation-with-AI-agents--VmlldzoxMTcyNjQ5Ng)

### 🌐 Community Resources

**Curated Collections:**
- [Awesome-LangGraph](https://github.com/von-development/awesome-LangGraph) - Comprehensive LangGraph ecosystem index
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Comprehensive guide covering latest papers and techniques
- [Learn Prompting](https://learnprompting.org/) - Free generative AI guide

**Related Projects:**
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AG2 (formerly AutoGen)](https://github.com/ag2ai/ag2)

### 🔬 Research Collections

- [Papers with Code: Agents](https://paperswithcode.com/task/agents) - Latest research with implementations
- [arXiv: Artificial Intelligence](https://arxiv.org/list/cs.AI/recent) - Recent AI papers
- [Hugging Face Papers](https://huggingface.co/papers) - Trending ML research

---

## 🏛️ Standards & Compliance

### NIST AI Risk Management Framework

Organizations deploying agentic AI systems should consider the NIST AI Risk Management Framework and associated guidance:

#### **Core Framework**
- **[NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework)** (January 2023)
  Voluntary framework to manage AI risks based on four core functions: Govern, Map, Measure, and Manage.

#### **Generative AI Profile**
- **[NIST AI RMF: Generative AI Profile (NIST.AI.600-1)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)** (July 2024)
  Addresses risks unique to Generative AI, including governance, content provenance, pre-deployment testing, and incident disclosure. Includes catalog of 400+ mitigation actions.

### Compliance Mapping

| NIST AI RMF Function | Relevant Patterns |
|---------------------|-------------------|
| **Govern** | [Human-in-the-Loop](./foundational_design_patterns/10_hitl/), [Guardrails](./reliability/guardrails/) |
| **Map** | [Planning](./foundational_design_patterns/6_planning/), [Goal Management](./orchestration/goal_management/) |
| **Measure** | [Evaluation & Monitoring](./observability/evaluation_monitoring/), [Adaptive Learning](./learning/adaptive_learning/) |
| **Manage** | [Error Recovery](./reliability/error_recovery/), [Guardrails](./reliability/guardrails/) |

### Key Focus Areas for Agentic Systems

- **Transparency:** ReAct pattern provides explicit reasoning traces for auditability
- **Human Oversight:** HITL pattern enables approval workflows
- **Safety Constraints:** Guardrails pattern enforces compliance boundaries
- **Evaluation:** Monitoring pattern tracks quality and bias metrics
- **Error Recovery:** Graceful degradation and incident response

### Additional Resources

- [AI Executive Order 14110](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/)
- [EU AI Act](https://artificialintelligenceact.eu/)
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) - AI Management System standard

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

This repository's structure and approach were inspired by:

### Primary References

> **Gullí, Antonio**, *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*, Springer Nature Switzerland, 2024.

> **Ng, Andrew**, *Agentic AI*, DeepLearning.AI, 2024.

### Academic Foundations

We gratefully acknowledge the research contributions that ground these patterns:

- **Yao, Shunyu et al.** - ReAct and Tree of Thoughts frameworks
- **Singh, Aditi et al.** - Agentic RAG survey and taxonomy
- **Besta, Maciej et al.** - Graph of Thoughts methodology
- **Lewis, Patrick et al.** - Foundational RAG research

### Community & Tools

Special thanks to:
- The **LangChain and LangGraph teams** for building production-grade agentic frameworks
- The **open-source AI community** for advancing the state of the art
- **NIST** for providing guidance on trustworthy AI development
- **[Claude Code](https://claude.ai/)** for assistance in developing and refining the implementations in this repository
- All **contributors** who help improve these patterns

---

## ⭐ Star History

If you find this repository helpful, please consider giving it a star! It helps others discover these patterns.

[![Star History Chart](https://api.star-history.com/svg?repos=gtesei/agentic_design_patterns&type=Date)](https://star-history.com/#gtesei/agentic_design_patterns&Date)
---

<div align="center">

**Built with ❤️ for the AI developer community**

[⬆ Back to Top](#-agentic-design-patterns)

</div>
