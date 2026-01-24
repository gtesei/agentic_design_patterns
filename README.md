# 🤖 Agentic Design Patterns

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Transform your AI applications from simple prompts to sophisticated intelligent systems.**

A comprehensive, hands-on collection of design patterns for building robust agentic AI systems. Each pattern is implemented with real-world examples and detailed documentation to help you architect scalable, maintainable AI applications.

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
git clone https://github.com/yourusername/agentic_design_patterns.git
cd agentic_design_patterns

# Set up environment
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Install dependencies (using uv)
uv sync
```

### Run Your First Pattern
```bash
# Try prompt chaining
cd foundational_design_patterns/1_prompt_chain
bash run.sh

# Try routing
cd ../2_routing
uv run python src/routing_example.py

# Try parallelization
cd ../3_parallelization
bash run.sh

# Try reflection (stateful loops)
cd ../4_reflection
bash run.sh
```

---

## 🗺️ Pattern Selection Guide

### Choose Your Pattern Based on Your Needs:

**Need speed?** → Start with **Routing** and **Parallelization**

**Need quality?** → Use **Prompt Chaining** + **Reflection**

**Need cost optimization?** → Implement **Routing** to avoid expensive models

**Need both speed AND quality?** → Combine **Parallelization** + **Prompt Chaining**

**Complex multi-step workflow?** → **Prompt Chaining** is your foundation

**Independent concurrent tasks?** → **Parallelization** will give you massive speedups

**High-stakes output?** → **Reflection** is worth the cost

**External system integration?** → **Tool Use** enables real-world interaction

**Multi-step automation requiring orchestration?** → **Planning** provides strategic execution

**Need multiple roles working together?** → **Multi-Agent Collaboration** (specialists + coordinator)


---

## 🏗️ Repository Structure
```
agentic_design_patterns/
├── foundational_design_patterns/
│   ├── 1_prompt_chain/
│   │   ├── src/
│   │   │   ├── chain_prompt.py            # Basic chaining
│   │   │   └── advanced_chain.py          # Complex workflows
│   │   ├── README.md                      # Pattern documentation
│   │   └── SKILL.md                       # Implementation guide
│   │
│   ├── 2_routing/
│   │   ├── src/
│   │   │   ├── routing.py                 # Intent-based routing
│   │   │   └── semantic_routing.py        # Advanced routing
│   │   ├── README.md
│   │   └── SKILL.md
│   │
│   ├── 3_parallelization/
│   │   ├── src/
│   │   │   ├── parallel_example.py        # LCEL parallelization
│   │   │   └── async_parallel.py          # Async operations
│   │   ├── README.md
│   │   └── SKILL.md
│   │
│   └── 4_reflection/
│       ├── src/
│       │   ├── reflection.py               # Single-step reflection
│       │   └── reflection_stateful_loop.py # Iterative refinement
│       ├── README.md
│       └── SKILL.md
...
...
├── .env                                # Environment variables
├── LICENSE                             # MIT License
└── README.md                           # This file
```

---

## 🎓 Learning Path

### Beginner → Intermediate → Advanced

1. **Start here**: [Prompt Chaining](./foundational_design_patterns/1_prompt_chain/) - Foundation for everything
2. **Next**: [Routing](./foundational_design_patterns/2_routing/) - Learn to optimize model selection
3. **Then**: [Parallelization](./foundational_design_patterns/3_parallelization/) - Scale your applications
4. **Finally**: [Reflection](./foundational_design_patterns/4_reflection/) - Master quality optimization
5. **Advanced**: [Multi-Agent Collaboration](./foundational_design_patterns/7_multi_agent_collaboration/) - Build coordinated agent teams

Each pattern builds on concepts from previous ones, so we recommend following this sequence.

---

## 🛠️ Tech Stack

- **[LangChain](https://python.langchain.com/)** - Framework for LLM applications
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Stateful workflows and agents
- **[OpenAI GPT-4/GPT-4o/GPT-5.2](https://openai.com/)** - Primary LLM (configurable)
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation and structured outputs
- **[Python 3.11+](https://www.python.org/)** - Modern Python features

---

## 🔮 Coming Soon

We're actively developing additional patterns:

- **Retrieval-Augmented Generation (RAG)** - Knowledge-grounded responses
- **Human-in-the-Loop** - Interactive approval and refinement
- **Guardrails** - Safety, compliance, and quality enforcement

**Want a specific pattern?** [Open an issue](https://github.com/gtesei/agentic_design_patterns/issues) and let us know!

---

## 📖 Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

### Related Projects
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

This repository's structure and approach were inspired by:

> **Gullí, Antonio**, *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*, Springer Nature Switzerland.

> **Andrew Ng**, Agentic AI, DeepLearning.AI.

Special thanks to:
- The LangChain team for building incredible tools
- The open-source AI community for pushing the boundaries
- All contributors who help improve these patterns

---

## ⭐ Star History

If you find this repository helpful, please consider giving it a star! It helps others discover these patterns.

[![Star History Chart](https://api.star-history.com/svg?repos=gtesei/agentic_design_patterns&type=Date)](https://star-history.com/#gtesei/agentic_design_patterns&Date)
---

<div align="center">

**Built with ❤️ for the AI developer community**

[⬆ Back to Top](#-agentic-design-patterns)

</div>
