# Planning Pattern

## Overview

The Planning Pattern turns reactive agents into strategic executors by decomposing complex, multi-step goals into ordered, testable sub-tasks before execution.
It’s useful when a user request requires coordination across tools, conditional logic, state changes, or multi-step reasoning that benefits from an explicit plan.

**Core idea**: generate a plan (steps + dependencies) → execute steps (sequentially or in parallel) → monitor & adapt.

---

## When to Use

### Use Planning when

* The task has multiple dependent steps (or tools) and order matters.

* You must orchestrate APIs, DB updates, or long workflows.

* Progress, accountability, and visibility matter (audit logs, approvals).

* The task needs retries, rollback, or conditional branching.

### Don’t use Planning when

* Task is single-step, trivial, or latency-sensitive.

* A fixed prompt chain already covers the flow.

* Immediate one-shot answers are sufficient.

## Two Flavors

### 1) Planning without code generation (JSON / Plain-text plans)

* Planner outputs a structured plan (JSON or bullet steps).

* An orchestration layer (executor) parses the plan and calls tools.

* Pros: easier to validate & restrict; simpler to audit.

* Cons: can become brittle with many tiny tools; requires a rich orchestration layer.

**Example plan (JSON):**

```json
{
  "objective": "Market analysis report",
  "steps": [
    {"id": 1, "action": "search_competitors", "tools": ["web_search"]},
    {"id": 2, "action": "collect_pricing", "depends_on": [1], "tools": ["web_search"]},
    {"id": 3, "action": "synthesize", "depends_on": [1,2], "tools": ["LLM"]}
  ]
}
```

### 2) Planning with code generation (Code-as-Plan / Code-as-Action)

* The LLM emits executable code (e.g., Python) that is the plan.

* Executor runs that code in a sandboxed environment (limited globals, no network).

* Pros: expressive, compact, leverages model’s knowledge of libraries and control flow; fewer micro-tools needed.

* Cons: higher security risk if not sandboxed; harder to statically validate; requires robust safety checks.

**Pattern**: LLM → ```<execute_python> ... </execute_python>``` → sandboxed ```exec()``` → extract ```answer_text``` + side-effects.

```text
User Request
    ↓
Planner (LLM)
    ├─ Option A: JSON plan  ──> Plan Executor (orchestrator) ──> Tools/APIs
    └─ Option B: Code-as-plan ──> Safe Sandbox Executor ──> DB/Tools (restricted)
    ↓
Monitoring & Adaptation (logs, replan)
    ↓
Final Answer (answer_text) + artifacts
```

---

## Lightweight Hierarchical Summarization (Tie-in)

When applying planning to summarization or long workflows, consider lightweight hierarchical steps:

* Chunk → local summary → grouped roll-ups → final reducer

* Optionally maintain a facts ledger that persists key facts between passes (disk-backed continuity).
This reduces “lost in the middle” and lets the reducer reason over reinforced facts instead of a single giant blob.






---

*Planning elevates agents from reactive executors to strategic problem-solvers capable of tackling complex, real-world objectives.*