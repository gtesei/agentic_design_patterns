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

## Example — Code as Plan

This lab illustrates the code-as-plan idea in practice: let the model write Python (TinyDB) that both documents and executes the plan.

### Key properties demonstrated

* Code is the plan: LLM produces commented Python that enumerates steps (filter → compute → update).

* Safe execution: run code in a locked namespace with only permitted helpers and TinyDB objects.

* Action policy & contracts: generated code must set answer_text and STATUS, print logs, and respect mutation policies.

* Before/after snapshots: executor renders diffs of inventory_tbl / transactions_tbl.

Core flow (pseudocode)

```python
# 1) Generate code plan from LLM
full_content = generate_llm_code(prompt, schema_block, model="o4-mini")

# 2) Extract the <execute_python> block
code = _extract_execute_block(full_content)

# 3) Execute in safe namespace
res = execute_generated_code(code, db, inventory_tbl, transactions_tbl)

# 4) Inspect res["answer"], res["stdout"], res["error"], and table snapshots
```

### Important constraints used in the lab example

* answer_text (1–2 sentence user-friendly output) is mandatory

* STATUS must be one of (success, no_match, insufficient_stock, invalid_request, unsupported_intent)

* Hard rules for mutations (e.g., single transaction per item) enforced in the prompt

* Sandbox disallows network, file I/O, or external imports (only TinyDB helpers allowed)

### Safety & Execution Guidelines (critical for Code-as-Plan)

* Restrict globals/locals — provide only safe helpers and limited objects.

* Parse & validate code before execution:

* Use regex to extract allowed <execute_python> block.

* Static checks: forbid import os, open, subprocess, sockets.

* Run in sandbox (container or hardened Python runtime).

* Limit execution time & resources (timeouts, CPU/memory caps).

* Require explicit output contract (answer_text, STATUS) to standardize results.

* Log everything — stdout, errors, mutated rows, and full code executed.

* Human-in-the-loop for high-risk mutations — approve before commit.

## Best Practices & Prompt Engineering

* Make plans concrete: each step must be actionable, have a success condition, and list tools.

* Bound side-effects: enforce DRY RUN behavior for ambiguous intents.

* State ledger: store canonical facts/goals in CONTINUITY.md or a DB table for long-running agents.

* Prefer small, testable steps: easier to replan and to surface failures.

* Define failure modes clearly and how to recover (retry, replan, escalate).

* Always require a short human-facing answer_text to avoid noisy outputs.

---

## Templates & Snippets

### Minimal plan prompt (JSON-style)

```css
Objective: {objective}
Constraints: {constraints}
Produce a stepwise JSON plan:
[
  {"id":1,"desc":"...","tools":["..."],"depends_on":[],"success_criteria":"..."},
  ...
]
```

### Minimal plan prompt (Code-as-Plan)

```yaml
You are an assistant that PLANs BY WRITING PYTHON.
Schema: {schema_block}
User request: {question}
Output: A single <execute_python>...</execute_python> block.
Requirements:
- Use provided TinyDB tables: inventory_tbl, transactions_tbl
- Set `answer_text` (1-2 lines) and `STATUS`
- No network, no file I/O, no imports other than allowed helpers
```

---

### Metrics to Track (KPIs)

* Plan quality: manual rating or automated completeness check

* Execution success rate: % plans that complete without human fix

* Time to completion: end-to-end wall time

* Step failure rate: % steps that fail/require replan

* Mutations per run: number of DB/APICalls executed

* Human edits: frequency & size of post-run human modifications

## Drawbacks & Caveats

* Code-as-plan increases attack surface — must be sandboxed rigorously.

* LLM-generated code can encode incorrect logic; require tests and invariant checks.

* Research ideas often don’t translate directly into production — validate empirically.

* Overplanning for simple tasks wastes time — choose patterns judiciously.

---

## Conclusion

The Planning Pattern—especially when combined with code-as-plan—is a powerful approach for orchestrating complex, multi-step workflows. It trades some upfront cost and safety complexity for expressiveness and reduced orchestration plumbing. Use structured prompts, pragmatic safeguards, and empirical evaluation (chunking, hierarchical roll-ups, ledgers) to get the best of both worlds: reliable automation and human-grade auditability.

---

*Planning elevates agents from reactive executors to strategic problem-solvers capable of tackling complex, real-world objectives.*