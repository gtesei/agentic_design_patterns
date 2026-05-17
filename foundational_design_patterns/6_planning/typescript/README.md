# Planning (Plan-and-Act) — TypeScript

**Direct port of `../src/planning_plan_and_act.py`.** Same scenario (payment-API incident response), same `PlanActState` shape, same three nodes (`planner`, `executor`, `reviewer`), same conditional routing.

## Flow

```
START → planner → executor → reviewer ─┬→ executor   (steps remain)
                                        ├→ planner    (replan needed)
                                        └→ END        (plan complete)
```

- **planner** — LLM call that produces an ordered list of incident-response actions (filtered against `completed`).
- **executor** — LLM call that simulates running one action and reports `STATUS: done|blocked`. Sets `requires_replan` on blocked.
- **reviewer** — pure routing logic that appends a status note. Decides via `routeAfterReview`.

## Run it

```bash
bash run.sh
```

## Smoke test

```bash
bun test
```

Tests cover the pure-function nodes (`reviewerNode`, `routeAfterReview`) and the graph topology. The LLM nodes (`plannerNode`, `executorNode`) are not invoked in smoke tests — they need an API key.

## Notes on parity

| Python | TypeScript |
|---|---|
| `@dataclass class PlanActState` | `Annotation.Root({ ... })` with explicit reducers |
| `StateGraph(PlanActState)` | `new StateGraph(PlanActState)` |
| `graph.add_conditional_edges("reviewer", route_after_review, {...})` | `.addConditionalEdges("reviewer", routeAfterReview, {...})` |
| `END` from `langgraph.graph` | `END` from `@langchain/langgraph` |
| `llm.invoke(prompt).content` | `(await llm.invoke(prompt)).content` (with string normalization) |
