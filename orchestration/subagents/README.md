# Subagents Pattern (Orchestrator-Worker)

Demonstrates local subagents with context isolation and structured sub-results.

- `src/subagents_basic.py`: 3 parallel workers -> structured summaries -> synthesis
- `src/subagents_advanced.py`: planner/executor/tester parallel pattern

Run:

```bash
uv sync
bash run.sh
```

If you're behind a corporate SSL-inspecting proxy:

```bash
AGENTIC_DISABLE_SSL=1 bash run.sh
```
