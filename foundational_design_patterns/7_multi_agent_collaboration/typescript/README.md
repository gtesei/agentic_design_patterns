# Multi-Agent Collaboration — TypeScript

**Direct ports of `../src/orchestrator_worker.py`, `../src/peer_swarm.py`, and `../src/research_report_agent.py`.** The TypeScript demos preserve the same three collaboration styles:

- orchestrator/worker with a lightweight LangGraph loop
- peer/swarm with explicit peer analysis and critique
- the legacy monolithic supervisor with planner, router, tool-using research agent, writer, and editor roles

## Run it

```bash
bash run.sh
```

## Smoke test

```bash
bun test
```
