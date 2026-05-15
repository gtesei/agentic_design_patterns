"""
Multi-Agent Collaboration: Orchestrator-Worker topology

Anchor scenario: deep research triage for support operations.

Preferred runtime in 2026:
- langgraph-supervisor (if installed)
Fallback:
- lightweight LangGraph orchestrator with explicit worker roles
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


@dataclass
class OrchestratorState:
    goal: str
    work_items: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    final_report: str = ""


llm = ChatOpenAI(model=get_default_model(), temperature=0)


def supervisor_plan(state: OrchestratorState) -> OrchestratorState:
    prompt = f"""
Break this support-ops research goal into exactly 3 worker tasks:
{state.goal}

Return one task per line.
"""
    tasks = [line.strip("-• ").strip() for line in llm.invoke(prompt).content.splitlines() if line.strip()]
    state.work_items = tasks[:3]
    return state


def worker_research(state: OrchestratorState) -> OrchestratorState:
    if not state.work_items:
        return state

    task = state.work_items.pop(0)
    result = llm.invoke(
        f"Research this subtask for support operations and return 2 concise bullets:\n{task}"
    ).content
    state.results.append(f"TASK: {task}\n{result}")
    return state


def synthesize(state: OrchestratorState) -> OrchestratorState:
    prompt = f"""
Synthesize a final report from these worker outputs.

Goal: {state.goal}
Worker outputs:
{state.results}

Return:
- executive summary
- immediate actions (3 bullets)
- escalation recommendation
"""
    state.final_report = llm.invoke(prompt).content
    return state


def route_worker(state: OrchestratorState) -> str:
    return "worker" if state.work_items else "synthesize"


def try_supervisor_package() -> bool:
    try:
        import langgraph_supervisor  # noqa: F401

        print("✅ langgraph-supervisor detected. (This demo uses lightweight fallback graph for portability.)")
        return True
    except Exception:
        print("ℹ️ langgraph-supervisor not installed; using built-in fallback graph.")
        return False


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", supervisor_plan)
    graph.add_node("worker", worker_research)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_worker, {"worker": "worker", "synthesize": "synthesize"})
    graph.add_conditional_edges("worker", route_worker, {"worker": "worker", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile()


def main() -> None:
    print("\n" + "=" * 80)
    print("MULTI-AGENT — ORCHESTRATOR/WORKER")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    try_supervisor_package()

    app = build_graph()
    goal = (
        "Investigate enterprise customer complaints about payment approval latency and produce "
        "a triage recommendation for support operations."
    )
    out = app.invoke(OrchestratorState(goal=goal))

    if isinstance(out, dict):
        final_report = out.get("final_report", "")
    else:
        final_report = getattr(out, "final_report", "")

    print("\n🧾 Final report:")
    print(final_report)


if __name__ == "__main__":
    main()
