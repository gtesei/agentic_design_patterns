"""
Planning Pattern: Plan-and-Act with explicit replanning (LangGraph)

Anchor scenario: incident response automation.

Flow:
- planner node creates ordered actions
- executor node performs one action at a time
- reviewer node decides whether replanning is required
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


@dataclass
class PlanActState:
    """State container for plan-and-act execution."""

    incident: str
    context: str
    plan: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    current_action: str = ""
    notes: list[str] = field(default_factory=list)
    requires_replan: bool = False


llm = ChatOpenAI(model=get_default_model(), temperature=0)


def planner_node(state: PlanActState) -> PlanActState:
    """Create or refresh the execution plan."""
    prompt = f"""
You are an incident commander. Build a short action plan (4-6 steps) for this incident.

Incident:
{state.incident}

Context:
{state.context}

Already completed:
{state.completed}

Return one step per line, no numbering.
"""
    response = llm.invoke(prompt).content.strip().splitlines()
    plan = [line.strip("-• ").strip() for line in response if line.strip()]

    state.plan = [step for step in plan if step not in state.completed]
    state.notes.append(f"planner: generated {len(state.plan)} remaining steps")
    state.requires_replan = False
    return state


def executor_node(state: PlanActState) -> PlanActState:
    """Execute one action from the plan."""
    if not state.plan:
        state.current_action = ""
        return state

    action = state.plan.pop(0)
    state.current_action = action

    exec_prompt = f"""
You are executing one incident response action.

Action: {action}
Incident: {state.incident}

Return exactly two lines:
STATUS: done|blocked
NOTE: <short operational note>
"""
    result = llm.invoke(exec_prompt).content
    lines = [line.strip() for line in result.splitlines() if line.strip()]

    status = "done"
    note = "no note"
    for line in lines:
        if line.lower().startswith("status:"):
            status = line.split(":", 1)[1].strip().lower()
        if line.lower().startswith("note:"):
            note = line.split(":", 1)[1].strip()

    if status == "done":
        state.completed.append(action)
        state.notes.append(f"executor: done -> {action} | {note}")
        state.requires_replan = False
    else:
        state.notes.append(f"executor: blocked -> {action} | {note}")
        state.requires_replan = True

    return state


def reviewer_node(state: PlanActState) -> PlanActState:
    """Decide if another step can run or if replanning is needed."""
    if state.requires_replan:
        state.notes.append("reviewer: requesting replanning")
    elif state.plan:
        state.notes.append("reviewer: continue with remaining plan")
    else:
        state.notes.append("reviewer: plan complete")
    return state


def route_after_review(state: PlanActState) -> Literal["planner", "executor", "end"]:
    if state.requires_replan:
        return "planner"
    if state.plan:
        return "executor"
    return "end"


def build_graph():
    graph = StateGraph(PlanActState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"planner": "planner", "executor": "executor", "end": END},
    )

    return graph.compile()


def main() -> None:
    print("\n" + "=" * 80)
    print("PLANNING PATTERN — PLAN-AND-ACT (LANGGRAPH)")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required for this example.")
        return

    initial = PlanActState(
        incident="Payment API error rate jumped to 35% after deployment.",
        context="Service: billing-gateway, Region: eu-west, Recent change: release v2.8.4",
    )

    app = build_graph()
    final = app.invoke(initial)

    if isinstance(final, dict):
        completed = final.get("completed", [])
        notes = final.get("notes", [])
    else:
        completed = getattr(final, "completed", [])
        notes = getattr(final, "notes", [])

    print("\n✅ Completed steps:")
    for idx, step in enumerate(completed, 1):
        print(f"  {idx}. {step}")

    print("\n🧾 Execution notes:")
    for note in notes:
        print(f"  - {note}")


if __name__ == "__main__":
    main()
