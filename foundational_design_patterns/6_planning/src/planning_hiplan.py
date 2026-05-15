"""
Planning Pattern: HiPlan-style hierarchical decomposition

Anchor scenario: onboarding automation for a new enterprise customer.

This script demonstrates:
- high-level milestones
- sub-plan generation per milestone
- milestone-by-milestone execution summaries
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


@dataclass
class Milestone:
    """HiPlan milestone with generated subtasks."""

    name: str
    goal: str
    subtasks: list[str] = field(default_factory=list)


llm = ChatOpenAI(model=get_default_model(), temperature=0)


def generate_milestones(objective: str) -> list[Milestone]:
    prompt = f"""
Create 5 milestones for the objective below.
Return each milestone as: name | goal

Objective:
{objective}
"""
    lines = llm.invoke(prompt).content.strip().splitlines()

    milestones: list[Milestone] = []
    for line in lines:
        if "|" not in line:
            continue
        left, right = line.split("|", 1)
        milestones.append(Milestone(name=left.strip("-• ").strip(), goal=right.strip()))

    return milestones[:5]


def generate_subtasks(milestone: Milestone, objective: str) -> Milestone:
    prompt = f"""
You are planning execution details for one milestone.
Objective: {objective}
Milestone: {milestone.name}
Goal: {milestone.goal}

Return 4 atomic subtasks, one per line.
"""
    lines = llm.invoke(prompt).content.strip().splitlines()
    milestone.subtasks = [line.strip("-• ").strip() for line in lines if line.strip()][:4]
    return milestone


def execute_subtasks(milestone: Milestone) -> list[str]:
    logs: list[str] = []
    for subtask in milestone.subtasks:
        prompt = f"""
Simulate execution of this onboarding subtask and return one short status sentence.
Subtask: {subtask}
"""
        status = llm.invoke(prompt).content.strip()
        logs.append(f"{subtask} -> {status}")
    return logs


def main() -> None:
    print("\n" + "=" * 80)
    print("PLANNING PATTERN — HIPLAN HIERARCHICAL DECOMPOSITION")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required for this example.")
        return

    objective = (
        "Onboard a new enterprise customer for our SaaS platform, including security review, "
        "identity integration, data migration, training, and go-live support."
    )

    milestones = generate_milestones(objective)
    if not milestones:
        print("⚠️ Could not generate milestones.")
        return

    print("\n🏁 Milestones:")
    for idx, ms in enumerate(milestones, 1):
        print(f"  {idx}. {ms.name} — {ms.goal}")

    print("\n🔧 Generating subtasks and execution logs...")
    for ms in milestones:
        generate_subtasks(ms, objective)
        logs = execute_subtasks(ms)

        print(f"\n📌 {ms.name}")
        for sub in ms.subtasks:
            print(f"   - {sub}")
        print("   Execution:")
        for log in logs:
            print(f"     • {log}")


if __name__ == "__main__":
    main()
