"""Subagents advanced: coding-agent style planner/executor/tester subagents."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_advanced_model

configure_example(__file__)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class SubagentOutput(BaseModel):
    role: str
    summary: str
    next_actions: list[str] = Field(default_factory=list)


def run_role(llm: ChatOpenAI, role: str, task: str) -> SubagentOutput:
    structured = llm.with_structured_output(SubagentOutput)
    return structured.invoke(f"Role={role}. Task={task}. Return concise structured output.")


def main() -> None:
    print("\n" + "=" * 80)
    print("SUBAGENTS — ADVANCED (PLANNER/EXECUTOR/TESTER)")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    llm = ChatOpenAI(model=get_advanced_model(), temperature=0)

    tasks = {
        "planner": "Plan changes to reduce flaky smoke tests for demo scripts.",
        "executor": "Implement robust non-interactive execution approach and timeout handling.",
        "tester": "Define validation checks and likely failure cases.",
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_role, llm, role, task) for role, task in tasks.items()]
        outputs = [f.result() for f in futures]

    print("\nSubagent outputs:")
    for out in outputs:
        print(out.model_dump())

    synth = llm.invoke("Create final implementation brief from these outputs:\n" + "\n".join(str(o.model_dump()) for o in outputs)).content
    print("\nFinal implementation brief:")
    print(synth)


if __name__ == "__main__":
    main()
