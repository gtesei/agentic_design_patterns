"""Subagents basic: orchestrator-worker with structured subagent summaries."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class WorkerSummary(BaseModel):
    subtask: str
    findings: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)


def worker_call(llm: ChatOpenAI, subtask: str) -> WorkerSummary:
    structured = llm.with_structured_output(WorkerSummary)
    return structured.invoke(
        f"Analyze this subtask for support-ops research and return structured summary:\n{subtask}"
    )


def main() -> None:
    print("\n" + "=" * 80)
    print("SUBAGENTS — BASIC ORCHESTRATOR/WORKER")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    llm = ChatOpenAI(model=get_default_model(), temperature=0)
    objective = "Investigate payment approval latency complaints from enterprise customers."
    subtasks = [
        "Check likely infrastructure bottlenecks and runbook actions.",
        "Assess customer-impact and SLA implications.",
        "Draft immediate triage and escalation recommendation.",
    ]

    print(f"\nObjective: {objective}")
    print("Running 3 subagents in parallel...\n")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker_call, llm, s) for s in subtasks]
        summaries = [f.result() for f in futures]

    for idx, summary in enumerate(summaries, 1):
        print(f"Worker {idx}: {summary.model_dump()}")

    synthesis_prompt = (
        "Synthesize a final report from these structured worker summaries:\n"
        + "\n".join(str(s.model_dump()) for s in summaries)
    )
    final_report = llm.invoke(synthesis_prompt).content

    print("\nFinal synthesis:")
    print(final_report)


if __name__ == "__main__":
    main()
