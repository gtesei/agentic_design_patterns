"""Deep Research advanced: plan + multi-query + gap-driven refinement with citation list."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_advanced_model

configure_example(__file__)

import wikipedia
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)
    stopping_criterion: str


@dataclass
class AdvancedState:
    question: str
    plan: ResearchPlan | None = None
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def fetch_summary(topic: str) -> tuple[str, str]:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        summary = wikipedia.summary(topic, sentences=2)
        return summary, page.url
    except Exception:
        return f"No reliable result for {topic}", "n/a"


def main() -> None:
    print("\n" + "=" * 80)
    print("DEEP RESEARCH — ADVANCED")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    llm = ChatOpenAI(model=get_advanced_model(), temperature=0)
    structured = llm.with_structured_output(ResearchPlan)

    try:
        state = AdvancedState(question="What operating model should support teams use for recurring payment API incidents?")
        state.plan = structured.invoke("Create deep-research sub-queries and stopping criterion for: " + state.question)

        print("\nPlan:")
        print(state.plan.model_dump())

        for query in state.plan.sub_queries[:3]:
            summary, src = fetch_summary(query)
            state.notes.append(f"{query}: {summary}")
            state.sources.append(src)

        gap_questions = llm.invoke("Given these notes, list 2 missing-information questions:\n" + "\n".join(state.notes)).content
        state.notes.append("Gap analysis:\n" + gap_questions)

        final = llm.invoke(
            "Synthesize a concise operating model with citations.\nNotes:\n"
            + "\n\n".join(state.notes)
            + "\nSources:\n"
            + "\n".join(state.sources)
        ).content

        print("\nFinal output:")
        print(final)
    except Exception as exc:
        print("\n⚠️ Advanced deep research run failed:")
        print(exc)
        print("Tip: AGENTIC_DISABLE_SSL=1 bash run.sh")


if __name__ == "__main__":
    main()
