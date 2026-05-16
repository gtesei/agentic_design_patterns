"""Deep Research basic: iterative search -> read -> reflect -> follow-up loops."""

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

import wikipedia
from langchain_openai import ChatOpenAI


@dataclass
class ResearchState:
    question: str
    queries: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as exc:
        return f"No summary for {query}: {exc}"


def main() -> None:
    print("\n" + "=" * 80)
    print("DEEP RESEARCH — BASIC ITERATIVE LOOP")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    llm = ChatOpenAI(model=get_default_model(), temperature=0)
    state = ResearchState(question="How should enterprises mitigate payment API latency incidents?")

    try:
        state.queries = [
            "incident response best practices",
            "API latency mitigation",
            "SLA escalation enterprise support",
        ]

        for round_idx in range(2):
            print(f"\n--- Round {round_idx + 1} ---")
            round_findings = []
            for query in state.queries:
                text = wiki_search(query)
                round_findings.append(f"Q={query}\n{text}")
                print(f"Retrieved: {query}")

            state.findings.extend(round_findings)

            reflection = llm.invoke(
                "Given these findings, list up to 3 missing-information follow-up queries:\n" + "\n\n".join(round_findings)
            ).content.splitlines()
            state.queries = [line.strip("-• ").strip() for line in reflection if line.strip()][:3]

        final = llm.invoke(
            f"Question: {state.question}\nFindings:\n" + "\n\n".join(state.findings) + "\nProvide cited synthesis with uncertainties."
        ).content

        print("\nFinal synthesis:")
        print(final)
    except Exception as exc:
        print("\n⚠️ Deep research loop failed due to connectivity/provider issue:")
        print(exc)
        print("Tip: AGENTIC_DISABLE_SSL=1 bash run.sh")


if __name__ == "__main__":
    main()
