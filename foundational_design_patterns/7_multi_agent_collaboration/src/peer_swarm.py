"""
Multi-Agent Collaboration: Peer/Swarm topology

Anchor scenario: support-ops triage with peer negotiation.

Preferred runtime in 2026:
- langgraph-swarm (if installed)
Fallback:
- explicit peer loop with critique/synthesis
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
class PeerState:
    problem: str
    analyst_notes: list[str] = field(default_factory=list)
    final_decision: str = ""


llm = ChatOpenAI(model=get_default_model(), temperature=0)


def peer_analysis(role: str, problem: str) -> str:
    return llm.invoke(
        f"You are {role}. Analyze this problem and propose one actionable recommendation:\n{problem}"
    ).content


def peer_critique(notes: list[str]) -> str:
    prompt = f"""
You are a peer reviewer. Critique these peer recommendations and resolve conflicts.

{notes}

Return: conflict_summary + merged recommendation.
"""
    return llm.invoke(prompt).content


def try_swarm_package() -> bool:
    try:
        import langgraph_swarm  # noqa: F401

        print("✅ langgraph-swarm detected. (This demo uses portable fallback loop.)")
        return True
    except Exception:
        print("ℹ️ langgraph-swarm not installed; using fallback peer loop.")
        return False


def run_peer_swarm(problem: str) -> PeerState:
    state = PeerState(problem=problem)
    roles = ["SRE analyst", "Support lead", "Product reliability manager"]

    for role in roles:
        state.analyst_notes.append(f"[{role}]\n{peer_analysis(role, problem)}")

    state.final_decision = peer_critique(state.analyst_notes)
    return state


def main() -> None:
    print("\n" + "=" * 80)
    print("MULTI-AGENT — PEER/SWARM")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    try_swarm_package()

    problem = "Multiple enterprise accounts report payment approval delays after release v2.8.4."
    state = run_peer_swarm(problem)

    print("\n🤝 Peer notes:")
    for note in state.analyst_notes:
        print(note)
        print("-" * 60)

    print("\n✅ Final peer decision:")
    print(state.final_decision)


if __name__ == "__main__":
    main()
