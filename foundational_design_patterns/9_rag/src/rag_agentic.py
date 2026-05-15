"""
RAG Pattern: Agentic loop (query rewrite -> retrieve -> grade -> fallback -> generate -> self-check)

Anchor scenario: deep research/support documentation.
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

from rag_basic import HybridRetriever, build_support_chunks


@dataclass
class AgenticRAGState:
    question: str
    rewritten_query: str = ""
    retrieved_context: list[str] = field(default_factory=list)
    context_grade: str = "insufficient"
    fallback_notes: str = ""
    draft_answer: str = ""
    final_answer: str = ""


llm = ChatOpenAI(model=get_default_model(), temperature=0)
retriever = HybridRetriever(build_support_chunks())


def rewrite_query(state: AgenticRAGState) -> AgenticRAGState:
    prompt = (
        "Rewrite this support query to maximize retrieval quality. Keep key entities and intent.\n\n"
        f"Query: {state.question}"
    )
    state.rewritten_query = llm.invoke(prompt).content.strip()
    return state


def retrieve(state: AgenticRAGState) -> AgenticRAGState:
    retrieved = retriever.retrieve(state.rewritten_query or state.question, top_k=5)
    state.retrieved_context = [f"[{c.source}] {c.text}" for c, _ in retrieved]
    return state


def grade_context(state: AgenticRAGState) -> AgenticRAGState:
    context = "\n".join(state.retrieved_context)
    prompt = f"""
You are grading retrieval quality.
Question: {state.question}
Context:
{context}

Return one word only: sufficient OR insufficient.
"""
    verdict = llm.invoke(prompt).content.strip().lower()
    state.context_grade = "sufficient" if "sufficient" in verdict and "insufficient" not in verdict else "insufficient"
    return state


def maybe_fallback(state: AgenticRAGState) -> AgenticRAGState:
    if state.context_grade == "sufficient":
        state.fallback_notes = "fallback not required"
        return state

    state.fallback_notes = (
        "Context was insufficient. In production this branch would trigger web search, "
        "ticket history retrieval, or another corpus."
    )
    return state


def generate(state: AgenticRAGState) -> AgenticRAGState:
    context = "\n\n".join(state.retrieved_context)
    prompt = f"""
Answer the question using provided context and fallback notes.

Question: {state.question}
Fallback notes: {state.fallback_notes}
Context:
{context}

Constraints:
- cite sources in [source] format
- if uncertain, explicitly say what is missing
"""
    state.draft_answer = llm.invoke(prompt).content
    return state


def self_check(state: AgenticRAGState) -> AgenticRAGState:
    prompt = f"""
Perform a quick self-check of the answer below.
If there are unsupported claims, revise to be more cautious.

Answer:
{state.draft_answer}
"""
    state.final_answer = llm.invoke(prompt).content
    return state


def after_grade(state: AgenticRAGState) -> Literal["fallback", "generate"]:
    return "fallback" if state.context_grade == "insufficient" else "generate"


def build_graph():
    graph = StateGraph(AgenticRAGState)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade_context)
    graph.add_node("fallback", maybe_fallback)
    graph.add_node("generate", generate)
    graph.add_node("self_check", self_check)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, {"fallback": "fallback", "generate": "generate"})
    graph.add_edge("fallback", "generate")
    graph.add_edge("generate", "self_check")
    graph.add_edge("self_check", END)
    return graph.compile()


def main() -> None:
    print("\n" + "=" * 80)
    print("RAG AGENTIC — CRAG/SELF-RAG STYLE LOOP")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    app = build_graph()
    question = "We see payment API latency for enterprise customers after a release. What should support do first?"
    result = app.invoke(AgenticRAGState(question=question))

    # LangGraph typically returns a dict-like state snapshot after invocation.
    if isinstance(result, dict):
        rewritten_query = result.get("rewritten_query", "")
        retrieved_context = result.get("retrieved_context", [])
        context_grade = result.get("context_grade", "unknown")
        fallback_notes = result.get("fallback_notes", "")
        final_answer = result.get("final_answer", "")
    else:
        rewritten_query = getattr(result, "rewritten_query", "")
        retrieved_context = getattr(result, "retrieved_context", [])
        context_grade = getattr(result, "context_grade", "unknown")
        fallback_notes = getattr(result, "fallback_notes", "")
        final_answer = getattr(result, "final_answer", "")

    print("\n🔄 Rewritten query:")
    print(rewritten_query)

    print("\n📚 Retrieved context:")
    for line in retrieved_context:
        print("-", line)

    print("\n🧪 Context grade:", context_grade)
    print("🛟 Fallback:", fallback_notes)

    print("\n✅ Final answer:")
    print(final_answer)


if __name__ == "__main__":
    main()
