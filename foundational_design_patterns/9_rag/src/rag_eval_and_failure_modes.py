"""
RAG Pattern: Evaluation and failure modes walkthrough.

Anchor scenario: support docs + incident runbooks.

This script demonstrates common retrieval failures:
- lexical mismatch
- missing corpus coverage
- conflicting sources
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example

configure_example(__file__)

from rag_basic import HybridRetriever, build_support_chunks


@dataclass
class EvalCase:
    name: str
    question: str
    expected_keywords: list[str]


def recall_at_k(retrieved_text: str, expected_keywords: list[str]) -> float:
    hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text.lower())
    return hits / max(1, len(expected_keywords))


def main() -> None:
    print("\n" + "=" * 80)
    print("RAG EVAL — FAILURE MODES")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required (retriever uses OpenAI embeddings).")
        return

    retriever = HybridRetriever(build_support_chunks())

    cases = [
        EvalCase(
            name="normal_case",
            question="How do we handle payment API latency for enterprise customers?",
            expected_keywords=["enterprise", "incident", "latency", "escalation"],
        ),
        EvalCase(
            name="lexical_mismatch",
            question="What should we do about checkout sluggishness in premium accounts?",
            expected_keywords=["enterprise", "payment", "latency"],
        ),
        EvalCase(
            name="missing_coverage",
            question="Which Kafka partition key should we use for ledger sharding?",
            expected_keywords=["kafka", "partition", "ledger"],
        ),
    ]

    for case in cases:
        retrieved = retriever.retrieve(case.question, top_k=5)
        merged = "\n".join(chunk.text for chunk, _ in retrieved)
        metric = recall_at_k(merged, case.expected_keywords)

        print(f"\n🧪 Case: {case.name}")
        print(f"Question: {case.question}")
        print(f"Recall@5 (keyword proxy): {metric:.2f}")

        if metric < 0.5:
            print("⚠️ Likely failure mode detected.")
            print("   Mitigations: query rewrite, corpus expansion, reranker tuning, web fallback.")

        print("Top sources:")
        for idx, (chunk, score) in enumerate(retrieved, 1):
            print(f"  {idx}. {chunk.source} (score={score:.4f})")


if __name__ == "__main__":
    main()
