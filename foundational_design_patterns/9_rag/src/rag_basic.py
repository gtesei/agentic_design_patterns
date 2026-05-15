"""
RAG Pattern: Hybrid retrieval baseline (BM25-ish + dense + RRF + rerank)

Anchor scenario: support documentation assistant.
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class Chunk:
    id: str
    text: str
    source: str


class HybridRetriever:
    """Simple hybrid retriever with RRF fusion and lightweight reranking."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embeddings = OpenAIEmbeddings()
        self.chunk_vectors = self.embeddings.embed_documents([c.text for c in chunks])
        self.tokenized = [self._tokenize(c.text) for c in chunks]
        self.idf = self._build_idf(self.tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _build_idf(docs: list[list[str]]) -> dict[str, float]:
        df: dict[str, int] = {}
        n = len(docs)
        for doc in docs:
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1
        return {tok: math.log((n + 1) / (freq + 1)) + 1 for tok, freq in df.items()}

    @staticmethod
    def _dot(a: Iterable[float], b: Iterable[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _bm25ish_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        score = 0.0
        tf: dict[str, int] = {}
        for tok in doc_tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok in query_tokens:
            if tok in tf:
                score += (1 + math.log(tf[tok])) * self.idf.get(tok, 1.0)
        return score

    def _dense_rank(self, query: str) -> list[tuple[int, float]]:
        qv = self.embeddings.embed_query(query)
        scored = [(idx, self._dot(qv, dv)) for idx, dv in enumerate(self.chunk_vectors)]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _lexical_rank(self, query: str) -> list[tuple[int, float]]:
        q_tokens = self._tokenize(query)
        scored = [(idx, self._bm25ish_score(q_tokens, dt)) for idx, dt in enumerate(self.tokenized)]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        lexical = self._lexical_rank(query)
        dense = self._dense_rank(query)

        rrf_scores: dict[int, float] = {}
        k = 60.0
        for rank, (idx, _) in enumerate(lexical, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(dense, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: top_k * 2]

        query_terms = set(self._tokenize(query))
        reranked: list[tuple[Chunk, float]] = []
        for idx, base_score in fused:
            chunk = self.chunks[idx]
            overlap = len(query_terms.intersection(set(self.tokenized[idx])))
            reranked.append((chunk, base_score + 0.01 * overlap))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


def build_support_chunks() -> list[Chunk]:
    docs = [
        (
            "billing_guide",
            "Payment latency troubleshooting: check queue depth, retry backlog, and database write IOPS before rolling back.",
        ),
        (
            "incident_runbook",
            "For incident severity P1, page on-call immediately, assign incident commander, and publish customer status every 15 minutes.",
        ),
        (
            "sla_policy",
            "Enterprise tier customers require first response within 2 hours and escalation to senior support for payment pipeline incidents.",
        ),
        (
            "release_notes",
            "Release v2.8.4 introduced a new payment reconciliation worker with configurable concurrency and timeout settings.",
        ),
        (
            "db_observability",
            "If payment approval API latency exceeds 1.5 seconds p95, inspect lock contention and slow query logs.",
        ),
        (
            "ops_playbook",
            "During rollout failures, pause canary, compare error budgets, and decide rollback using incident commander approval.",
        ),
    ]

    return [Chunk(id=f"doc-{i}", source=src, text=text) for i, (src, text) in enumerate(docs, 1)]


def answer_query(query: str, retrieved: list[tuple[Chunk, float]]) -> str:
    llm = ChatOpenAI(model=get_default_model(), temperature=0)
    context = "\n\n".join([f"[{chunk.source}] {chunk.text}" for chunk, _ in retrieved])

    prompt = f"""
You are a support documentation assistant.
Use only the context below. If information is missing, say so.

Context:
{context}

Question: {query}
"""
    return llm.invoke(prompt).content


def main() -> None:
    print("\n" + "=" * 80)
    print("RAG BASIC — HYBRID RETRIEVAL (LEXICAL + DENSE + RRF)")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required for this demo.")
        return

    retriever = HybridRetriever(build_support_chunks())

    query = "How should we handle payment API latency for an enterprise customer during incident response?"
    retrieved = retriever.retrieve(query, top_k=5)

    print("\n🔍 Top-5 retrieved chunks:")
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"{i}. score={score:.4f} source={chunk.source}")
        print(f"   {chunk.text}")

    answer = answer_query(query, retrieved)
    print("\n📝 Final answer:")
    print(answer)


if __name__ == "__main__":
    main()
