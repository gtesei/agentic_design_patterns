"""
RAG Pattern: Hosted file search + local hybrid fallback

Anchor scenario: deep research/support docs assistant.

This script demonstrates:
- OpenAI hosted file_search tool usage (when configured)
- graceful fallback to local hybrid retrieval when hosted setup is unavailable
"""

from __future__ import annotations

import os
import sys

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_advanced_model

configure_example(__file__)

from langchain_openai import ChatOpenAI

from rag_basic import HybridRetriever, answer_query, build_support_chunks


def try_hosted_file_search(query: str) -> str | None:
    """Attempt OpenAI Responses API with built-in file_search tool.

    Requires OPENAI_VECTOR_STORE_ID to be set and accessible for the current key.
    """
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    if not vector_store_id:
        return None

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=get_advanced_model(),
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
            input=(
                "Use file_search on the support corpus and answer this question with citations: "
                f"{query}"
            ),
        )

        return response.output_text
    except Exception as exc:  # pragma: no cover - demo fallback
        print(f"⚠️ Hosted file_search unavailable: {exc}")
        return None


def local_hybrid_answer(query: str) -> str:
    retriever = HybridRetriever(build_support_chunks())
    retrieved = retriever.retrieve(query, top_k=5)
    return answer_query(query, retrieved)


def compare_models(query: str) -> tuple[str, str]:
    """Show that generation model can be swapped independently from retrieval."""
    retriever = HybridRetriever(build_support_chunks())
    retrieved = retriever.retrieve(query, top_k=5)

    context = "\n\n".join([f"[{c.source}] {c.text}" for c, _ in retrieved])

    low_cost = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    high_cap = ChatOpenAI(model=get_advanced_model(), temperature=0)

    prompt = f"Answer with 3 bullets and cite source tags.\n\nContext:\n{context}\n\nQuestion: {query}"
    return low_cost.invoke(prompt).content, high_cap.invoke(prompt).content


def main() -> None:
    print("\n" + "=" * 80)
    print("RAG ADVANCED — HOSTED FILE_SEARCH + LOCAL HYBRID FALLBACK")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    query = "What are recommended actions for a payment incident affecting enterprise customers?"

    hosted = try_hosted_file_search(query)
    if hosted:
        print("\n🌐 Hosted file_search answer:")
        print(hosted)
    else:
        print("\n📚 Hosted file_search not configured. Using local hybrid fallback...")
        print(local_hybrid_answer(query))

    print("\n🧪 Model comparison on the same retrieved context:")
    low, high = compare_models(query)
    print("\n- gpt-4o-mini output:\n", low)
    print("\n- advanced model output:\n", high)


if __name__ == "__main__":
    main()
