"""
RAG (Retrieval-Augmented Generation) Pattern Implementations

This package provides two implementations of the RAG pattern:
- rag_basic: Simple RAG with Qdrant and sentence-transformers
- rag_advanced: Advanced RAG with MMR re-ranking and LangChain
"""

from .rag_basic import BasicRAG
from .rag_advanced import AdvancedRAG

__all__ = ["BasicRAG", "AdvancedRAG"]
