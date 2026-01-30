"""
Resource Optimization Pattern

This package implements resource optimization techniques for AI systems:
- Response caching (LRU, TTL, semantic)
- Prompt optimization (token reduction)
- Model selection and routing (cost-aware)
- Request batching (parallel processing)
- Predictive caching (pattern learning)
- Cost-aware quality management
"""

from .optimization_advanced import (
    AdvancedOptimizedLLM,
    AdaptiveBatchProcessor,
    CostAwareRouter,
    PredictiveCache,
)
from .optimization_basic import OptimizedLLM, ResponseCache

__all__ = [
    "OptimizedLLM",
    "ResponseCache",
    "AdvancedOptimizedLLM",
    "PredictiveCache",
    "AdaptiveBatchProcessor",
    "CostAwareRouter",
]
