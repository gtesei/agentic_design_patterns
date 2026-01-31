"""
Resource Optimization Pattern: Basic Implementation

This example demonstrates fundamental optimization techniques:
- Response caching with LRU and TTL
- Prompt optimization to reduce token usage
- Model selection based on query complexity
- Metrics tracking for cost savings and performance

Problem: High-volume AI applications face expensive API costs and slow response times
Solution: Cache responses, optimize prompts, route to appropriate models
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import hashlib
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

# Initialize models
gpt4 = ChatOpenAI(model="gpt-4", temperature=0)
gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ===========================
# Response Cache Implementation
# ===========================

class ResponseCache:
    """LRU cache with TTL (Time-To-Live) for LLM responses"""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: dict[str, tuple[str, datetime]] = {}
        self.hits = 0
        self.misses = 0

    def _generate_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        key_string = f"{prompt}|{model}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        key = self._generate_key(prompt, model)

        if key in self.cache:
            response, timestamp = self.cache[key]

            # Check if expired
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                self.hits += 1
                return response
            else:
                # Remove expired entry
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, prompt: str, model: str, response: str):
        """Store response in cache"""
        key = self._generate_key(prompt, model)

        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (response, datetime.now())

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
        }


# ===========================
# Prompt Optimization
# ===========================

def optimize_prompt(prompt: str) -> str:
    """Reduce token usage while preserving semantic meaning"""

    # Remove filler words
    filler_words = ["please", "kindly", "just", "really", "very", "quite", "actually"]
    for word in filler_words:
        prompt = prompt.replace(f" {word} ", " ")
        prompt = prompt.replace(f" {word.capitalize()} ", " ")

    # Remove redundant phrases
    redundant_phrases = [
        "You are a helpful AI assistant. ",
        "Please provide a detailed answer. ",
        "Please provide a detailed and comprehensive answer. ",
        "Make sure your response is accurate. ",
        "I would like to know ",
        "Can you please tell me ",
        "I want to understand ",
    ]
    for phrase in redundant_phrases:
        prompt = prompt.replace(phrase, "")

    # Strip extra whitespace
    prompt = " ".join(prompt.split())

    return prompt


def calculate_token_estimate(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4


def calculate_cost(tokens: int, model: str) -> float:
    """Estimate cost based on token count and model"""
    cost_per_1k = {
        "gpt-4": 0.03,  # $0.03 per 1K tokens (average of input/output)
        "gpt-3.5-turbo": 0.0005,  # $0.0005 per 1K tokens
    }
    return (tokens / 1000) * cost_per_1k.get(model, 0.03)


# ===========================
# Model Selection
# ===========================

def analyze_complexity(query: str) -> str:
    """Analyze query complexity and return appropriate tier"""

    query_lower = query.lower()
    query_length = len(query.split())

    # Complex keywords that require GPT-4
    complex_keywords = [
        "analyze",
        "compare",
        "evaluate",
        "design",
        "architect",
        "critique",
        "debate",
        "argue",
    ]

    # Medium keywords that work well with GPT-3.5
    medium_keywords = ["explain", "how does", "why", "summarize", "describe", "discuss"]

    # Simple keywords that definitely work with GPT-3.5
    simple_keywords = ["what is", "define", "who is", "when did", "where is"]

    # Check complexity
    if any(kw in query_lower for kw in complex_keywords):
        return "complex"

    if any(kw in query_lower for kw in medium_keywords) and query_length > 10:
        return "medium"

    if any(kw in query_lower for kw in simple_keywords):
        return "simple"

    # Default based on length
    if query_length > 20:
        return "medium"

    return "simple"


def select_model(query: str) -> tuple[ChatOpenAI, str]:
    """Select most cost-effective model based on query complexity"""
    complexity = analyze_complexity(query)

    if complexity == "complex":
        return gpt4, "gpt-4"
    else:
        # Both simple and medium use GPT-3.5 for cost efficiency
        return gpt35, "gpt-3.5-turbo"


# ===========================
# Optimized LLM Call
# ===========================

class OptimizedLLM:
    """LLM wrapper with caching, prompt optimization, and model routing"""

    def __init__(self):
        self.cache = ResponseCache(ttl_seconds=3600, max_size=1000)
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.request_count = 0

    def invoke(self, prompt: str, use_cache: bool = True, optimize_prompt: bool = True) -> dict:
        """Execute LLM call with optimizations"""

        self.request_count += 1
        start_time = time.time()

        # Step 1: Optimize prompt if enabled
        original_prompt = prompt
        if optimize_prompt:
            prompt = optimize_prompt(prompt)
            token_savings = calculate_token_estimate(original_prompt) - calculate_token_estimate(prompt)
        else:
            token_savings = 0

        # Step 2: Select appropriate model
        model, model_name = select_model(prompt)

        # Step 3: Check cache if enabled
        cached_response = None
        if use_cache:
            cached_response = self.cache.get(prompt, model_name)

        if cached_response:
            # Cache hit
            latency = time.time() - start_time
            self.total_latency += latency

            return {
                "response": cached_response,
                "model": model_name,
                "latency": latency,
                "cost": 0.0,  # No cost for cached response
                "cached": True,
                "token_savings": token_savings,
                "prompt_optimized": optimize_prompt,
            }
        else:
            # Cache miss - call LLM
            response = model.invoke(prompt)
            response_content = response.content

            # Store in cache if enabled
            if use_cache:
                self.cache.set(prompt, model_name, response_content)

            latency = time.time() - start_time
            self.total_latency += latency

            # Estimate cost
            total_tokens = calculate_token_estimate(prompt + response_content)
            cost = calculate_cost(total_tokens, model_name)
            self.total_cost += cost

            return {
                "response": response_content,
                "model": model_name,
                "latency": latency,
                "cost": cost,
                "cached": False,
                "token_savings": token_savings,
                "prompt_optimized": optimize_prompt,
                "tokens": total_tokens,
            }

    def get_stats(self) -> dict:
        """Get optimization statistics"""
        cache_stats = self.cache.stats()

        return {
            "total_requests": self.request_count,
            "total_cost": self.total_cost,
            "avg_cost": self.total_cost / self.request_count if self.request_count > 0 else 0,
            "total_latency": self.total_latency,
            "avg_latency": self.total_latency / self.request_count if self.request_count > 0 else 0,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
        }


# ===========================
# Demonstration
# ===========================

def print_result(query: str, result: dict, request_num: int):
    """Pretty print optimization result"""
    print(f"\n{'='*80}")
    print(f"Request #{request_num}: {query}")
    print(f"{'='*80}")
    print(f"Model: {result['model']}")
    print(f"Cached: {'✅ YES' if result['cached'] else '❌ NO'}")
    print(f"Prompt Optimized: {'✅ YES' if result['prompt_optimized'] else '❌ NO'}")
    if result["token_savings"] > 0:
        print(f"Tokens Saved: {result['token_savings']}")
    print(f"Latency: {result['latency']*1000:.0f}ms")
    print(f"Cost: ${result['cost']:.4f}")
    print(f"\nResponse: {result['response'][:200]}...")
    print(f"{'='*80}\n")


def run_comparison():
    """Compare optimized vs non-optimized approaches"""

    print("\n" + "=" * 80)
    print(" " * 25 + "BASELINE (No Optimization)")
    print("=" * 80)

    # Baseline: No optimization
    baseline_llm = OptimizedLLM()
    baseline_queries = [
        "What is Python?",
        "What is Python?",  # Duplicate
        "Explain Python programming language",  # Similar
        "What is 2 + 2?",  # Simple
        "Compare Python and Java for enterprise applications",  # Complex
    ]

    for i, query in enumerate(baseline_queries, 1):
        result = baseline_llm.invoke(query, use_cache=False, optimize_prompt=False)
        print(f"Request #{i}: {query}")
        print(f"  Model: {result['model']} | Latency: {result['latency']*1000:.0f}ms | Cost: ${result['cost']:.4f}")

    baseline_stats = baseline_llm.get_stats()
    print(f"\nBaseline Total Cost: ${baseline_stats['total_cost']:.4f}")
    print(f"Baseline Avg Latency: {baseline_stats['avg_latency']*1000:.0f}ms")

    print("\n" + "=" * 80)
    print(" " * 20 + "OPTIMIZED (Caching + Routing + Compression)")
    print("=" * 80)

    # Optimized: Full optimization
    optimized_llm = OptimizedLLM()

    for i, query in enumerate(baseline_queries, 1):
        result = optimized_llm.invoke(query, use_cache=True, optimize_prompt=True)
        print(f"Request #{i}: {query}")
        print(
            f"  Model: {result['model']} | Cached: {'✅' if result['cached'] else '❌'} | "
            f"Latency: {result['latency']*1000:.0f}ms | Cost: ${result['cost']:.4f}"
        )

    optimized_stats = optimized_llm.get_stats()
    print(f"\nOptimized Total Cost: ${optimized_stats['total_cost']:.4f}")
    print(f"Optimized Avg Latency: {optimized_stats['avg_latency']*1000:.0f}ms")
    print(f"Cache Hit Rate: {optimized_stats['cache_hit_rate']:.1%}")

    # Calculate savings
    cost_savings = baseline_stats["total_cost"] - optimized_stats["total_cost"]
    cost_savings_pct = (cost_savings / baseline_stats["total_cost"]) * 100 if baseline_stats["total_cost"] > 0 else 0

    latency_improvement = baseline_stats["avg_latency"] - optimized_stats["avg_latency"]
    latency_improvement_pct = (
        (latency_improvement / baseline_stats["avg_latency"]) * 100 if baseline_stats["avg_latency"] > 0 else 0
    )

    print("\n" + "=" * 80)
    print(" " * 30 + "SAVINGS SUMMARY")
    print("=" * 80)
    print(f"Cost Reduction: ${cost_savings:.4f} ({cost_savings_pct:.1f}%)")
    print(f"Latency Improvement: {latency_improvement*1000:.0f}ms ({latency_improvement_pct:.1f}%)")
    print(f"Cache Hit Rate: {optimized_stats['cache_hit_rate']:.1%}")

    # Extrapolate to scale
    print("\n" + "=" * 80)
    print(" " * 25 + "SCALE PROJECTIONS")
    print("=" * 80)
    requests_per_day = 100000
    baseline_daily = (baseline_stats["total_cost"] / len(baseline_queries)) * requests_per_day
    optimized_daily = (optimized_stats["total_cost"] / len(baseline_queries)) * requests_per_day
    monthly_savings = (baseline_daily - optimized_daily) * 30

    print(f"Daily requests: {requests_per_day:,}")
    print(f"Baseline cost: ${baseline_daily:,.2f}/day = ${baseline_daily*30:,.2f}/month")
    print(f"Optimized cost: ${optimized_daily:,.2f}/day = ${optimized_daily*30:,.2f}/month")
    print(f"Monthly savings: ${monthly_savings:,.2f} ({cost_savings_pct:.1f}%)")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       Resource Optimization - Basic Implementation            ║
    ║                                                               ║
    ║  Demonstrates:                                                ║
    ║  • Response caching (LRU + TTL)                               ║
    ║  • Prompt optimization (token reduction)                      ║
    ║  • Model selection (cost-aware routing)                       ║
    ║  • Metrics tracking (cost, latency, cache hit rate)           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    run_comparison()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Key Takeaways                              ║
    ║                                                               ║
    ║  ✅ Caching eliminates redundant API calls                    ║
    ║  ✅ Model routing uses cheaper models when appropriate        ║
    ║  ✅ Prompt optimization reduces token usage                   ║
    ║  ✅ Combined techniques provide 40-80% cost reduction         ║
    ║  ✅ Latency improves 2-10x for cached responses               ║
    ║                                                               ║
    ║  Next: Try optimization_advanced.py for more techniques!      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
