# Resource Optimization

## Overview

The **Resource Optimization Pattern** focuses on systematically reducing costs, minimizing latency, and maximizing compute efficiency in AI agent systems through intelligent caching, request batching, model selection, and prompt optimization. This pattern transforms expensive, slow AI applications into cost-effective, high-performance systems without sacrificing quality.

Unlike traditional performance optimization that focuses on code-level improvements, AI resource optimization must balance multiple dimensions: API costs (tokens, requests), latency (response time, throughput), compute efficiency (batching, parallelization), and quality (accuracy, relevance, user satisfaction). The key challenge is optimizing these trade-offs dynamically based on workload characteristics and business constraints.

## Why Use This Pattern?

AI systems face unique resource challenges:

- **High operational costs**: LLM API calls can cost $0.001-$0.10 per request, adding up quickly at scale
- **Variable latency**: Response times range from hundreds of milliseconds to several seconds
- **Token inefficiency**: Redundant prompts, verbose responses, and unnecessary context waste tokens
- **Request redundancy**: Similar queries repeated frequently without reuse
- **Suboptimal model selection**: Using expensive models when cheaper alternatives suffice
- **Sequential processing**: Processing requests one-at-a-time when batching would be faster

This pattern solves these by:
- **Intelligent caching**: Store and reuse responses for identical or semantically similar queries
- **Response caching with TTL**: Time-based invalidation for time-sensitive data
- **Prompt optimization**: Compress prompts while preserving semantic meaning
- **Dynamic model routing**: Select the most cost-effective model based on query complexity
- **Request batching**: Group multiple requests for parallel processing
- **Predictive prefetching**: Anticipate and cache likely future requests
- **Cost-aware decision making**: Balance quality vs. cost trade-offs intelligently

### Example: High-Volume Chatbot Without vs. With Optimization

```
Without Optimization:
User 1: "What's Python?"
→ GPT-4 call: 800ms, $0.015
User 2: "What's Python?"
→ GPT-4 call: 850ms, $0.015 (duplicate!)
User 3: "Explain Python programming"
→ GPT-4 call: 900ms, $0.018 (similar query!)
User 4: "What's 2+2?"
→ GPT-4 call: 700ms, $0.012 (overkill for simple task!)

Total: 3250ms, $0.060 for 4 requests
Daily @ 100K requests: $1,500/day = $45K/month

With Resource Optimization:
User 1: "What's Python?"
→ GPT-4 call: 800ms, $0.015 (cache miss)
→ ✅ Cached response (TTL: 1 hour)

User 2: "What's Python?"
→ Cache hit: 50ms, $0.000 (95% faster, 100% cheaper!)

User 3: "Explain Python programming"
→ Semantic similarity detected (0.94)
→ Cache hit with rewrite: 150ms, $0.002 (adapted from User 1)

User 4: "What's 2+2?"
→ Routed to GPT-3.5-turbo: 300ms, $0.002 (complexity analysis)

Total: 1300ms, $0.019 for 4 requests (60% faster, 68% cheaper)
Daily @ 100K requests: Cache hit rate 65% → $525/day = $15.7K/month
Monthly savings: $29.3K (65% cost reduction!)
```

## How It Works

The optimization pipeline follows a structured decision flow:

1. **Measure**: Profile current performance (latency, cost, cache hit rate, token usage)
2. **Analyze**: Identify optimization opportunities (repeated queries, slow responses, high costs)
3. **Optimize**: Apply techniques (caching, prompt compression, model routing, batching)
4. **Validate**: Ensure quality maintained (compare outputs, measure user satisfaction)
5. **Iterate**: Continuously refine based on metrics and feedback

```
┌──────────────────────────────────────────────────────┐
│                  User Request                         │
└────────────────────┬─────────────────────────────────┘
                     ↓
            ┌────────────────┐
            │  Cache Lookup   │ Check if response cached
            └────────┬───────┘
                     ↓
              ┌─────────────┐
              │ Cache Hit?  │
              └──┬──────┬───┘
         Yes    ↓      ↓ No
        ┌───────────┐  │
        │  Return    │  │
        │  Cached    │  │
        │  Response  │  │
        └────────────┘  │
                        ↓
               ┌────────────────┐
               │ Query Analysis  │ Analyze complexity
               └────────┬───────┘
                        ↓
               ┌────────────────┐
               │ Model Selection │ Route to appropriate model
               └────────┬───────┘
                        ↓
               ┌────────────────┐
               │ Prompt         │ Compress/optimize prompt
               │ Optimization   │
               └────────┬───────┘
                        ↓
               ┌────────────────┐
               │ Batch Check    │ Can batch with others?
               └────────┬───────┘
                        ↓
               ┌────────────────┐
               │ LLM Execution  │ Execute request
               └────────┬───────┘
                        ↓
               ┌────────────────┐
               │ Cache Storage  │ Store for future use
               └────────┬───────┘
                        ↓
               ┌────────────────┐
               │ Response       │ Return to user
               └────────────────┘
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **High-volume applications**: Chatbots, customer support, content generation at scale
- **Cost-sensitive scenarios**: Startups, prototypes, high-traffic services with tight budgets
- **Performance-critical apps**: Real-time responses, interactive experiences, low-latency requirements
- **Repetitive queries**: FAQ systems, documentation assistants, common question answering
- **Variable workloads**: Traffic spikes, batch processing, async workflows
- **Multi-tier systems**: Different quality/cost requirements for different user segments
- **Long-running operations**: Background processing, bulk data analysis, report generation

### ❌ When NOT to Use

- **Low-volume applications**: < 1000 requests/day where optimization overhead exceeds savings
- **Unique queries**: Every request completely different (no caching benefit)
- **Real-time data requirements**: Stock prices, live sports scores (caching counterproductive)
- **Highest quality mandatory**: Medical diagnosis, legal advice (no quality compromise acceptable)
- **Simple single-shot tasks**: One-off scripts, personal projects, development testing

## Rule of Thumb

**Use Resource Optimization when:**
1. Monthly LLM costs exceed **$500** or latency impacts user experience
2. **Cache hit rate** potential > 20% (queries have repetition or similarity)
3. You can trade **slight quality reduction** for significant cost/speed gains
4. System has **mixed complexity** queries (some simple, some complex)
5. You need **predictable costs** and performance at scale

**Don't use Resource Optimization when:**
1. Application is low-volume or development-only
2. Every query is unique and time-sensitive
3. Quality is non-negotiable (safety-critical systems)
4. Complexity overhead outweighs benefits

## Core Components

### 1. Response Caching

Store and reuse LLM outputs to avoid redundant API calls:

**Cache Key Strategies:**
- **Exact match**: Hash of complete input (prompt + parameters)
- **Semantic similarity**: Embedding-based similarity matching
- **Normalized matching**: Case-insensitive, whitespace-normalized

**Cache Policies:**
- **TTL (Time-To-Live)**: Expire cached responses after N minutes/hours
- **LRU (Least Recently Used)**: Evict oldest entries when cache is full
- **Size-based**: Limit by memory usage or entry count

**Implementation:**
```python
from functools import lru_cache
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: dict = {}

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return response  # Cache hit
            else:
                del self.cache[key]  # Expired
        return None  # Cache miss

    def set(self, key: str, response: str):
        if len(self.cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (response, datetime.now())
```

### 2. Prompt Optimization

Reduce token usage while maintaining semantic meaning:

**Techniques:**
- **Redundancy removal**: Strip unnecessary words, phrases, repetition
- **Template compression**: Shorten instruction templates
- **Context pruning**: Keep only relevant context, remove boilerplate
- **Output length control**: Request concise responses explicitly

**Example:**
```python
# Original (verbose): 85 tokens
prompt = """You are a helpful AI assistant. Please provide a detailed and comprehensive answer to the following question. Make sure your response is accurate and informative. Question: What is Python?"""

# Optimized (concise): 12 tokens
prompt_optimized = "Explain Python programming language concisely."

# Token savings: 73 tokens (86% reduction)
# Quality impact: Minimal (answer remains accurate)
```

### 3. Model Selection & Routing

Choose the most cost-effective model based on query complexity:

**Complexity Tiers:**
- **Simple** (Tier 1): Factual questions, basic math, definitions → GPT-3.5-turbo ($0.0005/1K tokens)
- **Medium** (Tier 2): Analysis, summarization, explanations → GPT-4o-mini ($0.003/1K tokens)
- **Complex** (Tier 3): Reasoning, creative tasks, code generation → GPT-4 ($0.03/1K tokens)

**Routing Logic:**
```python
def route_to_model(query: str) -> str:
    complexity_score = analyze_complexity(query)

    if complexity_score < 3:
        return "gpt-3.5-turbo"  # 60x cheaper than GPT-4
    elif complexity_score < 7:
        return "gpt-4o-mini"    # 10x cheaper than GPT-4
    else:
        return "gpt-4"          # Most capable, highest cost

def analyze_complexity(query: str) -> int:
    """Score query complexity 1-10 based on:
    - Length, keywords, sentence structure
    - Requires reasoning? Multiple steps? Domain expertise?
    """
    score = 0
    score += len(query.split()) // 10  # Length factor
    if any(kw in query.lower() for kw in ["why", "how", "analyze", "explain"]):
        score += 3  # Reasoning required
    if any(kw in query.lower() for kw in ["compare", "evaluate", "design"]):
        score += 4  # Complex reasoning
    return min(score, 10)
```

### 4. Request Batching

Group multiple requests for parallel processing:

**Benefits:**
- Reduced API overhead (1 HTTP request instead of N)
- Parallelization (process multiple queries simultaneously)
- Throughput optimization (higher queries/second)

**Implementation:**
```python
from typing import List
import asyncio

async def batch_process(queries: List[str], batch_size: int = 10):
    """Process queries in batches for efficiency"""
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        # Process batch in parallel
        batch_results = await asyncio.gather(
            *[process_query_async(q) for q in batch]
        )
        results.extend(batch_results)
    return results
```

### 5. Metrics Tracking

Monitor optimization effectiveness:

**Key Metrics:**
- **Cost per request**: Average API cost per query
- **Cache hit rate**: % of requests served from cache
- **Latency P50/P95/P99**: Response time percentiles
- **Token usage**: Input/output tokens per request
- **Model distribution**: % requests per model tier
- **Cost savings**: Total savings vs. baseline (no optimization)

## Implementation Approaches

### Approach 1: Basic Caching with LRU

Simple memory-based caching for immediate cost reduction:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_llm_call(prompt: str, model: str = "gpt-4") -> str:
    """Cache LLM responses in memory"""
    return llm.invoke(prompt, model=model)

# Usage
response1 = cached_llm_call("What is Python?")  # Cache miss → API call
response2 = cached_llm_call("What is Python?")  # Cache hit → instant return
```

**Pros**: Easy to implement, zero dependencies, works immediately
**Cons**: Memory-only (no persistence), no TTL, exact match only

### Approach 2: Semantic Caching with Embeddings

Match semantically similar queries for higher cache hit rate:

```python
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.embeddings = OpenAIEmbeddings()
        self.cache = []  # [(embedding, query, response), ...]
        self.threshold = similarity_threshold

    def get(self, query: str) -> Optional[str]:
        query_embedding = self.embeddings.embed_query(query)

        for cached_embedding, cached_query, cached_response in self.cache:
            similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
            if similarity >= self.threshold:
                return cached_response  # Semantic match!
        return None

    def set(self, query: str, response: str):
        query_embedding = self.embeddings.embed_query(query)
        self.cache.append((query_embedding, query, response))
```

**Pros**: Higher cache hit rate, handles variations, more flexible
**Cons**: Embedding cost, similarity computation overhead, more complex

### Approach 3: Dynamic Model Routing

Route queries to appropriate models based on complexity:

```python
from langchain_openai import ChatOpenAI

class ModelRouter:
    def __init__(self):
        self.models = {
            "fast": ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            "balanced": ChatOpenAI(model="gpt-4o-mini", temperature=0),
            "powerful": ChatOpenAI(model="gpt-4", temperature=0),
        }

    def route_and_execute(self, query: str) -> tuple[str, str, float]:
        """Route query to appropriate model, return (response, model, cost)"""
        complexity = self.analyze_complexity(query)

        if complexity == "simple":
            model_name = "fast"
            cost_multiplier = 1.0
        elif complexity == "medium":
            model_name = "balanced"
            cost_multiplier = 6.0
        else:
            model_name = "powerful"
            cost_multiplier = 60.0

        model = self.models[model_name]
        response = model.invoke(query)

        # Estimate cost
        tokens = len(query.split()) + len(response.content.split())
        cost = (tokens / 1000) * 0.0005 * cost_multiplier

        return response.content, model_name, cost
```

### Approach 4: Request Batching Pipeline

Accumulate and process requests in batches:

```python
import asyncio
from collections import deque
from typing import List, Tuple

class BatchProcessor:
    def __init__(self, batch_size: int = 10, wait_ms: int = 100):
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        self.queue = deque()
        self.results = {}

    async def submit(self, query_id: str, query: str) -> str:
        """Submit query and wait for batch processing"""
        future = asyncio.Future()
        self.queue.append((query_id, query, future))

        # Trigger batch processing if queue is full
        if len(self.queue) >= self.batch_size:
            await self.process_batch()

        return await future

    async def process_batch(self):
        """Process accumulated queries in parallel"""
        if not self.queue:
            return

        batch = [self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))]

        # Execute all queries in parallel
        responses = await asyncio.gather(
            *[llm_call_async(query) for _, query, _ in batch]
        )

        # Resolve futures
        for (query_id, query, future), response in zip(batch, responses):
            future.set_result(response)
```

## Key Benefits

### 💰 Cost Reduction

**Impact**: 40-80% reduction in LLM API costs

**How**:
- Caching eliminates redundant API calls (50-70% hit rate typical)
- Model routing uses cheaper models when appropriate (10-60x cost difference)
- Prompt optimization reduces token usage (20-40% reduction)
- Batching reduces per-request overhead

**Example**: $45K/month → $12K/month (73% savings)

### ⚡ Faster Response Times

**Impact**: 2-10x latency improvement

**How**:
- Cache hits return in < 100ms vs. 500-2000ms for API calls
- Cheaper models respond faster (GPT-3.5: 300ms vs. GPT-4: 800ms)
- Batching increases throughput
- Predictive prefetching serves likely queries instantly

**Example**: P95 latency: 2.5s → 450ms (5.5x faster)

### 📈 Better User Experience

**Impact**: Higher satisfaction, lower bounce rates

**How**:
- Faster responses keep users engaged
- Consistent performance (cache = predictable latency)
- Higher throughput handles traffic spikes
- Cost savings enable more features

### 🌱 Sustainability & Efficiency

**Impact**: Reduced computational waste, lower carbon footprint

**How**:
- Fewer redundant computations
- Right-sized model usage (no overkill)
- Better resource utilization

## Trade-offs

### ⚠️ Complexity Overhead

**Issue**: Additional code, dependencies, and mental model complexity

**Impact**: Higher development and maintenance costs

**Mitigation**:
- Start simple (basic caching only)
- Add optimizations incrementally as needed
- Use well-tested libraries (avoid reinventing)
- Document optimization logic clearly
- Monitor to ensure optimizations actually help

### 🎯 Quality vs. Cost Trade-off

**Issue**: Cheaper models or cached responses may reduce output quality

**Impact**: Some users may receive less accurate or less detailed responses

**Mitigation**:
- A/B test quality impact before full rollout
- Use quality metrics (LLM-as-judge, user feedback)
- Allow user tier selection (fast vs. accurate modes)
- Set quality thresholds (fallback to better model if confidence low)
- Monitor user satisfaction scores

### ⏱️ Cache Staleness

**Issue**: Cached responses may become outdated or incorrect

**Impact**: Users receive old information, hurting trust and accuracy

**Mitigation**:
- Use appropriate TTL for data freshness requirements
- Implement cache invalidation strategies
- Add timestamps to cached responses
- Monitor cache age distribution
- Clear cache on model updates or data changes

### 🔄 Cold Start Performance

**Issue**: First requests (cache misses) are slower and more expensive

**Impact**: Initial users or new query types have worse experience

**Mitigation**:
- Pre-warm cache with common queries
- Implement cache priming during low-traffic periods
- Use predictive prefetching for anticipated queries
- Gradual rollout (build cache before announcing feature)

## Best Practices

### 1. Start with Measurement

```python
# Profile before optimizing
import time

def profile_request(query: str):
    start = time.time()
    response = llm.invoke(query)
    latency = time.time() - start

    tokens = estimate_tokens(query, response)
    cost = calculate_cost(tokens, model="gpt-4")

    print(f"Latency: {latency:.3f}s | Tokens: {tokens} | Cost: ${cost:.4f}")
    return response

# Establish baseline metrics
baseline_metrics = {
    "avg_latency": 1.2,  # seconds
    "avg_cost": 0.015,   # dollars
    "p95_latency": 2.5,
}
```

### 2. Implement Caching First

```python
# Cache is the highest ROI optimization
from functools import lru_cache
import hashlib

def cache_key(prompt: str, **kwargs) -> str:
    """Generate cache key from prompt and parameters"""
    key_string = f"{prompt}|{sorted(kwargs.items())}"
    return hashlib.md5(key_string.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_llm_call(cache_key: str) -> str:
    # Actual LLM call
    return llm.invoke(extract_prompt_from_key(cache_key))

# Usage
key = cache_key("What is Python?", model="gpt-4", temperature=0)
response = cached_llm_call(key)  # Cached on subsequent calls
```

### 3. Monitor Cache Performance

```python
class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_latency_saved = 0
        self.total_cost_saved = 0

    def record_hit(self, saved_latency: float, saved_cost: float):
        self.hits += 1
        self.total_latency_saved += saved_latency
        self.total_cost_saved += saved_cost

    def record_miss(self):
        self.misses += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def report(self):
        print(f"Cache Hit Rate: {self.hit_rate:.1%}")
        print(f"Total Requests: {self.hits + self.misses}")
        print(f"Latency Saved: {self.total_latency_saved:.1f}s")
        print(f"Cost Saved: ${self.total_cost_saved:.2f}")
```

### 4. Use Tiered Model Selection

```python
COMPLEXITY_KEYWORDS = {
    "simple": ["what is", "define", "who is", "when did"],
    "medium": ["explain", "how does", "why", "summarize"],
    "complex": ["analyze", "compare", "evaluate", "design", "create"],
}

def select_model(query: str) -> str:
    """Select most cost-effective model for query"""
    query_lower = query.lower()

    # Check for complex keywords
    if any(kw in query_lower for kw in COMPLEXITY_KEYWORDS["complex"]):
        return "gpt-4"

    # Check for medium keywords
    if any(kw in query_lower for kw in COMPLEXITY_KEYWORDS["medium"]):
        return "gpt-4o-mini"

    # Default to fast model
    return "gpt-3.5-turbo"
```

### 5. Optimize Prompts Systematically

```python
def optimize_prompt(prompt: str) -> str:
    """Reduce token usage while preserving meaning"""

    # Remove filler words
    filler_words = ["please", "kindly", "just", "really", "very", "quite"]
    for word in filler_words:
        prompt = prompt.replace(f" {word} ", " ")

    # Remove redundant phrases
    redundant_phrases = [
        "You are a helpful AI assistant. ",
        "Please provide a detailed answer. ",
        "Make sure your response is accurate. ",
    ]
    for phrase in redundant_phrases:
        prompt = prompt.replace(phrase, "")

    # Strip extra whitespace
    prompt = " ".join(prompt.split())

    return prompt

# Example
original = "You are a helpful AI assistant. Please provide a detailed and accurate answer. What is Python?"
optimized = optimize_prompt(original)
print(f"Original: {len(original.split())} tokens")
print(f"Optimized: {len(optimized.split())} tokens")
print(f"Savings: {(1 - len(optimized.split())/len(original.split())):.1%}")
```

### 6. Implement Gradual Rollout

```python
def optimization_enabled(user_id: str, feature: str) -> bool:
    """Gradual rollout with A/B testing"""
    rollout_percentage = {
        "caching": 100,        # Fully rolled out
        "model_routing": 50,   # 50% of users
        "prompt_optimization": 10,  # 10% testing
    }

    # Deterministic assignment based on user_id hash
    user_hash = hash(user_id) % 100
    return user_hash < rollout_percentage.get(feature, 0)

# Usage
if optimization_enabled(user_id, "model_routing"):
    model = select_optimal_model(query)
else:
    model = "gpt-4"  # Default
```

## Performance Metrics

Track these metrics to measure optimization effectiveness:

### Cost Metrics
- **Cost per request**: Average API cost per query (target: 40-60% reduction)
- **Daily/monthly spend**: Total LLM costs over time
- **Cost by model**: Distribution of costs across model tiers
- **Cost savings**: Absolute and percentage savings vs. baseline

### Latency Metrics
- **P50/P95/P99 latency**: Response time percentiles (target: P95 < 1 second)
- **Cache hit latency**: Average response time for cache hits (target: < 100ms)
- **Cache miss latency**: Average response time for cache misses
- **Model-specific latency**: Response times by model tier

### Cache Metrics
- **Cache hit rate**: % of requests served from cache (target: > 30%)
- **Cache size**: Number of entries, memory usage
- **Cache age**: Distribution of entry ages (detect staleness)
- **Semantic hit rate**: % of queries matching via similarity (if using semantic cache)

### Quality Metrics
- **Output quality score**: LLM-as-judge or human evaluation (target: maintain > 90% of baseline)
- **User satisfaction**: Ratings, thumbs up/down, feedback
- **Task success rate**: % of queries achieving user goal
- **Quality by model**: Compare outputs across model tiers

### Efficiency Metrics
- **Tokens per request**: Input + output tokens (target: 20-40% reduction)
- **Requests per second**: Throughput (target: increase with batching)
- **Model distribution**: % queries routed to each model tier
- **Batch utilization**: % of requests processed in batches

## Example Scenarios

### Scenario 1: Customer Support Chatbot

**Problem**: 100K daily queries, $1,500/day LLM costs, 2.5s P95 latency

**Analysis**:
- 60% of queries are FAQ-style (highly repetitive)
- 80% of queries are simple (definitions, status checks)
- All queries use GPT-4 ($0.03/1K tokens)

**Optimization Strategy**:
1. **Caching**: LRU cache with 1-hour TTL for FAQ responses
2. **Model routing**: Route simple queries to GPT-3.5-turbo
3. **Prompt optimization**: Compress system prompts by 40%

**Results**:
- Cache hit rate: 58%
- Model distribution: 75% GPT-3.5, 20% GPT-4o-mini, 5% GPT-4
- Cost: $525/day (65% reduction, $29K/month savings)
- P95 latency: 600ms (76% improvement)
- Quality: 94% user satisfaction (vs. 96% baseline, acceptable)

### Scenario 2: Content Generation Pipeline

**Problem**: 10K article summaries/day, sequential processing, 4 hours total time

**Analysis**:
- Each summary takes ~1.5 seconds
- Articles are independent (no dependencies)
- High token usage due to verbose prompts

**Optimization Strategy**:
1. **Batching**: Process 50 articles in parallel batches
2. **Prompt optimization**: Reduce system prompt from 200 to 50 tokens
3. **Model selection**: Use GPT-4o-mini for summaries (sufficient quality)

**Results**:
- Processing time: 4 hours → 25 minutes (9.6x faster)
- Cost per article: $0.025 → $0.008 (68% reduction)
- Token usage: 800 → 500 tokens/article (37% reduction)
- Quality: 91% comparable to GPT-4 outputs (blind evaluation)

### Scenario 3: Multi-Language Translation Service

**Problem**: Translate 50K phrases/day, high costs, variable quality

**Analysis**:
- Many repeated phrases (common UI strings)
- Simple phrases use expensive GPT-4 unnecessarily
- No caching (translations re-done daily)

**Optimization Strategy**:
1. **Semantic caching**: Cache translations with similarity matching
2. **Tiered models**: Simple phrases → GPT-3.5, complex → GPT-4
3. **Batch processing**: Group translations by language for efficiency

**Results**:
- Cache hit rate: 72% (many repeated phrases)
- Cost: $750/day → $180/day (76% reduction)
- P95 latency: 1.8s → 350ms (5x faster for cached)
- Quality: BLEU score maintained at 0.89

## Advanced Patterns

### 1. Predictive Caching

Anticipate and pre-cache likely future queries:

```python
class PredictiveCache:
    def __init__(self):
        self.query_history = []
        self.pattern_cache = {}

    def record_query(self, query: str):
        """Track query patterns"""
        self.query_history.append(query)

        # Detect patterns (e.g., "What is X?" often followed by "How does X work?")
        if len(self.query_history) >= 2:
            prev_query = self.query_history[-2]
            self.learn_pattern(prev_query, query)

    def learn_pattern(self, query1: str, query2: str):
        """Learn query sequences"""
        if query1 not in self.pattern_cache:
            self.pattern_cache[query1] = {}

        if query2 not in self.pattern_cache[query1]:
            self.pattern_cache[query1][query2] = 0

        self.pattern_cache[query1][query2] += 1

    def prefetch(self, current_query: str):
        """Pre-cache likely next queries"""
        if current_query in self.pattern_cache:
            likely_next = max(
                self.pattern_cache[current_query].items(),
                key=lambda x: x[1]
            )[0]
            # Prefetch in background
            asyncio.create_task(cache_query_async(likely_next))
```

### 2. Adaptive Batching

Dynamically adjust batch size based on load:

```python
class AdaptiveBatcher:
    def __init__(self):
        self.min_batch_size = 5
        self.max_batch_size = 50
        self.current_batch_size = 10
        self.queue = deque()

    async def submit(self, query: str):
        self.queue.append(query)

        # Adjust batch size based on queue length
        queue_length = len(self.queue)
        if queue_length > 100:
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 5)
        elif queue_length < 20:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 2)

        # Process when batch is ready
        if len(self.queue) >= self.current_batch_size:
            await self.process_batch()
```

### 3. Cost-Aware Quality Thresholds

Balance cost and quality dynamically:

```python
class CostAwareRouter:
    def __init__(self, budget_per_request: float = 0.01):
        self.budget = budget_per_request
        self.quality_threshold = 0.85

    async def execute_with_fallback(self, query: str):
        """Try cheap model first, fallback to expensive if quality insufficient"""

        # Try cheap model
        cheap_response = await llm_call_async(query, model="gpt-3.5-turbo")
        cheap_cost = 0.002

        # Evaluate quality
        quality_score = await evaluate_quality_async(query, cheap_response)

        if quality_score >= self.quality_threshold:
            return cheap_response, cheap_cost, "gpt-3.5-turbo"

        # Quality insufficient, try better model
        if cheap_cost + 0.015 <= self.budget:  # Within budget
            better_response = await llm_call_async(query, model="gpt-4")
            return better_response, cheap_cost + 0.015, "gpt-4"

        # Over budget, return cheap response
        return cheap_response, cheap_cost, "gpt-3.5-turbo (budget limit)"
```

### 4. Multi-Tier Caching

Layer multiple cache strategies for optimal hit rate:

```python
class MultiTierCache:
    def __init__(self):
        self.l1_cache = {}  # Exact match, in-memory
        self.l2_cache = SemanticCache(threshold=0.95)  # High similarity
        self.l3_cache = SemanticCache(threshold=0.85)  # Medium similarity

    async def get(self, query: str) -> Optional[str]:
        # L1: Exact match (fastest)
        if query in self.l1_cache:
            return self.l1_cache[query]

        # L2: High similarity (fast)
        response = self.l2_cache.get(query)
        if response:
            self.l1_cache[query] = response  # Promote to L1
            return response

        # L3: Medium similarity (slower but broader)
        response = self.l3_cache.get(query)
        if response:
            self.l1_cache[query] = response  # Promote to L1
            return response

        return None  # Cache miss across all tiers
```

## Comparison with Related Patterns

| Pattern | Focus | Primary Goal | When to Use |
|---------|-------|--------------|-------------|
| **Resource Optimization** | Cost & performance | Reduce spend, increase speed | High-volume, cost-sensitive systems |
| **Evaluation & Monitoring** | Quality & observability | Track metrics, ensure quality | All production systems |
| **Prioritization** | Task ordering | Handle important tasks first | Multiple concurrent requests |
| **Planning** | Task decomposition | Break complex tasks into steps | Complex multi-step workflows |

**Complementary patterns**: Resource Optimization + Evaluation/Monitoring work excellently together (optimize while ensuring quality maintained).

**Alternative patterns**: Use Planning instead of batching for sequential dependencies.

## Common Pitfalls

### 1. Premature Optimization

**Problem**: Implementing complex optimizations before understanding bottlenecks

**Symptoms**:
- Complex code with minimal benefit
- Optimization overhead exceeds savings
- Development time wasted

**Solution**:
- Profile first, optimize second
- Start with simple caching
- Measure impact before adding complexity
- Use the 80/20 rule (simple optimizations give 80% of benefits)

### 2. Over-Caching

**Problem**: Caching responses that shouldn't be cached (time-sensitive, user-specific)

**Symptoms**:
- Users see outdated information
- Personalization breaks (wrong user's data)
- Stale cache causing incorrect outputs

**Solution**:
- Use appropriate TTL for data freshness
- Don't cache user-specific or time-sensitive data
- Implement cache invalidation strategies
- Add cache metadata (timestamp, user_id)

### 3. Quality Degradation

**Problem**: Aggressive cost optimization hurts output quality

**Symptoms**:
- User complaints about accuracy
- Declining satisfaction scores
- Increased support tickets

**Solution**:
- A/B test before full rollout
- Monitor quality metrics closely
- Set quality thresholds for model fallback
- Allow user selection (fast vs. accurate modes)
- Have override mechanisms for critical queries

### 4. Ignoring Cold Start

**Problem**: First users experience poor performance (cache empty)

**Symptoms**:
- Initial requests slow and expensive
- New features start poorly
- User complaints after deployments

**Solution**:
- Pre-warm cache with common queries
- Gradual rollout to build cache
- Use predictive prefetching
- Monitor cache miss rate after deployments

### 5. Metrics Blind Spots

**Problem**: Optimizing for cost/speed without tracking quality impact

**Symptoms**:
- Costs down but user satisfaction also down
- Faster but less accurate responses
- Business metrics declining despite technical wins

**Solution**:
- Track quality metrics alongside cost/speed
- Use LLM-as-judge for automated evaluation
- Collect user feedback continuously
- Set quality guardrails (minimum acceptable score)
- Balance metrics (cost, speed, quality together)

## Conclusion

The Resource Optimization Pattern is essential for building cost-effective, high-performance AI applications at scale. By intelligently applying caching, model routing, prompt optimization, and batching, you can dramatically reduce costs (40-80%) and improve latency (2-10x) while maintaining quality.

**Use Resource Optimization when:**
- LLM costs are significant ($500+/month)
- Latency impacts user experience
- Queries have repetition or similarity (cache hit potential > 20%)
- Workloads have mixed complexity (can benefit from model routing)
- Need predictable costs and performance at scale

**Implementation checklist:**
- ✅ Profile baseline performance (latency, cost, token usage)
- ✅ Implement response caching with appropriate TTL
- ✅ Monitor cache hit rate and cost savings
- ✅ Add model routing based on query complexity
- ✅ Optimize prompts to reduce token usage
- ✅ Implement batching for high-throughput scenarios
- ✅ Track quality metrics to ensure optimization doesn't hurt outputs
- ✅ Use gradual rollout with A/B testing
- ✅ Set up dashboards for cost, latency, and quality monitoring
- ✅ Document optimization strategies and trade-offs

**Key Takeaways:**
- 💰 Caching provides highest ROI (60-80% hit rate = 60-80% cost reduction)
- ⚡ Model routing balances cost and quality (60x cost difference between models)
- 🎯 Always measure quality impact alongside cost/speed improvements
- 🔄 Start simple (basic caching) and add complexity incrementally
- 📊 Monitor metrics continuously to validate optimizations
- 🚀 Pre-warm caches and use gradual rollout for smooth deployments
- ⚖️ Balance trade-offs explicitly (document quality vs. cost decisions)

---

*Resource Optimization transforms expensive, slow AI systems into cost-effective, high-performance applications—enabling sustainable scaling without compromising user experience.*
