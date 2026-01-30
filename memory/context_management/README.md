# Context Management

## Overview

The **Context Management Pattern** addresses one of the most critical challenges in LLM applications: fitting relevant information within the model's context window while optimizing for cost, relevance, and performance. As conversations grow longer, documents become larger, and data sources multiply, effective context management becomes essential for maintaining high-quality responses without exceeding token limits or incurring unnecessary costs.

Context management goes beyond simple truncation. It involves intelligent selection, compression, prioritization, and dynamic allocation of limited context space to maximize the value of every token sent to the model. This pattern is the difference between an agent that struggles with long conversations and one that gracefully handles extensive context while remaining cost-effective and performant.

## Why Use This Pattern?

Modern LLM applications face significant context-related challenges:

- **Token limits**: Models have hard limits (4K-200K tokens) that conversations easily exceed
- **Cost scaling**: Costs grow linearly with input tokens, making naive approaches expensive
- **Relevance degradation**: Irrelevant context dilutes important information
- **Performance impact**: Longer contexts increase latency and reduce response quality
- **Information overload**: Too much context can confuse the model ("lost in the middle")
- **Multi-source complexity**: Combining chat history, documents, and data requires careful orchestration

Context Management solves these by:
- **Smart selection**: Choose only the most relevant information for the current query
- **Intelligent compression**: Summarize less critical content while preserving key details
- **Dynamic allocation**: Adjust token budgets based on content importance and query needs
- **Hierarchical organization**: Structure context from high-level summaries to detailed content
- **Cost optimization**: Reduce token usage by 50-90% while maintaining quality
- **Performance improvement**: Shorter contexts mean faster responses and better focus

### Example: Document Q&A with Context Management

```
Without Context Management (Naive):
Document: 50,000 tokens of product documentation
Query: "What's the return policy?"
Context sent: All 50,000 tokens → expensive, slow, unfocused
Cost: $0.50 per query
Response time: 8 seconds

With Context Management (Smart):
Document: 50,000 tokens of product documentation
Query: "What's the return policy?"

Step 1 - Semantic Search: Find relevant sections
→ Found 3 sections mentioning returns (2,000 tokens)

Step 2 - Prioritization: Rank by relevance
→ Section A (returns policy): High priority - keep full (800 tokens)
→ Section B (warranty): Medium priority - summarize (800 → 200 tokens)
→ Section C (shipping): Low priority - exclude (0 tokens)

Step 3 - Context Assembly
→ High-level summary: 100 tokens
→ Critical section (A): 800 tokens
→ Supporting info (B): 200 tokens
→ Total context: 1,100 tokens

Cost: $0.011 per query (95% reduction)
Response time: 2 seconds (75% faster)
Quality: Improved focus on relevant content
```

## How It Works

Context management operates through a sophisticated pipeline:

1. **Selection**: Identify candidate content based on relevance to current query
2. **Scoring**: Rank content by importance, recency, and semantic similarity
3. **Compression**: Apply appropriate compression strategies to each content piece
4. **Allocation**: Distribute token budget based on priorities
5. **Assembly**: Construct final context with optimal structure
6. **Monitoring**: Track utilization and performance metrics

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                          │
└────────────────────┬────────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │  Content Selection    │
         │  - Semantic search    │
         │  - Keyword matching   │
         │  - Recency filtering  │
         └───────┬───────────────┘
                 ↓
         ┌───────────────────────┐
         │  Relevance Scoring    │
         │  - Semantic similarity│
         │  - Recency score      │
         │  - Importance weight  │
         └───────┬───────────────┘
                 ↓
         ┌───────────────────────┐
         │  Token Budget Calc    │
         │  - Available tokens   │
         │  - Priority allocation│
         │  - Reserve buffers    │
         └───────┬───────────────┘
                 ↓
         ┌───────────────────────┐
         │   Compression         │
         │  - Full inclusion     │
         │  - Summarization      │
         │  - Truncation         │
         │  - Exclusion          │
         └───────┬───────────────┘
                 ↓
         ┌───────────────────────┐
         │  Context Assembly     │
         │  - Hierarchical       │
         │  - Structured         │
         │  - Formatted          │
         └───────┬───────────────┘
                 ↓
         ┌───────────────────────┐
         │   LLM Processing      │
         │  (optimized context)  │
         └───────────────────────┘
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Long conversations**: Multi-turn dialogues that exceed context windows
- **Document analysis**: Large documents requiring focused extraction
- **Multi-document synthesis**: Combining information from multiple sources
- **Cost-sensitive applications**: When token costs are a primary concern
- **RAG systems**: Optimizing retrieved chunks before sending to LLM
- **Chat applications**: Maintaining history without overwhelming context
- **Research assistants**: Processing extensive materials efficiently
- **Support systems**: Referencing large knowledge bases economically

### ❌ When NOT to Use

- **Short conversations**: When all context easily fits in the window
- **Single queries**: One-off questions without history
- **Simple prompts**: Direct questions needing no additional context
- **Maximum quality required**: When no information loss is acceptable
- **Real-time critical**: When compression latency is prohibitive

## Rule of Thumb

**Use Context Management when:**
1. Context regularly **exceeds 50% of available tokens**
2. You're making **more than 100 requests per day** (cost matters)
3. Context includes **multiple sources** of varying importance
4. You need to **maintain long-term history** across sessions
5. **Response quality** suffers from too much irrelevant context

**Don't use Context Management when:**
1. All context fits comfortably in **<25% of token limit**
2. Every piece of context is **equally critical**
3. **Compression latency** exceeds value gained
4. Context is already **highly optimized**

## Core Components

### 1. Token Counting

Accurate measurement of content size:

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using model-specific tokenizer"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "This is a sample message."
tokens = count_tokens(text)  # Returns: 6 tokens
```

### 2. Relevance Scoring

Determining importance of content:

```python
def calculate_relevance(query: str, content: str) -> float:
    """Score content relevance to query (0-1)"""
    # Semantic similarity (e.g., embeddings cosine similarity)
    semantic_score = compute_similarity(query, content)

    # Recency bonus (for time-sensitive content)
    recency_score = calculate_recency_weight(content)

    # Keyword matching
    keyword_score = keyword_overlap(query, content)

    # Combined score
    return 0.5 * semantic_score + 0.3 * recency_score + 0.2 * keyword_score
```

### 3. Compression Strategies

Different approaches for different content:

```python
class CompressionStrategy:
    FULL = "full"           # Include complete content
    SUMMARIZE = "summarize" # Generate summary
    TRUNCATE = "truncate"   # Cut to fixed length
    EXTRACT = "extract"     # Pull key sentences
    EXCLUDE = "exclude"     # Don't include
```

### 4. Dynamic Allocation

Distributing token budget:

```python
def allocate_tokens(
    contents: List[Content],
    total_budget: int,
    reserve: int = 500  # Buffer for response
) -> Dict[str, int]:
    """Allocate tokens based on priority"""
    available = total_budget - reserve

    # Sort by relevance score
    sorted_contents = sorted(contents, key=lambda x: x.score, reverse=True)

    # Allocate proportionally to scores
    allocations = {}
    for content in sorted_contents:
        allocation = int(available * (content.score / total_score))
        allocations[content.id] = allocation

    return allocations
```

### 5. Context Window Management

```python
class ContextWindow:
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.system_tokens = 0
        self.query_tokens = 0
        self.response_reserve = 4096

    @property
    def available_for_context(self) -> int:
        """Calculate available space for dynamic context"""
        used = self.system_tokens + self.query_tokens + self.response_reserve
        return self.max_tokens - used
```

## Implementation Approaches

### Approach 1: Sliding Window (Simple)

Best for: Chat history with chronological importance

```python
class SlidingWindowContext:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)

    def add_message(self, message: str):
        """Add message, automatically removing oldest if full"""
        self.messages.append(message)

    def get_context(self) -> List[str]:
        """Return recent messages"""
        return list(self.messages)
```

**Pros**: Simple, fast, predictable
**Cons**: No semantic awareness, may lose important old context

### Approach 2: Priority-Based Selection

Best for: Multi-source content with varying importance

```python
class PriorityContext:
    def __init__(self, token_budget: int):
        self.token_budget = token_budget
        self.contents: List[PrioritizedContent] = []

    def add_content(self, content: str, priority: float):
        """Add content with priority score"""
        self.contents.append(
            PrioritizedContent(content, priority)
        )

    def build_context(self) -> str:
        """Select highest priority content within budget"""
        sorted_contents = sorted(
            self.contents,
            key=lambda x: x.priority,
            reverse=True
        )

        selected = []
        tokens_used = 0

        for content in sorted_contents:
            tokens = count_tokens(content.text)
            if tokens_used + tokens <= self.token_budget:
                selected.append(content.text)
                tokens_used += tokens
            else:
                break

        return "\n\n".join(selected)
```

**Pros**: Optimizes for importance, flexible
**Cons**: Requires priority scores, may miss context spread

### Approach 3: Semantic Chunking with Compression

Best for: Long documents needing intelligent extraction

```python
class SemanticContextManager:
    def __init__(self, token_budget: int):
        self.token_budget = token_budget
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def build_context(self, document: str, query: str) -> str:
        """Select and compress relevant chunks"""
        # 1. Split into semantic chunks
        chunks = self.semantic_chunking(document)

        # 2. Score relevance to query
        scored_chunks = [
            (chunk, self.relevance_score(query, chunk))
            for chunk in chunks
        ]

        # 3. Sort by relevance
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # 4. Allocate tokens and compress
        context_parts = []
        tokens_used = 0

        for chunk, score in scored_chunks:
            chunk_tokens = count_tokens(chunk)

            if score > 0.8:  # High relevance: keep full
                if tokens_used + chunk_tokens <= self.token_budget:
                    context_parts.append(chunk)
                    tokens_used += chunk_tokens
            elif score > 0.5:  # Medium: summarize
                summary = self.summarize(chunk, target_ratio=0.3)
                summary_tokens = count_tokens(summary)
                if tokens_used + summary_tokens <= self.token_budget:
                    context_parts.append(f"[Summary] {summary}")
                    tokens_used += summary_tokens
            # Low relevance: skip

            if tokens_used >= self.token_budget * 0.9:
                break

        return "\n\n".join(context_parts)
```

**Pros**: Intelligent, semantic awareness, adaptive compression
**Cons**: More complex, requires embeddings/LLM calls

### Approach 4: Hierarchical Context

Best for: Complex information with natural hierarchy

```python
class HierarchicalContext:
    def __init__(self, token_budget: int):
        self.token_budget = token_budget

    def build_context(self, hierarchy: ContextHierarchy) -> str:
        """Build context from abstract to detailed"""
        context_parts = []
        tokens_used = 0

        # Level 1: Always include executive summary
        summary = hierarchy.summary
        summary_tokens = count_tokens(summary)
        context_parts.append(f"=== OVERVIEW ===\n{summary}")
        tokens_used += summary_tokens

        # Level 2: Include section summaries
        remaining = self.token_budget - tokens_used
        for section in hierarchy.sections:
            section_summary = section.summary
            section_tokens = count_tokens(section_summary)

            if tokens_used + section_tokens <= self.token_budget * 0.6:
                context_parts.append(f"=== {section.title} ===\n{section_summary}")
                tokens_used += section_tokens

        # Level 3: Add detailed content for top sections
        remaining = self.token_budget - tokens_used
        for section in hierarchy.get_top_sections(by_relevance=True):
            if tokens_used < self.token_budget * 0.9:
                detail_tokens = count_tokens(section.detail)
                if tokens_used + detail_tokens <= self.token_budget:
                    context_parts.append(f"[Detail: {section.title}]\n{section.detail}")
                    tokens_used += detail_tokens

        return "\n\n".join(context_parts)
```

**Pros**: Natural structure, graceful degradation, balanced overview
**Cons**: Requires pre-structured content

## Key Benefits

### 💰 Cost Reduction

**Impact**: Reduce token usage by 50-90% while maintaining quality

**Example**:
- Before: 10,000 tokens/query × $0.01/1K tokens = $0.10/query
- After: 2,000 tokens/query × $0.01/1K tokens = $0.02/query
- **Savings: 80% cost reduction**

For 1,000 queries/day: $100/day → $20/day = **$2,400/month saved**

### ⚡ Performance Improvement

**Impact**: Faster responses, better focus, improved accuracy

**Metrics**:
- Response latency: ↓ 40-70% (shorter context = faster processing)
- Relevance score: ↑ 15-30% (focused context = better answers)
- "Lost in middle" errors: ↓ 60-80% (less irrelevant content)

### 🎯 Better Model Performance

**Impact**: Models perform better with focused, relevant context

LLMs exhibit "lost in the middle" phenomenon - they pay more attention to the beginning and end of context. By providing only relevant information, you:
- Reduce distraction from irrelevant content
- Improve factual grounding
- Decrease hallucination rates
- Enhance instruction following

### 📏 Scale Beyond Limits

**Impact**: Handle conversations and documents far exceeding model limits

- Support 100K+ token conversations in 8K context models
- Process multi-document research within single query
- Maintain months of chat history efficiently
- Scale to enterprise knowledge bases

## Trade-offs

### ⚠️ Information Loss

**Issue**: Compression and selection inevitably lose some information

**Impact**: Edge cases may lack critical context

**Mitigation**:
- Use conservative compression for high-stakes applications
- Implement relevance thresholds carefully
- Allow users to request full context when needed
- Log excluded content for debugging
- Implement "expand context" fallback mechanism

### 🔄 Processing Overhead

**Issue**: Context management adds latency before LLM call

**Impact**: Additional 100-500ms for selection and compression

**Mitigation**:
- Cache embeddings for repeated content
- Pre-compute summaries for static documents
- Use fast models for summarization (GPT-4o-mini)
- Parallelize independent operations
- Optimize hot paths with profiling

### 🧮 Complexity Increase

**Issue**: More moving parts, harder to debug

**Impact**: Increased development and maintenance burden

**Mitigation**:
- Start simple (sliding window) and add sophistication as needed
- Log all context management decisions
- Visualize token allocation for debugging
- Implement comprehensive testing
- Use clear abstractions and modular design

### 📊 Tuning Required

**Issue**: Optimal strategies vary by use case

**Impact**: Requires experimentation and monitoring

**Mitigation**:
- Implement A/B testing for strategies
- Track quality metrics (relevance, accuracy)
- Use automatic parameter tuning when possible
- Provide configuration presets for common scenarios
- Monitor and alert on performance degradation

## Best Practices

### 1. Accurate Token Counting

```python
import tiktoken

# Use model-specific tokenizer
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Always use tiktoken for accurate counts"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

# Account for message formatting
def count_message_tokens(messages: List[dict], model: str) -> int:
    """Count tokens including message formatting overhead"""
    tokens = 0
    for message in messages:
        tokens += 4  # Message formatting overhead
        tokens += count_tokens(message["content"], model)
        tokens += count_tokens(message["role"], model)
    tokens += 2  # Conversation formatting
    return tokens
```

### 2. Reserve Response Budget

```python
class TokenBudget:
    def __init__(self, model_limit: int = 128000):
        self.model_limit = model_limit
        self.response_reserve = 4096  # Reserve for model output
        self.system_prompt = 500      # Fixed system prompt size
        self.query_buffer = 200       # Safety buffer

    @property
    def available_for_context(self) -> int:
        """Calculate space available for dynamic context"""
        return (
            self.model_limit
            - self.response_reserve
            - self.system_prompt
            - self.query_buffer
        )
```

### 3. Implement Graceful Degradation

```python
def build_context(
    contents: List[Content],
    token_budget: int
) -> str:
    """Build context with graceful degradation"""

    # Priority 1: Critical content (always include)
    critical = [c for c in contents if c.priority == "critical"]
    critical_context = "\n".join([c.text for c in critical])
    tokens_used = count_tokens(critical_context)

    # Priority 2: High-value content (full if space, else summarize)
    high_value = [c for c in contents if c.priority == "high"]
    for content in high_value:
        content_tokens = count_tokens(content.text)

        if tokens_used + content_tokens <= token_budget * 0.8:
            critical_context += f"\n\n{content.text}"
            tokens_used += content_tokens
        else:
            # Summarize to fit
            summary = summarize(content.text, max_tokens=content_tokens // 3)
            summary_tokens = count_tokens(summary)
            if tokens_used + summary_tokens <= token_budget:
                critical_context += f"\n\n[Summary] {summary}"
                tokens_used += summary_tokens

    # Priority 3: Supporting content (include if space)
    supporting = [c for c in contents if c.priority == "supporting"]
    for content in supporting:
        content_tokens = count_tokens(content.text)
        if tokens_used + content_tokens <= token_budget * 0.95:
            critical_context += f"\n\n{content.text}"
            tokens_used += content_tokens

    return critical_context
```

### 4. Cache Expensive Operations

```python
from functools import lru_cache
import hashlib

class CachedContextManager:
    def __init__(self):
        self.embedding_cache = {}
        self.summary_cache = {}

    def get_embedding(self, text: str) -> List[float]:
        """Cache embeddings for repeated content"""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = compute_embedding(text)

        return self.embedding_cache[text_hash]

    @lru_cache(maxsize=100)
    def get_summary(self, text: str, target_tokens: int) -> str:
        """Cache summaries with LRU eviction"""
        return generate_summary(text, target_tokens)
```

### 5. Monitor and Log

```python
class ContextMetrics:
    def __init__(self):
        self.metrics = []

    def log_context_usage(
        self,
        query: str,
        context_built: str,
        tokens_available: int,
        tokens_used: int,
        compression_ratio: float
    ):
        """Log metrics for each context building operation"""
        self.metrics.append({
            "timestamp": datetime.now(),
            "query_tokens": count_tokens(query),
            "context_tokens": tokens_used,
            "utilization": tokens_used / tokens_available,
            "compression_ratio": compression_ratio,
        })

    def get_stats(self) -> dict:
        """Calculate aggregate statistics"""
        return {
            "avg_utilization": np.mean([m["utilization"] for m in self.metrics]),
            "avg_compression": np.mean([m["compression_ratio"] for m in self.metrics]),
            "total_tokens_saved": sum([m.get("tokens_saved", 0) for m in self.metrics]),
        }
```

## Performance Metrics

Track these key metrics:

### Context Utilization
```
Utilization = Tokens Used / Tokens Available
Target: 70-90% (efficient use without waste)
```

### Compression Ratio
```
Compression Ratio = Original Tokens / Final Tokens
Higher = More aggressive compression
Track against quality metrics
```

### Relevance Score
```
Relevance Score = Avg(semantic_similarity(chunk, query))
Target: >0.7 for included content
Monitor excluded content scores
```

### Cost Savings
```
Cost Savings = (Baseline Cost - Optimized Cost) / Baseline Cost
Track per query and aggregate
```

### Quality Metrics
```
- Answer accuracy (compared to full context baseline)
- User satisfaction scores
- Task completion rates
- Error rates
```

## Example Scenarios

### Scenario 1: Long Conversation Management

```python
# Problem: 50-turn conversation exceeds 32K token limit

# Naive approach: Include all messages = 45,000 tokens (fails!)

# Context Management approach:
context_manager = ConversationContextManager(max_tokens=8000)

# Add all messages with metadata
for turn in conversation_history:
    context_manager.add_message(
        role=turn.role,
        content=turn.content,
        timestamp=turn.timestamp
    )

# Build context for new query
new_query = "What was the price we discussed for the enterprise plan?"

context = context_manager.build_context(
    query=new_query,
    strategy="semantic_with_recency"
)

# Result:
# - Semantic search finds: Turns 15-17 (pricing discussion) - 1,200 tokens
# - Recency includes: Last 5 turns (context continuity) - 800 tokens
# - Summary of: Turns 1-14, 18-44 (background) - 600 tokens
# - Total: 2,600 tokens (94% reduction, all relevant info included)
```

### Scenario 2: Multi-Document Q&A

```python
# Problem: Answer question from 5 documents (100K+ tokens total)

documents = [
    ("product_specs.pdf", 25000),
    ("user_manual.pdf", 30000),
    ("faq.pdf", 15000),
    ("release_notes.pdf", 20000),
    ("api_docs.pdf", 35000),
]
# Total: 125,000 tokens (exceeds GPT-4's 128K limit with no room for response!)

query = "How do I configure OAuth authentication?"

# Context Management approach:
context_manager = DocumentContextManager(max_tokens=16000)

# Stage 1: Semantic search across all documents
relevant_chunks = context_manager.semantic_search(
    query=query,
    documents=documents,
    top_k=10
)
# Found 10 chunks totaling 8,000 tokens

# Stage 2: Re-rank by relevance
ranked_chunks = context_manager.rerank(
    query=query,
    chunks=relevant_chunks
)

# Stage 3: Build hierarchical context
context = context_manager.build_hierarchical_context(
    chunks=ranked_chunks,
    token_budget=16000
)

# Result:
# Level 1 - Summary: "OAuth setup involves... " (500 tokens)
# Level 2 - Detailed: Top 3 most relevant sections (6,000 tokens)
# Level 3 - Supporting: Relevant code examples (2,500 tokens)
# Total: 9,000 tokens (93% reduction, focused on OAuth)
```

### Scenario 3: RAG with Context Optimization

```python
# Problem: Retrieved 20 chunks (15K tokens), but only 8K budget

chunks = retrieval_system.retrieve(query, top_k=20)  # 15,000 tokens
token_budget = 8000

# Context Management approach:
context_optimizer = RAGContextOptimizer(
    token_budget=token_budget,
    min_chunks=3,  # Always include at least 3
    max_chunks=15
)

optimized_context = context_optimizer.optimize(
    query=query,
    chunks=chunks,
    strategy="adaptive_compression"
)

# Process:
# 1. Rank chunks by relevance (semantic similarity to query)
# 2. Allocate tokens proportionally to relevance scores
# 3. Apply compression based on allocation:
#    - Top 3 chunks (score >0.9): Full content (4,500 tokens)
#    - Next 5 chunks (score 0.7-0.9): Summarized 50% (2,000 tokens)
#    - Next 7 chunks (score 0.5-0.7): Key sentences only (1,200 tokens)
#    - Last 5 chunks: Excluded (too low relevance)
# Total: 7,700 tokens (49% reduction, maintained top content)
```

## Advanced Patterns

### 1. Adaptive Compression

Adjust compression based on content type and importance:

```python
class AdaptiveCompressor:
    def compress(self, content: Content, target_tokens: int) -> str:
        """Apply compression strategy based on content type"""

        if content.type == "code":
            # Preserve code structure, remove comments
            return self.compress_code(content.text, target_tokens)

        elif content.type == "data":
            # Extract key statistics, sample rows
            return self.compress_data(content.text, target_tokens)

        elif content.type == "conversation":
            # Summarize old, keep recent verbatim
            return self.compress_conversation(content.text, target_tokens)

        elif content.type == "document":
            # Extractive summarization
            return self.compress_document(content.text, target_tokens)

        else:
            # Generic LLM-based summarization
            return self.llm_summarize(content.text, target_tokens)
```

### 2. Context Caching

Leverage model-level context caching (e.g., Claude's prompt caching):

```python
class CachedContextManager:
    def build_context(
        self,
        static_content: str,      # Cached (docs, knowledge base)
        dynamic_content: str,     # Not cached (chat history)
        query: str
    ) -> dict:
        """Structure context to maximize cache hits"""

        # Structure: static (cached) + dynamic + query
        return {
            "system": f"""You are a helpful assistant.

Reference Documents (cached):
{static_content}
""",
            "messages": [
                {"role": "user", "content": f"""Recent Conversation:
{dynamic_content}

Current Question: {query}"""}
            ]
        }

        # With Claude, static_content gets cached after first use
        # Subsequent requests only pay for dynamic_content + query
```

### 3. Hierarchical Context Windows

Multi-level context for complex scenarios:

```python
class HierarchicalContextWindow:
    def __init__(self):
        self.L1_summary = ""        # 500 tokens - always included
        self.L2_sections = []       # 2000 tokens - section summaries
        self.L3_details = []        # 5000 tokens - full content
        self.L4_archive = []        # Not included, available on demand

    def build_context(self, detail_level: str = "auto") -> str:
        """Build context with appropriate detail level"""

        if detail_level == "auto":
            # Decide based on query complexity
            detail_level = self.determine_detail_level()

        context_parts = [self.L1_summary]

        if detail_level in ["medium", "high"]:
            context_parts.extend(self.L2_sections)

        if detail_level == "high":
            context_parts.extend(self.L3_details)

        return "\n\n".join(context_parts)
```

### 4. Query-Driven Context Selection

Analyze query to guide context selection:

```python
class QueryDrivenContextManager:
    def build_context(self, query: str, available_content: List[Content]) -> str:
        """Select context based on query characteristics"""

        # Analyze query
        query_type = self.classify_query(query)
        # Types: factual, analytical, creative, comparative, etc.

        needs_recent = self.requires_recency(query)
        needs_detail = self.requires_detail(query)
        needs_breadth = self.requires_breadth(query)

        # Select strategy based on analysis
        if query_type == "factual" and needs_detail:
            # Include fewer chunks with more detail
            strategy = "deep_and_narrow"
        elif query_type == "comparative":
            # Include multiple sources, summarized
            strategy = "broad_and_summarized"
        elif needs_recent:
            # Prioritize recent content
            strategy = "recency_weighted"
        else:
            # Default: relevance-based
            strategy = "semantic_similarity"

        return self.apply_strategy(strategy, query, available_content)
```

## Comparison with Related Patterns

| Pattern | Focus | Token Optimization | Use Case |
|---------|-------|-------------------|----------|
| **Context Management** | Fitting context in window | Yes, aggressive | Large contexts, cost-sensitive |
| **Memory Management** | Long-term persistence | Moderate (summarization) | Multi-session continuity |
| **RAG** | Relevant retrieval | Yes (retrieves subset) | Large knowledge bases |
| **Planning** | Task decomposition | Indirect (via task focus) | Complex workflows |
| **Prompt Chaining** | Sequential processing | No (each step independent) | Multi-stage tasks |

**Context Management complements these patterns:**
- Use with **Memory Management** to handle stored history
- Use with **RAG** to optimize retrieved chunks
- Use within **Planning** to manage context per subtask
- Use in **Prompt Chains** to optimize each step's context

## Common Pitfalls

### 1. Over-Aggressive Compression

**Problem**: Compressing too much loses critical information

**Symptoms**:
- Model asks for information already provided
- Answers miss key details
- User has to repeat themselves

**Solution**:
```python
# Implement compression thresholds
def compress_safely(content: str, max_tokens: int) -> str:
    min_compression_ratio = 0.5  # Never compress below 50% of original

    original_tokens = count_tokens(content)
    if max_tokens > original_tokens * min_compression_ratio:
        return summarize(content, max_tokens)
    else:
        # Compression would be too aggressive, exclude instead
        return None
```

### 2. Ignoring Content Structure

**Problem**: Breaking semantic units mid-thought

**Solution**: Compress at natural boundaries (paragraphs, sections)

```python
def compress_with_structure(document: str, target_tokens: int) -> str:
    """Compress while respecting document structure"""
    sections = split_by_sections(document)

    # Calculate per-section allocation
    section_allocations = allocate_tokens_proportionally(
        sections, target_tokens
    )

    # Compress each section independently
    compressed_sections = []
    for section, allocation in zip(sections, section_allocations):
        compressed = compress_section(section, allocation)
        compressed_sections.append(compressed)

    return "\n\n".join(compressed_sections)
```

### 3. Static Budgets

**Problem**: Using fixed token budgets regardless of query complexity

**Solution**: Adjust budgets dynamically

```python
def calculate_dynamic_budget(query: str, available_tokens: int) -> int:
    """Adjust budget based on query needs"""

    # Simple queries need less context
    if is_simple_query(query):
        return int(available_tokens * 0.5)

    # Complex queries need more context
    elif is_complex_query(query):
        return int(available_tokens * 0.9)

    # Default: moderate allocation
    else:
        return int(available_tokens * 0.7)
```

### 4. No Relevance Verification

**Problem**: Including content without verifying it helps answer the query

**Solution**: Implement relevance thresholds

```python
def select_relevant_content(
    query: str,
    candidates: List[Content],
    min_relevance: float = 0.5
) -> List[Content]:
    """Only include content above relevance threshold"""

    relevant = []
    for content in candidates:
        relevance_score = calculate_relevance(query, content.text)

        if relevance_score >= min_relevance:
            relevant.append((content, relevance_score))

    return [c for c, _ in sorted(relevant, key=lambda x: x[1], reverse=True)]
```

## Conclusion

Context Management is essential for building production-grade LLM applications that are cost-effective, performant, and scalable. By intelligently selecting, compressing, and organizing context, you can handle scenarios far beyond naive approaches while reducing costs by 50-90% and improving response quality.

**Use Context Management when:**
- Working with large documents or long conversations
- Cost optimization is important (>100 requests/day)
- Context regularly approaches token limits
- Combining multiple information sources
- Response quality suffers from information overload

**Implementation checklist:**
- ✅ Use tiktoken for accurate token counting
- ✅ Reserve tokens for model response
- ✅ Implement relevance scoring (semantic + recency)
- ✅ Support multiple compression strategies
- ✅ Build hierarchical context structures
- ✅ Cache expensive operations (embeddings, summaries)
- ✅ Monitor metrics (utilization, compression, quality)
- ✅ Log context decisions for debugging
- ✅ Implement graceful degradation
- ✅ Test against full-context baseline

**Key Takeaways:**
- 🎯 Context management is optimization under constraints
- 💰 Proper implementation saves 50-90% on token costs
- ⚡ Focused context improves both speed and quality
- 🔍 Relevance scoring is critical for selection
- 📊 Monitor metrics to tune strategies
- 🏗️ Start simple, add sophistication as needed
- ⚖️ Balance information preservation with compression
- 🔄 Adapt strategies to query and content types

---

*Context Management transforms how LLMs handle information—moving from "fit everything" to "fit the right things"—enabling scalable, cost-effective, and high-quality applications that gracefully handle the complexity of real-world data.*
