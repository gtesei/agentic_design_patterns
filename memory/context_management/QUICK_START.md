# Context Management - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Context Management Directory
```bash
cd memory/context_management
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic Context Management (windowing & compression)
- **Option 2**: Advanced Context Management (semantic selection & optimization)
- **Option 3**: Run all examples

---

## 📖 Understanding Context Management in 30 Seconds

**Context Management** = Intelligently fitting relevant information within token limits

The process:
1. **Select**: Find relevant content for the query
2. **Score**: Rank by importance and relevance
3. **Compress**: Summarize or truncate as needed
4. **Allocate**: Distribute token budget optimally
5. **Assemble**: Build final context structure

**Result**: 50-90% cost savings while maintaining quality

---

## 🎯 Key Concepts

### Token Budget
```
Total Tokens Available: 128,000 (model limit)
- System Prompt: -500
- User Query: -200
- Response Reserve: -4,096
= Available for Context: 123,204
```

### Compression Strategies

| Strategy | When to Use | Compression Ratio |
|----------|-------------|-------------------|
| **Full** | Critical content | 1.0 (no compression) |
| **Summarize** | Important but lengthy | 0.3-0.5 (50-70% reduction) |
| **Extract** | Medium priority | 0.2-0.3 (70-80% reduction) |
| **Exclude** | Low relevance | 0.0 (100% reduction) |

### Relevance Scoring
```
Relevance Score = 0.5 × Semantic Similarity
                + 0.3 × Recency Weight
                + 0.2 × Keyword Match
```

---

## 💡 Example Scenarios

### Scenario 1: Long Conversation (Basic Example)
```
Problem: 30-turn conversation = 25,000 tokens
Budget: Only 8,000 tokens available

Solution:
- Keep last 5 turns verbatim (recent context)
- Summarize turns 5-25 (background info)
- Include turn 1 (conversation start)
Result: 3,500 tokens used (86% reduction)
```

### Scenario 2: Multi-Document Q&A (Advanced Example)
```
Problem: 5 documents = 125,000 tokens
Query: "How do I configure OAuth authentication?"
Budget: 16,000 tokens available

Solution:
- Semantic search: Find 10 relevant chunks (8,000 tokens)
- Rerank by relevance to query
- Build hierarchical context:
  * Summary level: 500 tokens
  * Detail level (top 3 chunks): 6,000 tokens
  * Supporting info: 2,500 tokens
Result: 9,000 tokens used (93% reduction, focused on OAuth)
```

---

## 🛠️ Available Demonstrations

### Basic Implementation (`context_basic.py`)
Features:
- Sliding window for conversation history
- Simple compression via summarization
- Token counting with tiktoken
- Priority-based inclusion
- Visual context packing diagram

**Use Case**: Chat applications, conversation management

### Advanced Implementation (`context_advanced.py`)
Features:
- Semantic relevance scoring
- Multi-source context (documents, history, knowledge)
- Dynamic token allocation
- Hierarchical compression
- Adaptive strategies
- Context caching patterns
- Rich visualization

**Use Case**: Document Q&A, RAG systems, multi-source synthesis

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| Strategy | Sliding window | Semantic selection |
| Relevance | Recency-based | Semantic + recency |
| Compression | Simple summarization | Multi-level adaptive |
| Sources | Single (conversation) | Multiple (docs, chat, KB) |
| Visualization | Token usage | Full optimization pipeline |
| Complexity | Low | Medium-High |
| Use Cases | Chat history | Document analysis, RAG |

**Recommendation**: Start with Basic for chat apps, use Advanced for document/multi-source scenarios.

---

## 🔧 Customization Tips

### Adjust Token Budget

```python
# In context_basic.py
context_manager = ContextManager(
    max_tokens=8000,        # Change this
    response_reserve=2000   # Reserve for model output
)
```

### Change Compression Ratio

```python
# More aggressive compression
compression_ratio = 0.2  # Keep 20% (was 50%)

# Less aggressive compression
compression_ratio = 0.7  # Keep 70%
```

### Modify Relevance Scoring

```python
# In context_advanced.py
def calculate_relevance(query: str, content: str) -> float:
    semantic = compute_similarity(query, content)
    recency = calculate_recency_weight(content)
    keywords = keyword_overlap(query, content)

    # Adjust weights (must sum to 1.0)
    return 0.5 * semantic + 0.3 * recency + 0.2 * keywords
```

### Add Custom Compression Strategy

```python
def custom_compressor(text: str, target_tokens: int) -> str:
    """Your custom compression logic"""
    # Example: Keep first and last paragraphs
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 2:
        return f"{paragraphs[0]}\n\n[...content omitted...]\n\n{paragraphs[-1]}"
    return text
```

---

## ⚡ Common Issues & Solutions

### Issue: "Token count inaccurate"
**Solution**: Always use tiktoken with the correct model name
```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoding.encode(text))
```

### Issue: "Context still too long"
**Solution**: Increase compression ratio or decrease max_tokens
```python
compression_ratio = 0.3  # More aggressive (was 0.5)
```

### Issue: "Important info gets cut off"
**Solution**: Use priority levels
```python
content.priority = "critical"  # Always included
content.priority = "high"       # Summarized if needed
content.priority = "low"        # Excluded if space tight
```

### Issue: "Slow performance"
**Solution**: Cache embeddings and summaries
```python
@lru_cache(maxsize=100)
def get_embedding(text: str):
    return compute_embedding(text)
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ Start: Run basic example to see windowing and compression
2. ✅ Understand: Read the output, see token usage breakdown
3. ✅ Explore: Run advanced example for semantic selection
4. ✅ Analyze: Compare full context vs optimized context
5. ✅ Experiment: Adjust budgets and compression ratios
6. ✅ Customize: Add your own content and strategies
7. ✅ Integrate: Use in your applications (RAG, chat, docs)

---

## 🌟 Pro Tips

### 1. Token Budgeting
Always reserve 20-30% for the model's response:
```python
response_reserve = model_limit * 0.25
available_context = model_limit - response_reserve
```

### 2. Compression Thresholds
Never compress below 50% without testing quality impact:
```python
min_compression_ratio = 0.5
if target < original * min_compression_ratio:
    # Too aggressive, exclude instead
    return None
```

### 3. Hierarchical Structure
Organize context from general to specific:
```
1. Executive summary (always include)
2. Section summaries (include if space)
3. Full details (include top sections only)
```

### 4. Relevance Thresholds
Set minimum relevance scores:
```python
min_relevance = 0.5  # Only include content scoring >0.5
```

### 5. Monitor Metrics
Track these KPIs:
- Context utilization (target: 70-90%)
- Compression ratio (higher = more savings)
- Cost per query (track savings)
- Quality metrics (accuracy vs full context)

### 6. Cache Expensive Ops
Cache embeddings and summaries:
```python
@lru_cache(maxsize=100)
def get_summary(text: str, max_tokens: int) -> str:
    return llm.summarize(text, max_tokens)
```

---

## 💰 Cost Savings Calculator

**Before Context Management:**
```
Document: 50,000 tokens
Queries per day: 1,000
Cost: 50,000 tokens × $0.01/1K = $0.50 per query
Daily cost: $500
Monthly cost: $15,000
```

**After Context Management (80% reduction):**
```
Optimized context: 10,000 tokens
Queries per day: 1,000
Cost: 10,000 tokens × $0.01/1K = $0.10 per query
Daily cost: $100
Monthly cost: $3,000

SAVINGS: $12,000/month (80%)
```

---

## 🎯 When to Use Each Approach

### Use Basic Context Management When:
- ✅ Single source (conversation history)
- ✅ Recency is primary concern
- ✅ Simple implementation needed
- ✅ Chat applications
- ✅ Sequential data

### Use Advanced Context Management When:
- ✅ Multiple content sources
- ✅ Semantic relevance is critical
- ✅ Document analysis or RAG
- ✅ Cost optimization is priority
- ✅ Complex queries requiring focused context

---

## 🔍 Quick Reference

### Token Limits by Model
| Model | Context Window | Recommended Budget |
|-------|----------------|-------------------|
| GPT-4 | 8,192 | 6,000 |
| GPT-4 Turbo | 128,000 | 100,000 |
| GPT-4o | 128,000 | 100,000 |
| Claude 3 Opus | 200,000 | 150,000 |
| Claude 3.5 Sonnet | 200,000 | 150,000 |

### Compression Guidelines
| Original Size | Target Size | Strategy |
|---------------|-------------|----------|
| < 2K tokens | Keep full | No compression |
| 2K - 5K tokens | 1K - 2K | Light summarization |
| 5K - 20K tokens | 2K - 5K | Moderate summarization |
| 20K+ tokens | 5K - 10K | Aggressive + semantic selection |

---

**Happy Context Optimizing! 🚀**

For questions or issues, refer to the full [README.md](./README.md).
