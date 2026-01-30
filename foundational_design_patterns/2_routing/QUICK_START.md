# Routing - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd foundational_design_patterns/2_routing
```

### Step 2: Run the Example
```bash
uv run python src/routing_example.py
```

---

## 📖 Understanding Routing in 30 Seconds

**Routing** intelligently directs queries to specialized handlers:

```
User Query → Classifier → [Expert A | Expert B | Expert C]
```

Like a receptionist directing visitors to the right department:
- Technical questions → Technical expert
- Sales questions → Sales agent
- Support questions → Support bot

---

## 🎯 What This Example Does

The example demonstrates **intent-based routing**:

1. **Classify** - Determine the query type
2. **Route** - Send to appropriate handler
3. **Process** - Specialized agent handles the query
4. **Return** - Get optimized response

---

## 💡 Example Flow

```
Query: "How do I configure authentication?"
    ↓
Classifier: "Technical question"
    ↓
Route to: Technical Expert Agent
    ↓
Response: "To configure authentication, use..."
```

---

## 🔧 Key Concepts

### Intent Classification
Determine what type of query it is before processing.

### Specialized Handlers
Each route has an expert optimized for that query type.

### Cost Optimization
Route simple queries to fast/cheap models, complex ones to powerful models.

### Performance Gains
Fast models for simple tasks = lower latency and cost.

---

## 🎨 When to Use Routing

✅ **Good For:**
- Multi-domain applications (support, sales, technical)
- Cost optimization (use expensive models only when needed)
- Performance optimization (fast models for simple queries)
- Specialized handling (different expertise per domain)

❌ **Not Ideal For:**
- Single-domain applications
- Queries that don't fit clear categories
- Real-time systems where classification adds too much latency

---

## 🛠️ Customization Tips

### Add New Routes

```python
# Define a new handler
def billing_handler(query: str) -> str:
    """Handle billing questions"""
    return llm.invoke(f"Billing question: {query}")

# Add to router
routes = {
    "technical": technical_handler,
    "sales": sales_handler,
    "support": support_handler,
    "billing": billing_handler,  # New route
}
```

### Customize Classification

```python
classifier_prompt = """
Classify this query into one of: technical, sales, support, billing

Query: {query}
Category:
"""
```

### Use Different Models per Route

```python
# Fast model for simple queries
support_llm = ChatOpenAI(model="gpt-4o-mini")

# Powerful model for complex queries
technical_llm = ChatOpenAI(model="gpt-4")
```

---

## 📊 Performance Benefits

| Metric | Without Routing | With Routing |
|--------|----------------|--------------|
| Avg Latency | 3s | 1.5s (-50%) |
| Avg Cost | $0.05/query | $0.02/query (-60%) |
| Accuracy | 85% | 92% (+7%) |

**Why?** Simple queries use fast/cheap models; complex queries get expert handling.

---

## 🔍 Routing Strategies

### 1. Intent-Based (Keyword)
```python
if "how to" in query or "configure" in query:
    return "technical"
```

### 2. LLM-Based Classification
```python
category = classifier_llm.invoke(f"Classify: {query}")
return category
```

### 3. Semantic Similarity
```python
# Find most similar route using embeddings
similarity = cosine_similarity(query_embedding, route_embeddings)
return most_similar_route
```

### 4. Model Selection
```python
# Route based on complexity
if is_complex(query):
    return gpt4_handler
else:
    return gpt4_mini_handler
```

---

## 💡 Common Patterns

### Multi-Stage Routing
```
Query → Primary Router → Secondary Router → Handler
```

### Fallback Routing
```
Query → Try Specialist → If uncertain → General Handler
```

### Parallel Routing
```
Query → [Expert 1, Expert 2, Expert 3] → Combine Results
```

---

## 🐛 Debugging Tips

1. **Log classifications** to see if routing is correct
2. **Test edge cases** that might be misclassified
3. **Monitor route distribution** (are some routes unused?)
4. **A/B test routing strategies**
5. **Collect feedback** on routing accuracy

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)
- **Related Patterns**:
  - Pattern 3 (Parallelization) - Parallel routing
  - Pattern 7 (Multi-Agent) - Specialized agents

---

## 🎓 Next Steps

1. ✅ Run the basic routing example
2. ✅ Try different query types
3. ✅ Add a custom route
4. ✅ Implement semantic routing
5. ✅ Test with real user queries

---

**Pattern Type**: Intelligent Dispatch

**Complexity**: ⭐⭐ (Beginner-Friendly)

**Best For**: Multi-domain applications, cost optimization
