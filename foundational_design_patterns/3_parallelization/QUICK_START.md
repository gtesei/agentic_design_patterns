# Parallelization - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd foundational_design_patterns/3_parallelization
```

### Step 2: Run the Example
```bash
bash run.sh
```

---

## 📖 Understanding Parallelization in 30 Seconds

**Parallelization** executes independent tasks simultaneously:

```
Sequential (15s):          Parallel (5s):
Task A (5s) →             Task A (5s) ↘
Task B (5s) →      vs.    Task B (5s) → Combine → Output
Task C (5s) → Output      Task C (5s) ↗
```

**Result**: 2-10x faster execution for independent I/O-bound tasks!

---

## 🎯 What This Example Does

The example demonstrates **parallel API calls**:

1. **Analyze sentiment** of multiple documents
2. **Fetch data** from multiple sources
3. **Process texts** in parallel
4. **Combine results** into final output

---

## 💡 Example Scenarios

### Scenario 1: Multi-Document Analysis
```
Input: 5 documents
Sequential: 5 × 3s = 15 seconds
Parallel:   1 × 3s = 3 seconds (5x faster!)
```

### Scenario 2: Multi-Source Research
```
Query: "What's the weather in 3 cities?"
Sequential: 3 API calls × 2s = 6 seconds
Parallel:   3 API calls at once = 2 seconds
```

---

## 🔧 Key Concepts

### Independence
Tasks must not depend on each other's results.

### I/O-Bound Operations
Best for network calls, API requests, file reads (not CPU-heavy tasks).

### Concurrency
Multiple operations in progress at the same time.

### Result Aggregation
Combine parallel results into final output.

---

## 🎨 When to Use Parallelization

✅ **Good For:**
- Multiple API calls (search engines, databases)
- Multi-document processing
- Multi-source data retrieval
- Independent analysis tasks
- Batch operations

❌ **Not Ideal For:**
- Sequential dependencies (Task B needs Task A's result)
- CPU-bound operations
- Single task execution
- Tasks with shared state

---

## 🛠️ Implementation Approaches

### 1. LCEL Parallel (LangChain)
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    sentiment=sentiment_chain,
    summary=summary_chain,
    keywords=keyword_chain
)

result = parallel.invoke({"text": doc})
# All three run simultaneously!
```

### 2. Async/Await (Python)
```python
import asyncio

async def process_all(docs):
    tasks = [process_doc(doc) for doc in docs]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. ThreadPoolExecutor
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_doc, docs))
```

---

## 📊 Performance Gains

| Task Count | Sequential | Parallel | Speedup |
|------------|-----------|----------|---------|
| 2 tasks    | 6s        | 3s       | 2x      |
| 5 tasks    | 15s       | 3s       | 5x      |
| 10 tasks   | 30s       | 3s       | 10x     |

**Note**: Actual speedup depends on task duration and I/O vs CPU usage.

---

## 💡 Common Patterns

### Map-Reduce
```
Input: [doc1, doc2, doc3]
    ↓
Map (parallel): [result1, result2, result3]
    ↓
Reduce: Combined final result
```

### Fan-Out/Fan-In
```
Query → [API1, API2, API3] → Merge → Response
```

### Parallel Chains
```
Input → [Chain A, Chain B, Chain C] → Aggregate → Output
```

---

## 🐛 Common Issues & Solutions

### Issue: Tasks Taking Different Times
**Solution**: Results return as the slowest task completes. Consider timeouts.

### Issue: Too Many Parallel Tasks
**Solution**: Limit concurrency to avoid overwhelming APIs:
```python
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
```

### Issue: Error in One Task
**Solution**: Use try/except to handle errors without breaking others:
```python
async def safe_process(doc):
    try:
        return await process(doc)
    except Exception as e:
        return f"Error: {e}"
```

---

## 🔧 Customization Tips

### Process Multiple Documents
```python
documents = [
    "Document 1 content...",
    "Document 2 content...",
    "Document 3 content...",
]

# Parallel processing
results = parallel_chain.batch(documents)
```

### Add Custom Processing
```python
def custom_analysis(text: str) -> dict:
    """Your custom analysis function"""
    return {"metric": calculate_metric(text)}

parallel = RunnableParallel(
    sentiment=sentiment_chain,
    custom=custom_analysis  # Add your function
)
```

### Limit Concurrency
```python
# Process 3 at a time instead of all at once
async def process_in_batches(docs, batch_size=3):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        await asyncio.gather(*[process(d) for d in batch])
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)
- **Related Patterns**:
  - Pattern 2 (Routing) - Parallel route evaluation
  - Pattern 7 (Multi-Agent) - Parallel agent execution

---

## 🎓 Next Steps

1. ✅ Run the parallel processing example
2. ✅ Compare timing vs sequential
3. ✅ Add more parallel tasks
4. ✅ Implement error handling
5. ✅ Try async/await approach

---

**Pattern Type**: Concurrent Execution

**Complexity**: ⭐⭐⭐ (Intermediate)

**Best For**: Multiple independent I/O operations

**Speedup**: 2-10x for I/O-bound tasks
