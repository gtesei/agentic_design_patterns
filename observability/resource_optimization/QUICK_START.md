# Resource Optimization - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd observability/resource_optimization
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
- **Option 1**: Basic Optimization (caching + prompt optimization + model routing)
- **Option 2**: Advanced Optimization (batching + predictive caching + cost-aware routing)
- **Option 3**: Run all examples

---

## 📖 Understanding Resource Optimization in 30 Seconds

**Goal**: Reduce costs and improve performance without sacrificing quality

**Key Techniques**:
1. **Caching**: Store responses, reuse for identical/similar queries (60-80% cost reduction)
2. **Model Routing**: Use cheaper models for simple queries (10-60x cost savings)
3. **Prompt Optimization**: Compress prompts to reduce tokens (20-40% savings)
4. **Batching**: Process multiple requests in parallel (2-5x throughput)

**Example Impact**:
- Before: $45K/month, 2.5s P95 latency
- After: $12K/month (73% savings), 450ms P95 latency (5.5x faster)

---

## 💰 Cost Savings Breakdown

### Without Optimization
```
Query: "What is Python?"
→ GPT-4 call every time
→ 800ms latency, $0.015 cost
→ 100K daily queries = $1,500/day = $45K/month
```

### With Optimization
```
Query 1: "What is Python?"
→ GPT-4 call: 800ms, $0.015 (cache miss)
→ ✅ Cached for 1 hour

Query 2: "What is Python?" (5 min later)
→ Cache hit: 50ms, $0.000 (95% faster, 100% cheaper!)

Query 3: "Explain Python" (similar)
→ Semantic cache hit: 150ms, $0.002 (adapted from cache)

Query 4: "What's 2+2?" (simple)
→ Routed to GPT-3.5: 300ms, $0.002 (60x cheaper than GPT-4)

Average: 65% cache hit rate + 75% cheap model routing
→ $525/day = $15.7K/month
→ Savings: $29.3K/month (65% reduction!)
```

---

## 🎯 What Each Example Demonstrates

### Basic Optimization (`optimization_basic.py`)
- **Response Caching**: LRU cache with TTL (Time-To-Live)
- **Prompt Optimization**: Reduce token usage by 20-40%
- **Model Selection**: Route simple queries to GPT-3.5-turbo
- **Metrics Tracking**: Cost savings, cache hit rate, latency improvements
- **Visualization**: Before/after comparison charts

**Use this when**: Starting with optimization, need immediate cost reduction

### Advanced Optimization (`optimization_advanced.py`)
- **Request Batching**: Process multiple queries in parallel
- **Cost-Aware Routing**: Dynamic model selection with quality fallback
- **Adaptive Optimization**: Learn from patterns, adjust strategies
- **Predictive Caching**: Pre-cache likely next queries
- **A/B Testing**: Compare optimization strategies
- **Rich Dashboard**: Real-time cost/quality trade-offs

**Use this when**: Need maximum optimization, high-volume production systems

---

## 📊 Key Metrics to Watch

### Cost Metrics
- **Cost per request**: Target 40-60% reduction
- **Monthly spend**: Track total LLM costs
- **Model distribution**: % queries per model tier

### Performance Metrics
- **Cache hit rate**: Target > 30% (higher is better)
- **P95 latency**: Target < 1 second
- **Throughput**: Requests per second

### Quality Metrics
- **Output quality score**: Maintain > 90% of baseline
- **User satisfaction**: Track ratings, feedback
- **Task success rate**: % queries achieving user goal

---

## 💡 Example Queries to Try

### Simple Queries (Should use GPT-3.5-turbo)
```
"What is Python?"
"Define machine learning"
"When was Python created?"
"Who invented the telephone?"
```

### Medium Queries (Should use GPT-4o-mini)
```
"Explain how neural networks work"
"Summarize the benefits of cloud computing"
"Why is Python popular for data science?"
```

### Complex Queries (Should use GPT-4)
```
"Compare Python and Java for enterprise applications"
"Design a microservices architecture for e-commerce"
"Analyze the trade-offs between SQL and NoSQL databases"
```

### Repeated Queries (Should hit cache)
```
# Run the same query twice
"What is Python?"  # Cache miss
"What is Python?"  # Cache hit (instant!)
```

---

## 🔧 Configuration Options

### Adjust Cache Size and TTL
```python
# In optimization_basic.py
cache = ResponseCache(
    ttl_seconds=3600,  # 1 hour (adjust based on data freshness needs)
    max_size=1000      # Max cached entries
)
```

### Customize Model Selection
```python
# In optimization_advanced.py
def select_model(query: str) -> str:
    complexity = analyze_complexity(query)

    if complexity < 3:
        return "gpt-3.5-turbo"  # Simple queries
    elif complexity < 7:
        return "gpt-4o-mini"    # Medium queries
    else:
        return "gpt-4"          # Complex queries
```

### Modify Batch Size
```python
# In optimization_advanced.py
batch_processor = BatchProcessor(
    batch_size=10,   # Queries per batch
    wait_ms=100      # Max wait time before processing
)
```

---

## ⚡ Common Use Cases

### Use Case 1: FAQ Chatbot
**Problem**: Answering same questions repeatedly
**Solution**: Caching with 1-hour TTL
**Expected Impact**:
- Cache hit rate: 60-70%
- Cost reduction: 60-70%
- Latency improvement: 10x for cached queries

### Use Case 2: Customer Support
**Problem**: High volume, mixed complexity queries
**Solution**: Model routing + caching
**Expected Impact**:
- 75% queries use cheap models
- 50% cache hit rate
- Total cost reduction: 70-80%

### Use Case 3: Content Generation Pipeline
**Problem**: Slow sequential processing
**Solution**: Batching + prompt optimization
**Expected Impact**:
- Throughput: 5-10x increase
- Token usage: 30-40% reduction
- Processing time: 80-90% faster

---

## 🚨 Common Issues & Solutions

### Issue: Low Cache Hit Rate (< 20%)
**Cause**: Queries too unique or varied
**Solution**:
- Implement semantic caching (similarity matching)
- Normalize queries (case, whitespace)
- Increase cache size and TTL
- Check if caching is appropriate for your use case

### Issue: Quality Degradation
**Cause**: Cheap models or stale cache hurting accuracy
**Solution**:
- Use quality thresholds with model fallback
- Implement A/B testing before full rollout
- Monitor user satisfaction scores
- Reduce cache TTL for time-sensitive data

### Issue: Increased Latency
**Cause**: Cache lookup or routing overhead exceeds benefits
**Solution**:
- Optimize cache data structure (hash maps)
- Simplify routing logic
- Profile to identify bottlenecks
- Consider if optimization is worth the complexity

### Issue: Over-Budget
**Cause**: More cache misses or complex queries than expected
**Solution**:
- Analyze query patterns and adjust strategy
- Implement budget limits per request
- Use predictive caching for common queries
- Pre-warm cache during low-traffic periods

---

## 📈 Optimization Strategy Roadmap

### Phase 1: Baseline (Week 1)
- [ ] Profile current performance (cost, latency, quality)
- [ ] Identify optimization opportunities
- [ ] Set target metrics (cost reduction, latency goals)

### Phase 2: Quick Wins (Week 2-3)
- [ ] Implement basic LRU caching
- [ ] Add prompt optimization (remove filler words)
- [ ] Monitor cache hit rate and cost savings

### Phase 3: Model Routing (Week 4-5)
- [ ] Analyze query complexity distribution
- [ ] Implement tiered model selection
- [ ] A/B test quality impact

### Phase 4: Advanced Optimization (Week 6+)
- [ ] Add semantic caching for higher hit rate
- [ ] Implement batching for throughput
- [ ] Add predictive caching based on patterns
- [ ] Set up cost-aware quality thresholds

---

## 🎓 Learning Path

1. **Understand**: Read the baseline metrics in console output
2. **Observe**: Watch cache hits/misses and model routing in action
3. **Compare**: See before/after cost and latency improvements
4. **Experiment**: Try different query types and observe routing decisions
5. **Customize**: Adjust cache TTL, model thresholds, batch size
6. **Integrate**: Apply techniques to your own applications

---

## 🌟 Pro Tips

### Tip 1: Start with Caching
Caching provides the highest ROI with lowest complexity. Implement caching first, then add other optimizations.

### Tip 2: Monitor Quality Continuously
Cost/speed improvements mean nothing if quality degrades. Track user satisfaction alongside technical metrics.

### Tip 3: Use Appropriate TTL
- **Static content**: 24 hours or longer
- **Semi-static**: 1-6 hours
- **Dynamic content**: 5-30 minutes
- **Real-time data**: Don't cache!

### Tip 4: Pre-warm Cache
Prime your cache with common queries before launching or during low-traffic periods.

### Tip 5: Gradual Rollout
Test optimizations with 10% of traffic first, then gradually increase as you validate quality and cost impact.

### Tip 6: Document Trade-offs
Make cost vs. quality trade-offs explicit and document the rationale for routing decisions.

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Code Examples**: See `src/optimization_basic.py` and `src/optimization_advanced.py`
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎯 Success Checklist

Before deploying to production:
- [ ] Baseline metrics measured and documented
- [ ] Caching implemented with appropriate TTL
- [ ] Cache hit rate > 30% (or determined not applicable)
- [ ] Model routing logic tested and validated
- [ ] Quality metrics monitored (maintain > 90% of baseline)
- [ ] Cost savings validated (target: 40-60% reduction)
- [ ] Latency improvements measured (target: 2-5x for cached)
- [ ] A/B testing completed with statistical significance
- [ ] Alerting set up for cache hit rate drops, cost spikes
- [ ] Documentation updated with optimization strategies

---

**Happy Optimizing! 💰⚡**

For questions or issues, refer to the full [README.md](./README.md).
