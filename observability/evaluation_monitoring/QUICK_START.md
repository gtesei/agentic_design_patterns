# Evaluation and Monitoring - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd observability/evaluation_monitoring
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
- **Option 1**: Basic Monitoring (metrics and logs)
- **Option 2**: Advanced Evaluation (LLM-as-judge quality assessment)
- **Option 3**: Run all examples

---

## 📖 Understanding Evaluation & Monitoring in 30 Seconds

**Monitoring** tracks what your AI agent is doing:
- How fast? (latency)
- How much? (tokens, cost)
- How well? (success rate, quality)

**Evaluation** assesses quality:
- Is the output relevant?
- Is it coherent and helpful?
- Does it meet user needs?

Together they provide complete visibility into AI agent performance.

---

## 🛠️ What Gets Monitored

### Basic Monitoring
- **Performance**: Latency, throughput
- **Cost**: Token usage, API costs
- **Success**: Completion rate, error rate
- **Usage**: Tool calls, cache hits

### Advanced Evaluation
- **Relevance**: Does output address the query?
- **Coherence**: Is output logical and well-structured?
- **Helpfulness**: Does it meet user needs?
- **Factuality**: Are claims accurate?

---

## 💡 Example Queries to Try

### Customer Support Query
```
"My order hasn't arrived. Order #12345 placed last week."
```

Monitors:
- Response time
- Tool usage (order lookup)
- Quality scores (empathy, helpfulness)
- User satisfaction

### Research Question
```
"What are the key differences between GPT-4 and Claude?"
```

Monitors:
- Search tool calls
- Information relevance
- Answer completeness
- Token efficiency

### Complex Task
```
"Analyze sales data for Q4 and suggest improvements."
```

Monitors:
- Multi-step execution
- Tool coordination
- Analysis quality
- Cost per request

---

## 📊 Key Metrics Explained

### Latency
Time from request to response.
- **Target**: < 2 seconds for interactive agents
- **Why it matters**: User experience

### Token Usage
Number of input + output tokens.
- **Target**: Minimize while maintaining quality
- **Why it matters**: Directly impacts cost

### Cost
Dollar amount per request.
- **Target**: Stay within budget
- **Why it matters**: Financial sustainability

### Quality Scores (0-10)
AI-generated assessment of output quality.
- **Target**: > 8.0 for production
- **Why it matters**: User satisfaction

### Success Rate
Percentage of completed tasks.
- **Target**: > 95%
- **Why it matters**: Reliability indicator

---

## 🎯 Basic vs Advanced Examples

| Feature | Basic | Advanced |
|---------|-------|----------|
| Metrics | Performance only | Performance + Quality |
| Evaluation | None | LLM-as-judge |
| Visualization | Simple tables | Charts and trends |
| Alerts | None | Anomaly detection |
| Complexity | Low | Medium |
| Use Case | Development | Production |

**Recommendation**: Start with Basic to understand metrics, then use Advanced for production monitoring.

---

## 🔧 Customization Tips

### Add Custom Metrics

```python
from src.monitoring_basic import MetricsCollector

collector = MetricsCollector()

# Add your metric
collector.add_metric({
    "name": "custom_metric",
    "value": 123,
    "timestamp": time.time()
})
```

### Configure Evaluation

```python
from src.monitoring_advanced import QualityEvaluator

evaluator = QualityEvaluator()

# Customize evaluation
scores = evaluator.evaluate(
    query="Your query",
    response="Agent response",
    dimensions=["relevance", "coherence", "helpfulness"]
)
```

### Set Alert Thresholds

Edit in `monitoring_advanced.py`:
```python
ALERT_THRESHOLDS = {
    "latency_ms": 5000,      # Alert if > 5s
    "error_rate": 0.05,      # Alert if > 5%
    "quality_score": 7.0,    # Alert if < 7.0
    "cost_per_request": 0.10 # Alert if > $0.10
}
```

---

## ⚡ Common Issues & Solutions

### Issue: High Latency
**Symptoms**: Responses taking > 5 seconds
**Solutions**:
- Use faster model (gpt-4o-mini vs gpt-4)
- Reduce prompt size
- Implement caching
- Parallelize tool calls

### Issue: High Costs
**Symptoms**: Token usage or $ per request too high
**Solutions**:
- Shorten prompts
- Use cheaper models for simple tasks
- Implement caching
- Reduce redundant tool calls

### Issue: Low Quality Scores
**Symptoms**: Relevance or coherence < 7.0
**Solutions**:
- Improve prompt engineering
- Use higher-quality model
- Add examples to prompts
- Refine tool descriptions

### Issue: Alert Fatigue
**Symptoms**: Too many notifications
**Solutions**:
- Increase alert thresholds
- Add alert cooldown periods
- Use severity levels
- Aggregate related alerts

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ **Start**: Run basic monitoring example
2. ✅ **Understand**: Review metrics collected (latency, tokens, cost)
3. ✅ **Observe**: See how metrics vary across requests
4. ✅ **Advance**: Run advanced example with quality evaluation
5. ✅ **Analyze**: Compare quality scores with performance metrics
6. ✅ **Customize**: Add your own metrics or evaluations
7. ✅ **Integrate**: Apply to your own agents

---

## 🌟 Pro Tips

### 1. Start Simple
Don't track everything immediately. Start with:
- Latency
- Token count
- Success rate

Add more as needs arise.

### 2. Quality Over Quantity
Better to track 5 meaningful metrics well than 50 poorly.

### 3. Visualize Trends
Single metrics are less valuable than trends over time.

### 4. Act on Insights
Monitoring without action is waste. Set up alerts and runbooks.

### 5. Sample Expensive Evaluations
LLM-as-judge is costly. Evaluate 10-20% of requests, not 100%.

### 6. Privacy First
Redact PII from logs. Use hashed user IDs.

### 7. Baseline Everything
Can't detect degradation without knowing normal performance.

---

## 📋 Quick Reference

### Basic Monitoring Flow
```
Request → Agent → Response
            ↓
      Collect Metrics
            ↓
    Log and Aggregate
            ↓
      Display Dashboard
```

### Advanced Evaluation Flow
```
Request → Agent → Response
            ↓
    Collect Metrics + Evaluate Quality
            ↓
    Detect Anomalies + Alert if needed
            ↓
    Update Trends + Display Dashboard
```

---

## 🚦 When to Alert

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Latency | < 2s | 2-5s | > 5s |
| Error Rate | < 1% | 1-5% | > 5% |
| Quality | > 8.0 | 7.0-8.0 | < 7.0 |
| Cost/Request | < $0.05 | $0.05-$0.10 | > $0.10 |

---

## 🎬 Example Output

### Basic Monitoring
```
=== Agent Metrics Report ===

Request: "What's the weather in Paris?"
Response: "Currently in Paris: 15°C, partly cloudy."

Performance Metrics:
  Latency: 850ms
  Tokens: 156 (35 input + 121 output)
  Cost: $0.0023
  Success: ✓

Tool Usage:
  weather_api: 1 call (780ms)

Status: ✓ All metrics within targets
```

### Advanced Evaluation
```
=== Quality Evaluation Report ===

Request: "Explain quantum computing"
Response: "Quantum computing uses quantum mechanics..."

Performance:
  Latency: 1.2s | Tokens: 342 | Cost: $0.0051

Quality Scores:
  Relevance:   9.0/10 ✓
  Coherence:   8.5/10 ✓
  Helpfulness: 8.8/10 ✓
  Overall:     8.8/10 ✓

Anomaly Check: No anomalies detected

Status: ✓ High quality response
```

---

**Happy Monitoring! 📊**

For questions or issues, refer to the full [README.md](./README.md).
