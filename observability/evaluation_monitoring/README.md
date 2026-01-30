# Evaluation and Monitoring

## Overview

The **Evaluation and Monitoring Pattern** provides systematic instrumentation, measurement, and analysis of AI agent performance through comprehensive metrics tracking, logging, and quality evaluation. This pattern transforms opaque AI systems into observable, measurable, and continuously improving applications.

Unlike traditional software monitoring that focuses primarily on system metrics (CPU, memory, uptime), AI agent monitoring must also track quality metrics (accuracy, relevance, coherence), behavioral patterns (tool usage, reasoning effectiveness), and business outcomes (user satisfaction, task completion rates, cost efficiency).

## Why Use This Pattern?

AI systems present unique monitoring challenges:

- **Non-deterministic behavior**: Same input can produce different outputs
- **Quality assessment complexity**: "Correct" answers often subjective or context-dependent
- **Multi-dimensional success**: Speed, accuracy, cost, and user satisfaction all matter
- **Silent failures**: Agents may produce plausible but incorrect or biased outputs
- **Cost transparency**: Token usage and API costs must be tracked and optimized

This pattern solves these by:
- **Comprehensive metrics collection**: Quantitative (latency, tokens, cost) and qualitative (relevance, coherence) measures
- **Real-time observability**: Live dashboards and alerts for immediate visibility
- **Quality evaluation frameworks**: LLM-as-judge and automated evaluation pipelines
- **Trend analysis**: Historical tracking to detect performance degradation
- **Data-driven optimization**: Metrics inform model selection, prompt engineering, and architecture decisions
- **Anomaly detection**: Automated identification of performance issues

### Example: Chatbot Without vs. With Monitoring

```
Without Monitoring:
User: "What's the weather in Paris?"
Agent: "Paris typically has mild weather."
→ Response seems OK, but may be outdated, vague, or unhelpful
→ No visibility into latency, cost, or user satisfaction
→ Performance issues go undetected until users complain

With Monitoring:
User: "What's the weather in Paris?"
Agent: "Currently in Paris: 15°C, partly cloudy, 60% humidity."

Metrics Captured:
✓ Latency: 850ms (within 2s SLA)
✓ Tokens: 156 (35 input + 121 output)
✓ Cost: $0.0023
✓ Tool Calls: weather_api (success)
✓ Relevance Score: 9.2/10 (LLM judge)
✓ User Satisfaction: Positive (thumbs up)
✓ Response Completeness: High

Dashboard Alert:
⚠️ Average latency increased 25% in last hour
→ Investigation reveals API slowdown, switch to backup provider
```

## How It Works

The monitoring pipeline follows a structured flow:

1. **Instrument**: Add tracking to agent execution (before/after tool calls, LLM invocations)
2. **Collect**: Gather metrics during execution (latency, tokens, tool usage, outputs)
3. **Evaluate**: Assess quality using automated evaluators (LLM-as-judge, heuristics, rubrics)
4. **Aggregate**: Compute statistics (averages, percentiles, trends) over time windows
5. **Visualize**: Present metrics in dashboards (charts, tables, alerts)
6. **Alert**: Notify on anomalies (degraded performance, high costs, failures)
7. **Optimize**: Use insights to improve prompts, models, or architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                          │
└──────────────────────┬──────────────────────────────────┘
                       ↓
              ┌────────────────┐
              │  Instrumented   │ ← Start timer, log input
              │  Agent Entry    │
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │  Agent Logic    │ ← Track tool calls, LLM invocations
              │  Execution      │
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │  Metrics        │ ← Latency, tokens, cost, success
              │  Collection     │
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │  Quality        │ ← LLM-as-judge, heuristics
              │  Evaluation     │
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │  Aggregation    │ ← Stats, trends, percentiles
              │  & Storage      │
              └────────┬───────┘
                       ↓
         ┌────────────┴────────────┐
         ↓                         ↓
┌────────────────┐      ┌────────────────┐
│  Real-time     │      │  Alerting      │
│  Dashboard     │      │  System        │
└────────────────┘      └────────────────┘
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Production AI systems**: Any agent deployed to real users requires monitoring
- **Quality-critical applications**: Healthcare, finance, legal domains where accuracy matters
- **Cost-sensitive deployments**: Track token usage to optimize expenses
- **Continuous deployment**: Monitor impact of prompt/model changes
- **SLA enforcement**: Ensure latency and availability targets are met
- **User-facing agents**: Track user satisfaction and task completion rates
- **Multi-agent systems**: Understand agent interactions and bottlenecks
- **RAG pipelines**: Monitor retrieval relevance and generation quality
- **A/B testing**: Compare different prompts, models, or configurations
- **Compliance requirements**: Audit trails for regulated industries

### ❌ When NOT to Use (or Use Minimal Monitoring)

- **One-off experiments**: Quick prototypes that won't be maintained
- **Offline analysis**: Batch processing where real-time monitoring isn't needed
- **Extremely latency-sensitive**: When even minimal logging overhead is unacceptable
- **No quality concerns**: Simple deterministic tasks with clear success criteria
- **Local development only**: Though basic logging still helps debugging

## Rule of Thumb

**Use comprehensive monitoring when:**
1. System is **deployed to production** with real users
2. **Quality or cost** are critical concerns
3. You need to **detect performance degradation** over time
4. System involves **multiple components** (tools, models, retrievers)
5. **Compliance or auditing** requirements exist

**Use minimal monitoring when:**
1. Rapid prototyping phase (add later before deployment)
2. Tasks are simple and deterministic
3. Offline batch processing with post-hoc analysis
4. Extreme latency requirements prohibit any overhead

## Core Components

### 1. Quantitative Metrics

Objective, numerical measurements:

**Performance Metrics:**
- Latency (end-to-end, per-component)
- Throughput (requests/second)
- Token usage (input/output counts)
- API cost ($ per request)
- Cache hit rates

**Success Metrics:**
- Task completion rate
- Error rate (by error type)
- Timeout rate
- Retry rate

**Resource Metrics:**
- Tool invocation counts
- Tool success/failure rates
- Database query counts
- API rate limit consumption

### 2. Qualitative Metrics

Subjective quality assessments:

**LLM-as-Judge Evaluations:**
- Relevance (does output address the query?)
- Coherence (is output logical and well-structured?)
- Helpfulness (does it meet user needs?)
- Factuality (are claims accurate?)
- Safety (is content appropriate and unbiased?)

**Human Evaluations:**
- User satisfaction ratings
- Thumbs up/down feedback
- Follow-up question necessity
- Task completion confirmation

### 3. Structured Logging

Contextual information capture:

**Log Levels:**
- DEBUG: Detailed execution traces
- INFO: Key events (task start/end, tool calls)
- WARNING: Recoverable issues (retries, fallbacks)
- ERROR: Failures requiring attention

**Log Context:**
- Request ID (trace across components)
- User ID (privacy-safe identifier)
- Session ID (group related interactions)
- Agent configuration (model, temperature, tools)
- Timestamps (high-resolution)

### 4. Evaluation Frameworks

Automated quality assessment:

**Rule-based Evaluators:**
- Response length checks
- Required keyword presence
- Format validation (JSON, specific structure)
- Sentiment analysis

**Model-based Evaluators:**
- LLM-as-judge (GPT-4 evaluates GPT-3.5 outputs)
- Embedding similarity (output vs. ground truth)
- Classification (topic, intent, toxicity)

**Statistical Evaluators:**
- Distribution shift detection
- Outlier identification
- Regression testing (compare to baseline)

### 5. Dashboards and Visualization

Insights presentation:

**Real-time Dashboards:**
- Current metrics (rolling averages)
- Active request monitoring
- Error rate trends
- Cost burndown

**Historical Analysis:**
- Time-series charts (latency, quality over days/weeks)
- Percentile distributions (P50, P95, P99)
- Comparison views (before/after changes)
- Correlation analysis (quality vs. latency, etc.)

### 6. Alerting Systems

Proactive issue detection:

**Threshold-based Alerts:**
- Latency > 5 seconds
- Error rate > 5%
- Hourly cost > $50
- Quality score < 7/10

**Anomaly-based Alerts:**
- Sudden latency spike (> 2 std devs)
- Quality drop (> 20% decline)
- Unusual tool usage pattern
- Token usage surge

## Implementation Approaches

### Approach 1: Decorator-based Monitoring

Minimal code changes, maximum reuse:

```python
from functools import wraps
import time
from typing import Any, Callable

class MetricsCollector:
    def __init__(self):
        self.metrics = []

    def track_invocation(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                latency = time.time() - start_time

                self.metrics.append({
                    "function": func.__name__,
                    "latency_ms": latency * 1000,
                    "success": success,
                    "error": error,
                    "timestamp": time.time()
                })

            return result
        return wrapper

# Usage
collector = MetricsCollector()

@collector.track_invocation
def my_agent_function(query: str):
    # Agent logic here
    return response
```

### Approach 2: Context Manager for Request Tracking

Explicit scope control:

```python
class MonitoringContext:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.metrics = {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics["latency"] = time.time() - self.start_time
        self.metrics["success"] = exc_type is None
        # Log metrics
        logger.info(f"Request {self.request_id}: {self.metrics}")

# Usage
with MonitoringContext(request_id="req-123") as ctx:
    result = agent.invoke(query)
    ctx.metrics["tokens"] = result.usage.total_tokens
```

### Approach 3: Custom Callbacks with LangChain

Integrate with LangChain/LangGraph:

```python
from langchain.callbacks.base import BaseCallbackHandler

class MetricsCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.metrics = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_llm_call = {
            "start_time": time.time(),
            "prompts": prompts
        }

    def on_llm_end(self, response, **kwargs):
        latency = time.time() - self.current_llm_call["start_time"]
        tokens = response.llm_output.get("token_usage", {})

        self.metrics.append({
            "type": "llm_call",
            "latency_ms": latency * 1000,
            "tokens": tokens,
            "timestamp": time.time()
        })

    def on_tool_start(self, serialized, input_str, **kwargs):
        # Track tool usage
        pass

    def on_tool_end(self, output, **kwargs):
        # Track tool results
        pass

# Usage
callback = MetricsCallbackHandler()
agent.invoke(query, config={"callbacks": [callback]})
```

### Approach 4: LLM-as-Judge Evaluation

Use AI to evaluate AI:

```python
from langchain_openai import ChatOpenAI

class LLMJudge:
    def __init__(self):
        self.judge_llm = ChatOpenAI(model="gpt-4", temperature=0)

    def evaluate_relevance(self, query: str, response: str) -> dict:
        prompt = f"""Evaluate the relevance of this response to the query.

Query: {query}
Response: {response}

Rate the relevance from 1-10 and provide a brief explanation.
Format your response as:
Score: [1-10]
Reasoning: [Your explanation]"""

        result = self.judge_llm.invoke(prompt)
        # Parse score and reasoning
        return {
            "metric": "relevance",
            "score": parsed_score,
            "reasoning": parsed_reasoning
        }
```

## Key Benefits

### 🔍 Visibility into AI Systems

**Benefit**: Understand what's happening inside the "black box"

**Example**: See that agent is calling search 5x per query (optimization opportunity)

**Impact**: Enables debugging, optimization, and trust-building

### 📊 Data-Driven Decision Making

**Benefit**: Choose models, prompts, and architectures based on evidence, not intuition

**Example**: A/B test shows GPT-4o-mini has 90% of GPT-4's quality at 10% of the cost

**Impact**: Optimize cost/quality tradeoffs systematically

### 🚨 Early Issue Detection

**Benefit**: Catch problems before users notice or complain

**Example**: Alert triggers when latency spikes from 800ms to 3s due to API slowdown

**Impact**: Proactive incident response reduces user impact

### 📈 Continuous Improvement

**Benefit**: Track performance over time and measure impact of changes

**Example**: New prompt increases relevance score from 7.2 to 8.9

**Impact**: Systematic, measurable progress on quality and efficiency

### 💰 Cost Management

**Benefit**: Understand and control LLM API expenses

**Example**: Dashboard shows 60% of tokens used in redundant tool calls

**Impact**: Identify optimization opportunities, prevent cost overruns

### 🛡️ Risk Mitigation

**Benefit**: Detect safety, bias, and compliance issues

**Example**: Monitoring catches agent occasionally generating harmful content

**Impact**: Maintain trust and regulatory compliance

## Trade-offs

### ⚖️ Monitoring Overhead

**Issue**: Metrics collection, evaluation, and logging add latency and complexity

**Impact**:
- 50-200ms additional latency per request (for comprehensive monitoring)
- Extra API calls for LLM-as-judge evaluation
- Storage and processing costs for metrics

**Mitigation**:
- Use async logging (don't block main execution)
- Sample evaluation (don't judge every response)
- Efficient metrics storage (time-series databases)
- Separate monitoring pipeline from critical path

### 🔐 Privacy and Security Concerns

**Issue**: Logging user queries and agent responses may expose sensitive data

**Impact**: Compliance risks (GDPR, HIPAA), user trust issues

**Mitigation**:
- Redact PII before logging
- Use privacy-safe identifiers (hashed user IDs)
- Implement data retention policies
- Provide user opt-out mechanisms
- Encrypt logs at rest and in transit

### 📊 Alert Fatigue

**Issue**: Too many alerts lead to ignoring important ones

**Impact**: Critical issues missed due to notification overload

**Mitigation**:
- Tune alert thresholds carefully
- Use alert severity levels
- Implement alert aggregation (multiple related alerts → one notification)
- On-call rotation to distribute alert burden
- Regular alert review and refinement

### 🔍 Metric Selection Complexity

**Issue**: Choosing what to measure is challenging with many possible metrics

**Impact**: Missing critical metrics or tracking irrelevant ones

**Mitigation**:
- Start with core metrics (latency, cost, success rate, quality)
- Add metrics based on specific goals and issues
- Review metrics quarterly, remove unused ones
- Balance quantitative and qualitative measures

### 💾 Storage and Processing Costs

**Issue**: Metrics and logs accumulate quickly, especially at scale

**Impact**: Storage costs can exceed monitoring benefits

**Mitigation**:
- Implement retention policies (e.g., detailed logs for 7 days, aggregates for 90 days)
- Use efficient storage (columnar formats, compression)
- Archive cold data to cheaper storage
- Sample high-volume, low-value data

## Best Practices

### 1. Define Clear Metrics Early

```python
# ✅ Good: Explicit metrics definition
class AgentMetrics:
    # Performance
    latency_ms: float
    token_count: int
    cost_usd: float

    # Quality
    relevance_score: float  # 0-10
    coherence_score: float  # 0-10
    user_satisfaction: Optional[str]  # thumbs up/down

    # Success
    task_completed: bool
    error_type: Optional[str]
    retry_count: int

# ❌ Bad: Ad-hoc metrics
metrics = {"some_time": 123, "stuff": "yes", "thing": 456}
```

### 2. Use Structured Logging

```python
import logging
import json

# ✅ Good: Structured logs
logger.info(json.dumps({
    "event": "agent_invocation",
    "request_id": "req-123",
    "user_id": "user-456",
    "query": "What's the weather?",
    "latency_ms": 850,
    "tokens": 156,
    "success": True,
    "timestamp": "2024-01-30T10:30:00Z"
}))

# ❌ Bad: Unstructured logs
logger.info(f"Agent took 850ms to process weather query for user 456")
```

### 3. Separate Monitoring from Business Logic

```python
# ✅ Good: Clean separation
def process_query(query: str) -> str:
    return agent.invoke(query)

def monitored_process_query(query: str) -> tuple[str, dict]:
    metrics = {}
    start = time.time()

    result = process_query(query)

    metrics["latency_ms"] = (time.time() - start) * 1000
    metrics["success"] = True

    return result, metrics

# ❌ Bad: Monitoring mixed with logic
def process_query(query: str) -> str:
    start = time.time()  # Monitoring code mixed in
    result = agent.invoke(query)
    print(f"Took {time.time() - start}s")  # Mixed in
    return result
```

### 4. Implement Graduated Alerting

```python
class AlertSeverity:
    INFO = "info"        # FYI, no action needed
    WARNING = "warning"  # Review when convenient
    ERROR = "error"      # Investigate within hours
    CRITICAL = "critical"  # Page on-call immediately

def check_metrics(metrics: dict):
    if metrics["latency_ms"] > 10000:
        alert(AlertSeverity.CRITICAL, "Latency >10s")
    elif metrics["latency_ms"] > 5000:
        alert(AlertSeverity.ERROR, "Latency >5s")
    elif metrics["latency_ms"] > 2000:
        alert(AlertSeverity.WARNING, "Latency >2s")
```

### 5. Use Sampling for Expensive Evaluations

```python
import random

def evaluate_if_sampled(query: str, response: str, sample_rate: float = 0.1):
    """Run expensive LLM-as-judge evaluation on sample of requests."""
    if random.random() < sample_rate:
        return llm_judge.evaluate(query, response)
    return None

# Evaluate 10% of responses for quality
quality_score = evaluate_if_sampled(query, response, sample_rate=0.1)
```

### 6. Track Metrics Distributions, Not Just Averages

```python
import numpy as np

class MetricsAggregator:
    def __init__(self):
        self.latencies = []

    def add_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def get_statistics(self) -> dict:
        return {
            "mean": np.mean(self.latencies),
            "median": np.median(self.latencies),
            "p95": np.percentile(self.latencies, 95),
            "p99": np.percentile(self.latencies, 99),
            "max": np.max(self.latencies),
            "min": np.min(self.latencies)
        }
```

## Performance Metrics

Key metrics to track for evaluation and monitoring systems:

### Quantitative Metrics

- **Latency**: End-to-end response time (P50, P95, P99)
  - Target: < 2s for interactive agents
  - Measurement: Time from request to response

- **Token Usage**: Total tokens (input + output)
  - Target: Minimize while maintaining quality
  - Measurement: Count from LLM API response

- **Cost**: $ per request, per day, per user
  - Target: Stay within budget constraints
  - Measurement: Tokens × model pricing

- **Throughput**: Requests per second handled
  - Target: Match user load with headroom
  - Measurement: Successful requests / time window

- **Error Rate**: % of failed requests
  - Target: < 1% (varies by criticality)
  - Measurement: Errors / total requests

- **Success Rate**: % of completed tasks
  - Target: > 95% (varies by task complexity)
  - Measurement: Successful completions / attempts

### Qualitative Metrics

- **Relevance**: Does output address the query? (0-10)
  - Target: > 8.0
  - Measurement: LLM-as-judge or human evaluation

- **Coherence**: Is output logical and well-structured? (0-10)
  - Target: > 8.5
  - Measurement: LLM-as-judge evaluation

- **Helpfulness**: Does it meet user needs? (0-10)
  - Target: > 8.0
  - Measurement: User feedback or LLM-as-judge

- **Factuality**: Are claims accurate? (0-10)
  - Target: > 9.0 for fact-critical domains
  - Measurement: Fact-checking or ground truth comparison

- **User Satisfaction**: Thumbs up/down, ratings
  - Target: > 80% positive
  - Measurement: Explicit user feedback

### System Health Metrics

- **Monitoring Coverage**: % of components instrumented
  - Target: 100% of critical paths

- **Alert Response Time**: Time to acknowledge alerts
  - Target: < 5 minutes for critical alerts

- **Mean Time to Detection (MTTD)**: Time to identify issues
  - Target: < 5 minutes

- **Mean Time to Resolution (MTTR)**: Time to fix issues
  - Target: < 1 hour for critical issues

## Example Scenarios

### Scenario 1: Basic Metrics Collection for Chatbot

```
Setup: Customer support chatbot with search and FAQ tools

Implementation:
- Track latency, tokens, cost per conversation
- Log tool usage (which tools called, how often)
- Capture success rate (conversation resolved vs. escalated)

Sample Metrics (1 hour):
┌───────────────────────┬──────────┐
│ Metric                │ Value    │
├───────────────────────┼──────────┤
│ Total Requests        │ 1,247    │
│ Avg Latency           │ 1.3s     │
│ P95 Latency           │ 2.8s     │
│ Total Tokens          │ 1.2M     │
│ Total Cost            │ $18.50   │
│ Success Rate          │ 92.3%    │
│ Escalation Rate       │ 7.7%     │
│ Avg Tools per Request │ 2.4      │
└───────────────────────┴──────────┘

Alert Triggered:
⚠️ Escalation rate increased from 5% to 7.7% (baseline: 5%)
→ Investigation: Recent questions about new product feature not in FAQ
→ Action: Add new FAQ entries, update knowledge base
```

### Scenario 2: Quality Monitoring for RAG System

```
Setup: Document Q&A system using RAG (retrieval-augmented generation)

Monitoring Focus:
- Retrieval quality (relevant docs found?)
- Generation quality (answer addresses question?)
- End-to-end relevance (user satisfied?)

Evaluation Pipeline:
1. Quantitative: Track retrieval scores, token usage
2. LLM-as-Judge: Evaluate relevance, coherence (10% sample)
3. User Feedback: Thumbs up/down buttons

Sample Quality Report (1 day):
┌────────────────────────┬─────────┬─────────┬──────────┐
│ Metric                 │ Today   │ 7d Avg  │ Target   │
├────────────────────────┼─────────┼─────────┼──────────┤
│ Retrieval Precision    │ 0.85    │ 0.87    │ > 0.80   │
│ Relevance Score        │ 8.2/10  │ 8.5/10  │ > 8.0    │
│ Coherence Score        │ 9.1/10  │ 9.0/10  │ > 8.5    │
│ User Satisfaction      │ 78%     │ 82%     │ > 80%    │
│ Answer Length (tokens) │ 145     │ 150     │ 100-200  │
└────────────────────────┴─────────┴─────────┴──────────┘

Insight:
User satisfaction slightly down despite good quality scores
→ Hypothesis: Answers too long/technical
→ A/B Test: Try shorter, simpler responses for comparison
```

### Scenario 3: Multi-Agent System Observability

```
Setup: Research agent system with specialized sub-agents (search, summarize, analyze)

Monitoring Challenges:
- Track metrics per agent
- Understand agent interaction patterns
- Identify bottlenecks in workflow

Dashboard View:
┌─────────────────────────────────────────────────────────┐
│ Agent Performance Overview (Last Hour)                  │
├──────────────┬──────────┬──────────┬──────────┬────────┤
│ Agent        │ Calls    │ Avg Lat. │ Success  │ Cost   │
├──────────────┼──────────┼──────────┼──────────┼────────┤
│ Orchestrator │ 523      │ 3.2s     │ 95%      │ $4.20  │
│ Search Agent │ 1,105    │ 1.1s     │ 98%      │ $8.30  │
│ Summarizer   │ 897      │ 2.5s     │ 97%      │ $15.40 │
│ Analyzer     │ 334      │ 4.8s     │ 91%      │ $12.10 │
└──────────────┴──────────┴──────────┴──────────┴────────┘

Alert:
🚨 Analyzer agent success rate dropped to 91% (baseline: 96%)
→ Investigation: Complex queries causing timeout
→ Action: Increase timeout, optimize prompt for conciseness
```

## Advanced Patterns

### 1. Automated Regression Testing

Continuously compare against baseline:

```python
class RegressionTester:
    def __init__(self, baseline_metrics: dict):
        self.baseline = baseline_metrics

    def check_regression(self, current_metrics: dict) -> list[str]:
        issues = []

        # Check quality degradation
        if current_metrics["relevance"] < self.baseline["relevance"] - 0.5:
            issues.append(f"Relevance dropped: {current_metrics['relevance']} vs {self.baseline['relevance']}")

        # Check performance degradation
        if current_metrics["latency_p95"] > self.baseline["latency_p95"] * 1.5:
            issues.append(f"Latency increased: {current_metrics['latency_p95']} vs {self.baseline['latency_p95']}")

        # Check cost increase
        if current_metrics["cost_per_request"] > self.baseline["cost_per_request"] * 1.2:
            issues.append(f"Cost increased: {current_metrics['cost_per_request']} vs {self.baseline['cost_per_request']}")

        return issues

# Usage: Run after each deployment
tester = RegressionTester(baseline_metrics=production_baseline)
issues = tester.check_regression(new_version_metrics)
if issues:
    alert("Regression detected!", issues)
```

### 2. Anomaly Detection with Statistical Methods

Identify unusual patterns:

```python
import numpy as np
from collections import deque

class AnomalyDetector:
    def __init__(self, window_size: int = 100, threshold_std: float = 3.0):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std

    def is_anomaly(self, value: float) -> tuple[bool, str]:
        if len(self.window) < 30:  # Need baseline
            self.window.append(value)
            return False, ""

        mean = np.mean(self.window)
        std = np.std(self.window)

        z_score = abs(value - mean) / std if std > 0 else 0

        is_anomalous = z_score > self.threshold_std

        self.window.append(value)

        if is_anomalous:
            return True, f"Value {value:.2f} is {z_score:.1f} std devs from mean {mean:.2f}"

        return False, ""

# Usage
latency_detector = AnomalyDetector()

for request_latency in stream_of_requests:
    is_anomaly, reason = latency_detector.is_anomaly(request_latency)
    if is_anomaly:
        alert(f"Latency anomaly: {reason}")
```

### 3. A/B Testing Framework

Compare variants systematically:

```python
import random
from collections import defaultdict

class ABTestFramework:
    def __init__(self):
        self.variants = {}
        self.metrics = defaultdict(list)

    def register_variant(self, name: str, agent_config: dict):
        self.variants[name] = agent_config

    def select_variant(self, user_id: str) -> str:
        # Consistent assignment based on user_id
        random.seed(hash(user_id))
        return random.choice(list(self.variants.keys()))

    def record_metrics(self, variant: str, metrics: dict):
        self.metrics[variant].append(metrics)

    def get_results(self) -> dict:
        results = {}
        for variant, metric_list in self.metrics.items():
            results[variant] = {
                "count": len(metric_list),
                "avg_latency": np.mean([m["latency"] for m in metric_list]),
                "avg_quality": np.mean([m["quality"] for m in metric_list]),
                "avg_cost": np.mean([m["cost"] for m in metric_list])
            }
        return results

# Usage
ab_test = ABTestFramework()
ab_test.register_variant("control", {"model": "gpt-4", "temperature": 0})
ab_test.register_variant("experimental", {"model": "gpt-4o-mini", "temperature": 0})

variant = ab_test.select_variant(user_id)
result = run_agent_with_config(ab_test.variants[variant])
ab_test.record_metrics(variant, result.metrics)
```

### 4. Custom Evaluation Rubrics

Domain-specific quality assessment:

```python
class CustomEvaluator:
    """Evaluate customer support responses."""

    def evaluate(self, query: str, response: str) -> dict:
        scores = {}

        # Empathy check
        empathy_words = ["sorry", "understand", "apologize", "appreciate"]
        scores["empathy"] = any(word in response.lower() for word in empathy_words)

        # Actionability check
        action_indicators = ["can", "will", "let me", "here's how"]
        scores["actionable"] = any(phrase in response.lower() for phrase in action_indicators)

        # Brevity check (ideal: 50-200 words)
        word_count = len(response.split())
        scores["brevity"] = 1.0 if 50 <= word_count <= 200 else 0.5 if word_count < 300 else 0.0

        # Overall score
        scores["overall"] = (
            (1.0 if scores["empathy"] else 0.0) +
            (1.0 if scores["actionable"] else 0.0) +
            scores["brevity"]
        ) / 3.0

        return scores
```

### 5. Distributed Tracing for Multi-Agent Systems

Track requests across agent boundaries:

```python
import uuid
from contextvars import ContextVar

# Thread-safe trace context
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
span_stack_var: ContextVar[list] = ContextVar("span_stack", default=[])

class Tracer:
    @staticmethod
    def start_trace(request_id: str = None):
        trace_id = request_id or str(uuid.uuid4())
        trace_id_var.set(trace_id)
        span_stack_var.set([])
        return trace_id

    @staticmethod
    def start_span(name: str):
        trace_id = trace_id_var.get()
        span = {
            "trace_id": trace_id,
            "span_id": str(uuid.uuid4()),
            "name": name,
            "start_time": time.time()
        }
        span_stack_var.get().append(span)
        return span

    @staticmethod
    def end_span():
        span = span_stack_var.get().pop()
        span["duration"] = time.time() - span["start_time"]
        # Log span
        logger.info(f"Span: {span}")
        return span

# Usage
trace_id = Tracer.start_trace()

Tracer.start_span("orchestrator")
# ... orchestrator work ...
Tracer.end_span()

Tracer.start_span("search_agent")
# ... search agent work ...
Tracer.end_span()
```

## Comparison with Related Patterns

| Pattern | Focus | Metrics | Evaluation | When to Use |
|---------|-------|---------|------------|-------------|
| **Evaluation & Monitoring** | Observability, quality, performance | Comprehensive (quant + qual) | Automated + human | Production systems |
| **Error Recovery** | Handling failures | Error rates, retry counts | Success after recovery | Unreliable environments |
| **Guardrails** | Safety, compliance | Violation rates, filter accuracy | Rule compliance | Regulated domains |
| **Resource Optimization** | Cost, efficiency | Token usage, $ per request | Cost/quality ratio | Budget-constrained |

**Key Differences:**

- **Error Recovery**: Reactive (fix failures), Monitoring is proactive (detect trends)
- **Guardrails**: Preventive (block bad outputs), Monitoring is observational (track what happens)
- **Resource Optimization**: Focused on efficiency, Monitoring tracks all aspects (quality, speed, cost)

**Complementary Use**: These patterns work together:
- Monitoring identifies that errors are increasing → Error Recovery pattern mitigates them
- Monitoring detects safety violations → Guardrails pattern prevents them
- Monitoring reveals high costs → Resource Optimization pattern reduces them

## Common Pitfalls

### 1. Monitoring Without Acting

**Problem**: Collecting metrics but not using them to drive improvements

**Example**: Dashboard shows latency increasing for 2 weeks, no action taken

**Solution**:
- Set up alerts for actionable metrics
- Schedule regular metric reviews
- Tie metrics to team OKRs/goals
- Create runbooks for common issues

### 2. Vanity Metrics

**Problem**: Tracking metrics that look good but don't indicate real success

**Example**: "99% success rate" but users are unhappy because answers are technically correct but unhelpful

**Solution**:
- Focus on metrics that correlate with user outcomes
- Combine quantitative and qualitative measures
- Include user satisfaction metrics
- Validate metrics against actual user behavior

### 3. Over-Engineering Monitoring

**Problem**: Complex monitoring setup that's hard to maintain and provides little value

**Example**: 100+ metrics tracked, 20 dashboards, but core issues still missed

**Solution**:
- Start simple (latency, cost, success rate, quality)
- Add metrics when specific needs arise
- Regularly prune unused metrics
- Prioritize actionable over interesting metrics

### 4. Ignoring Statistical Significance

**Problem**: Reacting to noise in metrics rather than meaningful changes

**Example**: Quality score fluctuates from 8.2 to 8.3, triggering alert

**Solution**:
- Use confidence intervals for metric changes
- Require minimum sample size before alerting
- Track trends over time, not point values
- Use statistical tests for A/B comparisons

### 5. Privacy Violations in Logging

**Problem**: Logging sensitive user data without proper safeguards

**Example**: Logs contain user emails, credit card numbers, health info

**Solution**:
- Redact PII automatically
- Use privacy-safe identifiers
- Implement data retention policies
- Regular privacy audits of logs
- Encrypt logs at rest and in transit

### 6. Alert Fatigue

**Problem**: So many alerts that important ones are ignored

**Example**: 50 alerts per day, most are false positives

**Solution**:
- Tune alert thresholds based on false positive rate
- Use alert severity levels
- Aggregate related alerts
- Implement snooze/acknowledgment system
- Regular alert effectiveness review

## Conclusion

The Evaluation and Monitoring pattern is essential for building reliable, high-quality AI agents in production. It provides the observability, measurement, and continuous improvement capabilities needed to maintain and optimize AI systems over time.

**Use Evaluation & Monitoring when:**
- Deploying agents to production with real users
- Quality and reliability are critical to success
- Costs need to be tracked and optimized
- You need to detect and respond to issues quickly
- Continuous improvement is a goal
- Compliance or auditing requirements exist

**Implementation checklist:**
- ✅ Define core metrics (latency, cost, quality, success rate)
- ✅ Implement structured logging with proper context
- ✅ Set up automated quality evaluation (LLM-as-judge or heuristics)
- ✅ Create dashboards for real-time visibility
- ✅ Configure alerts for critical thresholds
- ✅ Implement anomaly detection for trend changes
- ✅ Establish data retention and privacy policies
- ✅ Set up A/B testing framework for comparisons
- ✅ Create runbooks for common issues
- ✅ Schedule regular metric reviews
- ✅ Track distributions, not just averages
- ✅ Separate monitoring from business logic

**Key Takeaways:**
- 📊 Monitor both quantitative (latency, tokens) and qualitative (relevance, coherence) metrics
- 🔍 Visibility enables debugging, optimization, and trust
- 📈 Track trends over time to detect degradation early
- 💰 Cost monitoring is essential for budget control
- 🤖 LLM-as-judge enables automated quality evaluation
- 🚨 Alerts should be actionable and properly tuned
- 🔐 Privacy and security must be built into monitoring
- 📊 Dashboards make metrics accessible to stakeholders
- 🧪 A/B testing enables data-driven improvements
- ⚡ Trade-off: Monitoring overhead vs. observability benefits

---

*Evaluation and Monitoring transforms AI agents from opaque systems into observable, measurable, and continuously improving applications—enabling confident deployment and systematic optimization in production environments.*
