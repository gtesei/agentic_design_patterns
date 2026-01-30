"""
Basic Monitoring Implementation

This example demonstrates fundamental monitoring capabilities:
- Performance metrics collection (latency, tokens, cost)
- Structured logging with context
- Simple dashboard/report generation
- Metric aggregation and statistics
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Metrics Data Structures ---

@dataclass
class RequestMetrics:
    """Metrics for a single agent request."""
    request_id: str
    query: str
    response: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float
    success: bool
    error: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "request_id": self.request_id,
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "response_length": len(self.response),
            "latency_ms": round(self.latency_ms, 2),
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_total
            },
            "cost_usd": round(self.cost_usd, 6),
            "success": self.success,
            "error": self.error,
            "tool_calls": len(self.tool_calls),
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple requests."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    avg_tokens_per_request: float
    avg_cost_per_request: float
    success_rate: float
    total_tool_calls: int
    avg_tools_per_request: float


# --- Metrics Collector ---

class MetricsCollector:
    """Collects and aggregates agent performance metrics."""

    def __init__(self):
        self.metrics: list[RequestMetrics] = []

    def record_request(self, metrics: RequestMetrics) -> None:
        """Record metrics for a single request."""
        self.metrics.append(metrics)

        # Structured logging
        logger.info(json.dumps({
            "event": "agent_request_completed",
            **metrics.to_dict()
        }))

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Calculate aggregated statistics across all requests."""
        if not self.metrics:
            return AggregatedMetrics(
                total_requests=0, successful_requests=0, failed_requests=0,
                avg_latency_ms=0, p50_latency_ms=0, p95_latency_ms=0, p99_latency_ms=0,
                total_tokens=0, total_cost_usd=0, avg_tokens_per_request=0,
                avg_cost_per_request=0, success_rate=0, total_tool_calls=0,
                avg_tools_per_request=0
            )

        latencies = [m.latency_ms for m in self.metrics]
        successful = [m for m in self.metrics if m.success]
        total_tool_calls = sum(len(m.tool_calls) for m in self.metrics)

        return AggregatedMetrics(
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=len(self.metrics) - len(successful),
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            total_tokens=sum(m.tokens_total for m in self.metrics),
            total_cost_usd=sum(m.cost_usd for m in self.metrics),
            avg_tokens_per_request=float(np.mean([m.tokens_total for m in self.metrics])),
            avg_cost_per_request=float(np.mean([m.cost_usd for m in self.metrics])),
            success_rate=len(successful) / len(self.metrics),
            total_tool_calls=total_tool_calls,
            avg_tools_per_request=total_tool_calls / len(self.metrics)
        )

    def print_report(self) -> None:
        """Print a formatted metrics report."""
        agg = self.get_aggregated_metrics()

        print("\n" + "="*60)
        print("AGENT PERFORMANCE METRICS REPORT")
        print("="*60)

        print("\n📊 REQUEST SUMMARY")
        print(f"  Total Requests:      {agg.total_requests}")
        print(f"  Successful:          {agg.successful_requests} ({agg.success_rate*100:.1f}%)")
        print(f"  Failed:              {agg.failed_requests}")

        print("\n⏱️  LATENCY METRICS")
        print(f"  Average:             {agg.avg_latency_ms:.0f}ms")
        print(f"  Median (P50):        {agg.p50_latency_ms:.0f}ms")
        print(f"  P95:                 {agg.p95_latency_ms:.0f}ms")
        print(f"  P99:                 {agg.p99_latency_ms:.0f}ms")

        print("\n🎯 TOKEN USAGE")
        print(f"  Total Tokens:        {agg.total_tokens:,}")
        print(f"  Avg per Request:     {agg.avg_tokens_per_request:.0f}")

        print("\n💰 COST ANALYSIS")
        print(f"  Total Cost:          ${agg.total_cost_usd:.4f}")
        print(f"  Avg per Request:     ${agg.avg_cost_per_request:.6f}")

        print("\n🔧 TOOL USAGE")
        print(f"  Total Tool Calls:    {agg.total_tool_calls}")
        print(f"  Avg per Request:     {agg.avg_tools_per_request:.1f}")

        print("\n" + "="*60)

        # Show individual request details
        print("\n📋 INDIVIDUAL REQUEST DETAILS")
        print("-" * 60)
        for i, m in enumerate(self.metrics, 1):
            status = "✓" if m.success else "✗"
            print(f"\n{i}. Request: {m.request_id}")
            print(f"   Query: {m.query[:80]}{'...' if len(m.query) > 80 else ''}")
            print(f"   Status: {status} | Latency: {m.latency_ms:.0f}ms | Tokens: {m.tokens_total} | Cost: ${m.cost_usd:.6f}")
            if m.tool_calls:
                print(f"   Tools: {', '.join(t['tool'] for t in m.tool_calls)}")
            if m.error:
                print(f"   Error: {m.error}")

        print("\n" + "="*60 + "\n")


# --- Monitored Agent Implementation ---

class MonitoredAgent:
    """Agent wrapper with built-in metrics collection."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.metrics_collector = MetricsCollector()
        self.tools = [search_knowledge_base, get_current_time, simple_calculator]

        # Pricing (per 1M tokens) - approximate for gpt-4o-mini
        self.input_price_per_1m = 0.150
        self.output_price_per_1m = 0.600

    def invoke(self, query: str, request_id: str) -> tuple[str, RequestMetrics]:
        """
        Process a query with comprehensive monitoring.

        Args:
            query: User query
            request_id: Unique identifier for this request

        Returns:
            Tuple of (response, metrics)
        """
        start_time = time.time()
        tool_calls = []

        try:
            # Create a simple prompt that may trigger tool use
            prompt = f"""You are a helpful assistant. Answer the following query.
If you need additional information, you can use these tools:
- search_knowledge_base(query): Search for factual information
- get_current_time(): Get the current time
- simple_calculator(expression): Perform calculations

Query: {query}

Provide a clear, concise answer."""

            # Simulate tool usage tracking
            response_text = ""
            tokens_input = len(prompt.split()) * 1.3  # Rough token estimate
            tokens_output = 0

            # Call LLM
            response = self.llm.invoke(prompt)
            response_text = response.content
            tokens_output = len(response_text.split()) * 1.3

            # Check if query might need tools (simple heuristic)
            if any(keyword in query.lower() for keyword in ["calculate", "compute", "math"]):
                tool_calls.append({"tool": "simple_calculator", "latency_ms": 50})
            if any(keyword in query.lower() for keyword in ["search", "find", "look up", "what is"]):
                tool_calls.append({"tool": "search_knowledge_base", "latency_ms": 120})
            if "time" in query.lower() or "date" in query.lower():
                tool_calls.append({"tool": "get_current_time", "latency_ms": 10})

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_input_int = int(tokens_input)
            tokens_output_int = int(tokens_output)
            tokens_total = tokens_input_int + tokens_output_int

            cost_usd = (
                (tokens_input_int / 1_000_000) * self.input_price_per_1m +
                (tokens_output_int / 1_000_000) * self.output_price_per_1m
            )

            metrics = RequestMetrics(
                request_id=request_id,
                query=query,
                response=response_text,
                latency_ms=latency_ms,
                tokens_input=tokens_input_int,
                tokens_output=tokens_output_int,
                tokens_total=tokens_total,
                cost_usd=cost_usd,
                success=True,
                tool_calls=tool_calls
            )

            self.metrics_collector.record_request(metrics)
            return response_text, metrics

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            metrics = RequestMetrics(
                request_id=request_id,
                query=query,
                response="",
                latency_ms=latency_ms,
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                cost_usd=0.0,
                success=False,
                error=str(e),
                tool_calls=tool_calls
            )

            self.metrics_collector.record_request(metrics)
            logger.error(f"Request {request_id} failed: {e}")
            raise


# --- Mock Tools ---

@tool
def search_knowledge_base(query: str) -> str:
    """Search for factual information in the knowledge base."""
    knowledge = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "monitoring": "Monitoring involves tracking metrics, logs, and traces to understand system behavior.",
        "ai agent": "An AI agent is an autonomous system that perceives its environment and takes actions to achieve goals.",
    }

    for key, value in knowledge.items():
        if key in query.lower():
            return value

    return "Information not found in knowledge base."


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def simple_calculator(expression: str) -> str:
    """Perform simple mathematical calculations."""
    try:
        # Safe evaluation for basic math
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"


# --- Demo Runner ---

def run_basic_monitoring_demo():
    """Run demonstration of basic monitoring capabilities."""
    print("\n" + "="*60)
    print("BASIC MONITORING DEMONSTRATION")
    print("="*60)
    print("\nThis example demonstrates:")
    print("  - Performance metrics collection (latency, tokens, cost)")
    print("  - Structured logging")
    print("  - Metric aggregation and statistics")
    print("  - Simple dashboard reporting")
    print("\n" + "="*60)

    # Initialize monitored agent
    agent = MonitoredAgent(model="gpt-4o-mini")

    # Test queries
    test_queries = [
        "What is Python?",
        "Calculate 15 * 23 + 100",
        "What is an AI agent?",
        "What is the current time?",
        "Explain monitoring in software systems",
    ]

    print("\n🚀 Processing queries with monitoring...\n")

    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Processing: {query[:60]}{'...' if len(query) > 60 else ''}")

        request_id = f"req-{i:03d}"

        try:
            response, metrics = agent.invoke(query, request_id)

            print(f"  ✓ Completed in {metrics.latency_ms:.0f}ms")
            print(f"    Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

        print()

    # Display aggregated metrics
    agent.metrics_collector.print_report()

    # Show example of structured log entry
    print("\n📝 EXAMPLE STRUCTURED LOG ENTRY")
    print("-" * 60)
    if agent.metrics_collector.metrics:
        example_log = json.dumps(
            {"event": "agent_request_completed", **agent.metrics_collector.metrics[0].to_dict()},
            indent=2
        )
        print(example_log)
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    run_basic_monitoring_demo()
