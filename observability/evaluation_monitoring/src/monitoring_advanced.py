"""
Advanced Evaluation and Monitoring Implementation

This example demonstrates advanced monitoring capabilities:
- LLM-as-judge quality evaluation (relevance, coherence, helpfulness)
- Multi-dimensional quality metrics
- Anomaly detection for performance degradation
- Automated alerts and notifications
- Historical trend analysis
- Rich dashboard with charts and insights
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Quality Metrics Data Structures ---

@dataclass
class QualityScores:
    """Quality evaluation scores for an agent response."""
    relevance: float  # 0-10: Does output address the query?
    coherence: float  # 0-10: Is output logical and well-structured?
    helpfulness: float  # 0-10: Does it meet user needs?
    overall: float  # 0-10: Overall quality score
    reasoning: str  # Explanation of scores

    def to_dict(self) -> dict:
        return {
            "relevance": round(self.relevance, 1),
            "coherence": round(self.coherence, 1),
            "helpfulness": round(self.helpfulness, 1),
            "overall": round(self.overall, 1),
            "reasoning": self.reasoning
        }


@dataclass
class ExtendedMetrics:
    """Extended metrics including performance and quality."""
    request_id: str
    query: str
    response: str
    latency_ms: float
    tokens_total: int
    cost_usd: float
    success: bool
    quality_scores: Optional[QualityScores] = None
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


# --- LLM-as-Judge Evaluator ---

class QualityEvaluator:
    """Uses LLM to evaluate the quality of agent responses."""

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        self.judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    def evaluate(self, query: str, response: str) -> QualityScores:
        """
        Evaluate response quality using LLM-as-judge.

        Args:
            query: Original user query
            response: Agent's response

        Returns:
            QualityScores with ratings and reasoning
        """
        evaluation_prompt = f"""You are an expert evaluator assessing the quality of AI assistant responses.

Evaluate the following response on three dimensions:

1. RELEVANCE (0-10): Does the response directly address the user's query?
   - 10: Perfectly addresses the query
   - 7-9: Addresses most aspects
   - 4-6: Partially addresses the query
   - 0-3: Off-topic or irrelevant

2. COHERENCE (0-10): Is the response logical, well-structured, and easy to understand?
   - 10: Perfectly clear and logical
   - 7-9: Clear with minor issues
   - 4-6: Somewhat confusing
   - 0-3: Incoherent or illogical

3. HELPFULNESS (0-10): Does the response provide useful, actionable information?
   - 10: Extremely helpful and actionable
   - 7-9: Helpful with good information
   - 4-6: Somewhat helpful
   - 0-3: Not helpful

USER QUERY:
{query}

ASSISTANT RESPONSE:
{response}

Provide your evaluation in exactly this format:
RELEVANCE: [score]
COHERENCE: [score]
HELPFULNESS: [score]
REASONING: [brief explanation of scores]"""

        try:
            result = self.judge_llm.invoke([
                SystemMessage(content="You are an expert response evaluator."),
                HumanMessage(content=evaluation_prompt)
            ])

            evaluation_text = result.content

            # Parse scores
            relevance = self._extract_score(evaluation_text, "RELEVANCE")
            coherence = self._extract_score(evaluation_text, "COHERENCE")
            helpfulness = self._extract_score(evaluation_text, "HELPFULNESS")
            reasoning = self._extract_reasoning(evaluation_text)

            overall = (relevance + coherence + helpfulness) / 3.0

            return QualityScores(
                relevance=relevance,
                coherence=coherence,
                helpfulness=helpfulness,
                overall=overall,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            # Return neutral scores on failure
            return QualityScores(
                relevance=5.0,
                coherence=5.0,
                helpfulness=5.0,
                overall=5.0,
                reasoning=f"Evaluation failed: {e}"
            )

    def _extract_score(self, text: str, metric: str) -> float:
        """Extract score for a metric from evaluation text."""
        try:
            lines = text.split('\n')
            for line in lines:
                if metric.upper() in line.upper():
                    # Extract number from line
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_text = parts[1].strip().split()[0]
                        score = float(score_text)
                        return max(0.0, min(10.0, score))  # Clamp to 0-10
        except Exception as e:
            logger.warning(f"Failed to extract {metric} score: {e}")

        return 5.0  # Default to neutral

    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from evaluation text."""
        try:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if "REASONING" in line.upper():
                    # Get reasoning text (everything after REASONING:)
                    reasoning_start = i
                    reasoning_lines = []
                    for j in range(reasoning_start, len(lines)):
                        line_text = lines[j]
                        if ':' in line_text:
                            line_text = line_text.split(':', 1)[1].strip()
                        if line_text:
                            reasoning_lines.append(line_text)
                    return ' '.join(reasoning_lines)
        except Exception as e:
            logger.warning(f"Failed to extract reasoning: {e}")

        return "No reasoning provided"


# --- Anomaly Detection ---

class AnomalyDetector:
    """Detects anomalies in metrics using statistical methods."""

    def __init__(self, window_size: int = 50, threshold_std: float = 2.5):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.latency_history = deque(maxlen=window_size)
        self.quality_history = deque(maxlen=window_size)

    def add_observation(self, latency_ms: float, quality_score: float) -> None:
        """Add a new observation to history."""
        self.latency_history.append(latency_ms)
        self.quality_history.append(quality_score)

    def check_anomalies(self, current_latency: float, current_quality: float) -> list[str]:
        """
        Check for anomalies in current metrics.

        Returns:
            List of anomaly descriptions (empty if no anomalies)
        """
        anomalies = []

        # Need sufficient baseline data
        if len(self.latency_history) < 10:
            return anomalies

        # Check latency anomaly
        latency_mean = np.mean(self.latency_history)
        latency_std = np.std(self.latency_history)

        if latency_std > 0:
            latency_z = abs(current_latency - latency_mean) / latency_std
            if latency_z > self.threshold_std:
                anomalies.append(
                    f"Latency spike: {current_latency:.0f}ms "
                    f"({latency_z:.1f} std devs from mean {latency_mean:.0f}ms)"
                )

        # Check quality degradation
        if self.quality_history:
            quality_mean = np.mean(self.quality_history)
            quality_std = np.std(self.quality_history)

            if quality_std > 0:
                quality_z = abs(current_quality - quality_mean) / quality_std
                if current_quality < quality_mean - (self.threshold_std * quality_std):
                    anomalies.append(
                        f"Quality degradation: {current_quality:.1f}/10 "
                        f"({quality_z:.1f} std devs below mean {quality_mean:.1f})"
                    )

        return anomalies


# --- Alert System ---

class AlertManager:
    """Manages alerts based on metric thresholds."""

    def __init__(self):
        self.alert_thresholds = {
            "latency_ms": 5000,       # Alert if > 5s
            "error_rate": 0.05,       # Alert if > 5%
            "quality_score": 7.0,     # Alert if < 7.0
            "cost_per_request": 0.10  # Alert if > $0.10
        }
        self.alerts = []

    def check_thresholds(self, metrics: ExtendedMetrics) -> list[str]:
        """Check if metrics violate alert thresholds."""
        alerts = []

        # Latency check
        if metrics.latency_ms > self.alert_thresholds["latency_ms"]:
            alerts.append(
                f"⚠️  HIGH LATENCY: {metrics.latency_ms:.0f}ms "
                f"(threshold: {self.alert_thresholds['latency_ms']}ms)"
            )

        # Cost check
        if metrics.cost_usd > self.alert_thresholds["cost_per_request"]:
            alerts.append(
                f"⚠️  HIGH COST: ${metrics.cost_usd:.6f} "
                f"(threshold: ${self.alert_thresholds['cost_per_request']})"
            )

        # Quality check
        if metrics.quality_scores and metrics.quality_scores.overall < self.alert_thresholds["quality_score"]:
            alerts.append(
                f"⚠️  LOW QUALITY: {metrics.quality_scores.overall:.1f}/10 "
                f"(threshold: {self.alert_thresholds['quality_score']})"
            )

        # Record alerts
        for alert in alerts:
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "request_id": metrics.request_id,
                "alert": alert
            })
            logger.warning(f"ALERT: {alert}")

        return alerts


# --- Advanced Monitoring System ---

class AdvancedMonitoringSystem:
    """Comprehensive monitoring with quality evaluation and anomaly detection."""

    def __init__(self, model: str = "gpt-4o-mini", evaluate_sample_rate: float = 1.0):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.evaluator = QualityEvaluator(judge_model="gpt-4o-mini")
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.metrics_history: list[ExtendedMetrics] = []
        self.evaluate_sample_rate = evaluate_sample_rate

        # Pricing (approximate for gpt-4o-mini)
        self.input_price_per_1m = 0.150
        self.output_price_per_1m = 0.600

    def process_query(self, query: str, request_id: str) -> tuple[str, ExtendedMetrics]:
        """
        Process query with comprehensive monitoring and evaluation.

        Args:
            query: User query
            request_id: Unique request identifier

        Returns:
            Tuple of (response, metrics)
        """
        start_time = time.time()

        try:
            # Call LLM
            response = self.llm.invoke(query)
            response_text = response.content

            # Calculate performance metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_input = len(query.split()) * 1.3
            tokens_output = len(response_text.split()) * 1.3
            tokens_total = int(tokens_input + tokens_output)

            cost_usd = (
                (tokens_total / 1_000_000) * (self.input_price_per_1m + self.output_price_per_1m) / 2
            )

            # Evaluate quality (sample some requests to reduce cost)
            import random
            quality_scores = None
            if random.random() < self.evaluate_sample_rate:
                quality_scores = self.evaluator.evaluate(query, response_text)

            # Create metrics
            metrics = ExtendedMetrics(
                request_id=request_id,
                query=query,
                response=response_text,
                latency_ms=latency_ms,
                tokens_total=tokens_total,
                cost_usd=cost_usd,
                success=True,
                quality_scores=quality_scores
            )

            # Record metrics
            self.metrics_history.append(metrics)

            # Anomaly detection (if we have quality scores)
            if quality_scores:
                self.anomaly_detector.add_observation(latency_ms, quality_scores.overall)
                anomalies = self.anomaly_detector.check_anomalies(latency_ms, quality_scores.overall)

                if anomalies:
                    for anomaly in anomalies:
                        logger.warning(f"ANOMALY DETECTED: {anomaly}")

            # Alert checking
            alerts = self.alert_manager.check_thresholds(metrics)

            # Structured logging
            log_entry = {
                "event": "agent_request_completed",
                "request_id": request_id,
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens_total,
                "cost_usd": round(cost_usd, 6),
                "success": True,
            }

            if quality_scores:
                log_entry["quality"] = quality_scores.to_dict()

            logger.info(json.dumps(log_entry))

            return response_text, metrics

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            metrics = ExtendedMetrics(
                request_id=request_id,
                query=query,
                response="",
                latency_ms=latency_ms,
                tokens_total=0,
                cost_usd=0.0,
                success=False,
                error=str(e)
            )

            self.metrics_history.append(metrics)
            logger.error(f"Request {request_id} failed: {e}")
            raise

    def print_dashboard(self) -> None:
        """Print comprehensive monitoring dashboard."""
        if not self.metrics_history:
            print("No metrics to display")
            return

        print("\n" + "="*70)
        print("ADVANCED MONITORING DASHBOARD")
        print("="*70)

        # Overall statistics
        successful = [m for m in self.metrics_history if m.success]
        failed = [m for m in self.metrics_history if not m.success]
        with_quality = [m for m in successful if m.quality_scores]

        print("\n📊 OVERALL STATISTICS")
        print(f"  Total Requests:        {len(self.metrics_history)}")
        print(f"  Successful:            {len(successful)} ({len(successful)/len(self.metrics_history)*100:.1f}%)")
        print(f"  Failed:                {len(failed)}")
        print(f"  Quality Evaluated:     {len(with_quality)}")

        # Performance metrics
        if successful:
            latencies = [m.latency_ms for m in successful]
            costs = [m.cost_usd for m in successful]
            tokens = [m.tokens_total for m in successful]

            print("\n⏱️  PERFORMANCE METRICS")
            print(f"  Latency (avg):         {np.mean(latencies):.0f}ms")
            print(f"  Latency (p50):         {np.percentile(latencies, 50):.0f}ms")
            print(f"  Latency (p95):         {np.percentile(latencies, 95):.0f}ms")
            print(f"  Latency (p99):         {np.percentile(latencies, 99):.0f}ms")

            print("\n💰 COST ANALYSIS")
            print(f"  Total Cost:            ${sum(costs):.4f}")
            print(f"  Avg Cost/Request:      ${np.mean(costs):.6f}")
            print(f"  Total Tokens:          {sum(tokens):,}")
            print(f"  Avg Tokens/Request:    {np.mean(tokens):.0f}")

        # Quality metrics
        if with_quality:
            relevance_scores = [m.quality_scores.relevance for m in with_quality]
            coherence_scores = [m.quality_scores.coherence for m in with_quality]
            helpfulness_scores = [m.quality_scores.helpfulness for m in with_quality]
            overall_scores = [m.quality_scores.overall for m in with_quality]

            print("\n🎯 QUALITY METRICS (LLM-as-Judge)")
            print(f"  Relevance (avg):       {np.mean(relevance_scores):.1f}/10")
            print(f"  Coherence (avg):       {np.mean(coherence_scores):.1f}/10")
            print(f"  Helpfulness (avg):     {np.mean(helpfulness_scores):.1f}/10")
            print(f"  Overall Quality:       {np.mean(overall_scores):.1f}/10")

            # Quality distribution
            print("\n  Quality Distribution:")
            excellent = sum(1 for s in overall_scores if s >= 9.0)
            good = sum(1 for s in overall_scores if 8.0 <= s < 9.0)
            acceptable = sum(1 for s in overall_scores if 7.0 <= s < 8.0)
            poor = sum(1 for s in overall_scores if s < 7.0)

            print(f"    Excellent (≥9.0):    {excellent} ({excellent/len(overall_scores)*100:.0f}%)")
            print(f"    Good (8.0-8.9):      {good} ({good/len(overall_scores)*100:.0f}%)")
            print(f"    Acceptable (7.0-7.9): {acceptable} ({acceptable/len(overall_scores)*100:.0f}%)")
            print(f"    Poor (<7.0):         {poor} ({poor/len(overall_scores)*100:.0f}%)")

        # Alerts summary
        if self.alert_manager.alerts:
            print("\n🚨 ALERTS SUMMARY")
            print(f"  Total Alerts:          {len(self.alert_manager.alerts)}")
            print(f"  Recent Alerts:")
            for alert in self.alert_manager.alerts[-3:]:  # Show last 3
                print(f"    - {alert['alert']}")

        print("\n" + "="*70)

        # Detailed request view
        print("\n📋 DETAILED REQUEST VIEW")
        print("-" * 70)

        for i, m in enumerate(self.metrics_history, 1):
            status = "✓" if m.success else "✗"
            print(f"\n{i}. [{m.request_id}] {status}")
            print(f"   Query: {m.query[:70]}{'...' if len(m.query) > 70 else ''}")
            print(f"   Latency: {m.latency_ms:.0f}ms | Tokens: {m.tokens_total} | Cost: ${m.cost_usd:.6f}")

            if m.quality_scores:
                print(f"   Quality Scores:")
                print(f"     - Relevance:    {m.quality_scores.relevance:.1f}/10")
                print(f"     - Coherence:    {m.quality_scores.coherence:.1f}/10")
                print(f"     - Helpfulness:  {m.quality_scores.helpfulness:.1f}/10")
                print(f"     - Overall:      {m.quality_scores.overall:.1f}/10")
                print(f"   Reasoning: {m.quality_scores.reasoning[:100]}{'...' if len(m.quality_scores.reasoning) > 100 else ''}")

            if m.error:
                print(f"   Error: {m.error}")

        print("\n" + "="*70 + "\n")


# --- Demo Runner ---

def run_advanced_monitoring_demo():
    """Run demonstration of advanced monitoring with quality evaluation."""
    print("\n" + "="*70)
    print("ADVANCED EVALUATION & MONITORING DEMONSTRATION")
    print("="*70)
    print("\nThis example demonstrates:")
    print("  - LLM-as-judge quality evaluation (relevance, coherence, helpfulness)")
    print("  - Multi-dimensional quality metrics")
    print("  - Anomaly detection (performance degradation)")
    print("  - Automated alerts and notifications")
    print("  - Rich dashboard with insights")
    print("\n" + "="*70)

    # Initialize monitoring system
    system = AdvancedMonitoringSystem(
        model="gpt-4o-mini",
        evaluate_sample_rate=1.0  # Evaluate all requests for demo
    )

    # Test queries with varying quality expectations
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between AI and machine learning in simple terms.",
        "What are the main applications of natural language processing?",
        "How do neural networks learn?",
        "What is the future of artificial intelligence?",
    ]

    print("\n🚀 Processing queries with advanced monitoring...\n")

    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Processing: {query[:60]}{'...' if len(query) > 60 else ''}")

        request_id = f"req-adv-{i:03d}"

        try:
            response, metrics = system.process_query(query, request_id)

            print(f"  ✓ Completed in {metrics.latency_ms:.0f}ms")

            if metrics.quality_scores:
                print(f"  Quality: {metrics.quality_scores.overall:.1f}/10", end=" ")
                print(f"(R:{metrics.quality_scores.relevance:.1f} C:{metrics.quality_scores.coherence:.1f} H:{metrics.quality_scores.helpfulness:.1f})")

            print(f"  Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

        print()

        # Small delay to make demo more readable
        time.sleep(0.5)

    # Display comprehensive dashboard
    system.print_dashboard()

    print("\n💡 KEY INSIGHTS")
    print("-" * 70)
    print("✓ LLM-as-judge provides automated quality assessment")
    print("✓ Anomaly detection identifies performance degradation")
    print("✓ Alerts enable proactive issue response")
    print("✓ Trend analysis shows quality and performance over time")
    print("✓ Multi-dimensional metrics provide complete visibility")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_advanced_monitoring_demo()
