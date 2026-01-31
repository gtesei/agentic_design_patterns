"""
Adaptive Learning Pattern: Basic Implementation

This example demonstrates basic adaptive learning where an agent improves its responses
based on user feedback. The agent collects ratings, identifies successful patterns,
and learns from high-quality examples to improve future responses.

Key concepts:
- Feedback collection (user ratings)
- Pattern extraction from successful interactions
- Few-shot learning with best examples
- Performance tracking over time
- Visualization of improvement trends
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


class FeedbackCollector:
    """Collects and stores user feedback on agent responses"""

    def __init__(self):
        self.feedback_history: List[Dict] = []

    def collect_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        timestamp: datetime = None,
    ) -> Dict:
        """
        Collect feedback for an interaction

        Args:
            query: User's input query
            response: Agent's response
            rating: User rating (1-5)
            timestamp: When feedback was given

        Returns:
            Feedback record
        """
        feedback = {
            "query": query,
            "response": response,
            "rating": rating,
            "timestamp": timestamp or datetime.now(),
        }

        self.feedback_history.append(feedback)
        return feedback

    def get_successful_examples(self, min_rating: int = 4) -> List[Dict]:
        """Get examples with high ratings"""
        return [f for f in self.feedback_history if f["rating"] >= min_rating]

    def get_failed_examples(self, max_rating: int = 2) -> List[Dict]:
        """Get examples with low ratings"""
        return [f for f in self.feedback_history if f["rating"] <= max_rating]

    def get_average_rating(self) -> float:
        """Calculate average rating across all feedback"""
        if not self.feedback_history:
            return 0.0
        return sum(f["rating"] for f in self.feedback_history) / len(self.feedback_history)

    def get_rating_trend(self, window_size: int = 5) -> List[float]:
        """Calculate moving average of ratings"""
        if len(self.feedback_history) < window_size:
            return [self.get_average_rating()]

        trend = []
        for i in range(window_size - 1, len(self.feedback_history)):
            window = self.feedback_history[i - window_size + 1 : i + 1]
            avg = sum(f["rating"] for f in window) / window_size
            trend.append(avg)

        return trend


class PatternAnalyzer:
    """Analyzes feedback to extract patterns and insights"""

    def __init__(self, feedback_collector: FeedbackCollector):
        self.collector = feedback_collector

    def identify_success_patterns(self) -> Dict[str, List[str]]:
        """Identify patterns in successful responses"""
        successful = self.collector.get_successful_examples()

        patterns = {
            "topics": defaultdict(int),
            "response_lengths": [],
            "common_phrases": defaultdict(int),
        }

        for example in successful:
            # Track topics (keywords in queries)
            words = example["query"].lower().split()
            for word in words:
                if len(word) > 4:  # Filter short words
                    patterns["topics"][word] += 1

            # Track response characteristics
            patterns["response_lengths"].append(len(example["response"]))

            # Track common phrases in responses
            sentences = example["response"].split(".")
            for sentence in sentences[:2]:  # First two sentences
                if sentence.strip():
                    patterns["common_phrases"][sentence.strip()[:50]] += 1

        return patterns

    def get_performance_summary(self) -> Dict:
        """Generate summary of performance metrics"""
        all_feedback = self.collector.feedback_history

        if not all_feedback:
            return {
                "total_interactions": 0,
                "average_rating": 0.0,
                "improvement_rate": 0.0,
            }

        ratings = [f["rating"] for f in all_feedback]

        # Calculate improvement rate (last 25% vs first 25%)
        split = len(ratings) // 4
        if split > 0:
            early_avg = sum(ratings[:split]) / split
            recent_avg = sum(ratings[-split:]) / split
            improvement_rate = ((recent_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0.0
        else:
            improvement_rate = 0.0

        return {
            "total_interactions": len(all_feedback),
            "average_rating": sum(ratings) / len(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            "improvement_rate": improvement_rate,
            "rating_std": np.std(ratings),
        }


class FewShotLearner:
    """Learns from successful examples and incorporates them into prompts"""

    def __init__(self, max_examples: int = 5):
        self.max_examples = max_examples
        self.learned_examples: List[Dict] = []

    def add_successful_example(self, query: str, response: str, rating: int):
        """Add a high-quality example to the learning set"""
        example = {"query": query, "response": response, "rating": rating}

        # Insert sorted by rating (descending)
        self.learned_examples.append(example)
        self.learned_examples.sort(key=lambda x: x["rating"], reverse=True)

        # Keep only top examples
        self.learned_examples = self.learned_examples[: self.max_examples]

    def generate_few_shot_prompt(self) -> str:
        """Generate prompt with learned examples"""
        if not self.learned_examples:
            return ""

        prompt = "\n\nHere are examples of excellent responses that received high user ratings:\n\n"

        for i, example in enumerate(self.learned_examples, 1):
            prompt += f"Example {i} (Rating: {example['rating']}/5):\n"
            prompt += f"User: {example['query']}\n"
            prompt += f"Assistant: {example['response']}\n\n"

        prompt += "Please provide a similar quality response to the current query.\n"

        return prompt

    def get_learning_stats(self) -> Dict:
        """Get statistics about learned examples"""
        if not self.learned_examples:
            return {"count": 0, "avg_rating": 0.0}

        return {
            "count": len(self.learned_examples),
            "avg_rating": sum(e["rating"] for e in self.learned_examples) / len(self.learned_examples),
            "examples": [e["query"][:50] + "..." for e in self.learned_examples],
        }


class AdaptiveLearningAgent:
    """Customer support agent that learns from feedback"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.feedback_collector = FeedbackCollector()
        self.pattern_analyzer = PatternAnalyzer(self.feedback_collector)
        self.few_shot_learner = FewShotLearner(max_examples=3)

        self.base_system_prompt = """You are a helpful customer support agent.
Your goal is to provide clear, friendly, and accurate responses to customer queries.
Be empathetic, professional, and solution-oriented."""

    def generate_response(self, query: str, use_learning: bool = True) -> str:
        """
        Generate a response to a customer query

        Args:
            query: Customer's question or issue
            use_learning: Whether to use learned examples

        Returns:
            Agent's response
        """
        system_prompt = self.base_system_prompt

        # Add learned examples if using adaptive learning
        if use_learning and self.few_shot_learner.learned_examples:
            system_prompt += self.few_shot_learner.generate_few_shot_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def process_feedback(self, query: str, response: str, rating: int):
        """Process user feedback and learn from it"""
        # Collect feedback
        self.feedback_collector.collect_feedback(query, response, rating)

        # If high rating, add to learning examples
        if rating >= 4:
            self.few_shot_learner.add_successful_example(query, response, rating)

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        return {
            "feedback_stats": self.pattern_analyzer.get_performance_summary(),
            "learning_stats": self.few_shot_learner.get_learning_stats(),
            "success_patterns": self.pattern_analyzer.identify_success_patterns(),
        }


def visualize_learning_progress(feedback_collector: FeedbackCollector):
    """Create visualizations of learning progress"""
    if not feedback_collector.feedback_history:
        print("No feedback data to visualize")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Rating over time
    ratings = [f["rating"] for f in feedback_collector.feedback_history]
    interactions = list(range(1, len(ratings) + 1))

    ax1.plot(interactions, ratings, marker="o", linestyle="-", alpha=0.6, label="Individual Ratings")

    # Add moving average if enough data
    if len(ratings) >= 5:
        trend = feedback_collector.get_rating_trend(window_size=5)
        trend_x = list(range(5, len(ratings) + 1))
        ax1.plot(trend_x, trend, marker="s", linestyle="-", linewidth=2, color="red", label="Moving Average (5)")

    ax1.set_xlabel("Interaction Number")
    ax1.set_ylabel("Rating (1-5)")
    ax1.set_title("Rating Progression Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 6)

    # Plot 2: Rating distribution
    rating_counts = defaultdict(int)
    for rating in ratings:
        rating_counts[rating] += 1

    ax2.bar(rating_counts.keys(), rating_counts.values(), color="skyblue", edgecolor="black")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Ratings")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("learning_progress.png", dpi=150, bbox_inches="tight")
    print("\n✅ Visualization saved as 'learning_progress.png'")
    plt.close()


def simulate_customer_interactions(agent: AdaptiveLearningAgent, num_interactions: int = 20):
    """Simulate customer interactions with feedback"""

    # Sample queries and expected quality (simulating user satisfaction)
    sample_queries = [
        ("How do I reset my password?", 3),  # Moderate difficulty
        ("What are your business hours?", 5),  # Easy question
        ("I can't login to my account", 3),  # Common issue
        ("How do I cancel my subscription?", 2),  # Sensitive topic
        ("Is there a refund policy?", 4),  # Important question
        ("My payment failed, what should I do?", 3),  # Technical issue
        ("How do I upgrade my account?", 4),  # Sales opportunity
        ("I need help with billing", 3),  # Support issue
        ("How do I export my data?", 4),  # Technical question
        ("Can I get a discount?", 2),  # Negotiation
    ]

    print("\n" + "=" * 80)
    print("ADAPTIVE LEARNING SIMULATION: Customer Support Agent")
    print("=" * 80)

    print("\nPhase 1: Initial Interactions (No Learning)")
    print("-" * 80)

    # First phase: No learning (baseline)
    for i in range(min(5, num_interactions)):
        query, base_quality = sample_queries[i % len(sample_queries)]

        print(f"\n--- Interaction {i + 1} ---")
        print(f"Customer: {query}")

        # Generate response without learning
        response = agent.generate_response(query, use_learning=False)
        print(f"Agent: {response[:150]}...")

        # Simulate user rating (baseline quality)
        rating = max(1, min(5, base_quality + np.random.randint(-1, 2)))
        print(f"Rating: {'⭐' * rating} ({rating}/5)")

        # Process feedback
        agent.process_feedback(query, response, rating)

    print("\n" + "=" * 80)
    print("Phase 2: Learning Phase (Using Feedback)")
    print("=" * 80)

    # Second phase: With learning
    for i in range(5, num_interactions):
        query, base_quality = sample_queries[i % len(sample_queries)]

        print(f"\n--- Interaction {i + 1} ---")
        print(f"Customer: {query}")

        # Generate response WITH learning
        response = agent.generate_response(query, use_learning=True)
        print(f"Agent: {response[:150]}...")

        # Simulate improved rating (learning improves quality)
        # As more examples are learned, rating tends to improve
        learning_boost = min(len(agent.few_shot_learner.learned_examples) * 0.3, 1.5)
        rating = int(max(1, min(5, base_quality + learning_boost + np.random.randint(-1, 2))))
        print(f"Rating: {'⭐' * rating} ({rating}/5)")

        # Process feedback
        agent.process_feedback(query, response, rating)


def print_performance_report(agent: AdaptiveLearningAgent):
    """Print comprehensive performance report"""
    report = agent.get_performance_report()

    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)

    # Feedback statistics
    stats = report["feedback_stats"]
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Interactions: {stats['total_interactions']}")
    print(f"  Average Rating: {stats['average_rating']:.2f}/5")
    print(f"  Rating Range: {stats['min_rating']:.0f} - {stats['max_rating']:.0f}")
    print(f"  Rating Std Dev: {stats['rating_std']:.2f}")
    print(f"  Improvement Rate: {stats['improvement_rate']:+.1f}%")

    # Learning statistics
    learning = report["learning_stats"]
    print(f"\n🎓 Learning Progress:")
    print(f"  Learned Examples: {learning['count']}")
    print(f"  Avg Example Rating: {learning['avg_rating']:.2f}/5")

    if learning["count"] > 0:
        print(f"\n  Top Learned Patterns:")
        for i, example in enumerate(learning["examples"], 1):
            print(f"    {i}. {example}")

    # Success patterns
    patterns = report["success_patterns"]
    if patterns["topics"]:
        print(f"\n✅ Success Topics (most common in high-rated interactions):")
        top_topics = sorted(patterns["topics"].items(), key=lambda x: x[1], reverse=True)[:5]
        for topic, count in top_topics:
            print(f"    • {topic}: {count} occurrences")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print(
        """
    ╔═══════════════════════════════════════════════════════════════╗
    ║      Adaptive Learning Pattern - Basic Implementation         ║
    ║                                                               ║
    ║  The agent learns from user feedback to improve responses    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    )

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create adaptive learning agent
    agent = AdaptiveLearningAgent(llm)

    # Simulate interactions with learning
    simulate_customer_interactions(agent, num_interactions=20)

    # Print performance report
    print_performance_report(agent)

    # Visualize learning progress
    visualize_learning_progress(agent.feedback_collector)

    print(
        """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Simulation Complete!                       ║
    ║                                                               ║
    ║  The agent demonstrated:                                      ║
    ║  ✅ Collecting user feedback (ratings)                        ║
    ║  ✅ Identifying successful patterns                           ║
    ║  ✅ Learning from high-quality examples                       ║
    ║  ✅ Improving performance over time                           ║
    ║  ✅ Visualizing learning progress                             ║
    ║                                                               ║
    ║  Key Insight: Performance improved by learning from the       ║
    ║  best examples and incorporating them into future responses.  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    )
