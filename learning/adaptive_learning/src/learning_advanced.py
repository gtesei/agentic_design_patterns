"""
Adaptive Learning Pattern: Advanced Implementation

This example demonstrates advanced adaptive learning using a multi-armed bandit approach
to optimize strategy selection. The agent experiments with different response strategies,
tracks their performance, and learns which strategies work best through exploration
and exploitation.

Key concepts:
- Multi-armed bandit optimization (epsilon-greedy)
- Multiple competing strategies
- Exploration vs exploitation trade-off
- A/B testing framework
- Prompt evolution based on success patterns
- Real-time strategy adaptation
- Rich performance visualization
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


class MultiArmedBandit:
    """
    Multi-armed bandit for strategy optimization using epsilon-greedy approach

    Balances exploration (trying different strategies) with exploitation
    (using the best known strategy)
    """

    def __init__(self, strategies: List[str], epsilon: float = 0.2):
        """
        Initialize bandit

        Args:
            strategies: List of strategy names to choose from
            epsilon: Exploration rate (0-1). Higher = more exploration
        """
        self.strategies = strategies
        self.epsilon = epsilon

        # Track performance of each strategy
        self.counts = {s: 0 for s in strategies}
        self.rewards = {s: 0.0 for s in strategies}
        self.reward_history = {s: [] for s in strategies}

    def select_strategy(self) -> Tuple[str, bool]:
        """
        Select a strategy using epsilon-greedy approach

        Returns:
            (strategy_name, is_exploration)
        """
        if np.random.random() < self.epsilon:
            # Exploration: Random strategy
            strategy = np.random.choice(self.strategies)
            return strategy, True
        else:
            # Exploitation: Best known strategy
            avg_rewards = self.get_average_rewards()
            if all(r == 0 for r in avg_rewards.values()):
                # No data yet, random choice
                strategy = np.random.choice(self.strategies)
            else:
                strategy = max(avg_rewards, key=avg_rewards.get)
            return strategy, False

    def update(self, strategy: str, reward: float):
        """
        Update strategy performance based on outcome

        Args:
            strategy: Strategy that was used
            reward: Reward received (e.g., user rating normalized to 0-1)
        """
        self.counts[strategy] += 1
        self.rewards[strategy] += reward
        self.reward_history[strategy].append(reward)

    def get_average_rewards(self) -> Dict[str, float]:
        """Get average reward for each strategy"""
        return {s: self.rewards[s] / max(self.counts[s], 1) for s in self.strategies}

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        avg_rewards = self.get_average_rewards()

        total_trials = sum(self.counts.values())
        exploration_count = int(total_trials * self.epsilon) if total_trials > 0 else 0

        return {
            "total_trials": total_trials,
            "strategy_counts": self.counts.copy(),
            "average_rewards": avg_rewards,
            "best_strategy": max(avg_rewards, key=avg_rewards.get) if avg_rewards else None,
            "exploration_rate": self.epsilon,
            "exploration_count": exploration_count,
            "exploitation_count": total_trials - exploration_count,
        }


class ResponseStrategy:
    """Defines a specific response strategy with its own prompt engineering"""

    def __init__(self, name: str, system_prompt: str, description: str):
        self.name = name
        self.system_prompt = system_prompt
        self.description = description


class StrategyManager:
    """Manages multiple response strategies"""

    def __init__(self):
        self.strategies: Dict[str, ResponseStrategy] = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize different response strategies"""

        # Strategy 1: Concise and Direct
        self.add_strategy(
            ResponseStrategy(
                name="concise",
                system_prompt="""You are a customer support agent focused on efficiency and clarity.
Provide direct, concise answers. Get straight to the point without unnecessary elaboration.
Use bullet points when listing steps. Be friendly but brief.""",
                description="Short, direct responses",
            )
        )

        # Strategy 2: Empathetic and Detailed
        self.add_strategy(
            ResponseStrategy(
                name="empathetic",
                system_prompt="""You are a customer support agent who prioritizes empathy and understanding.
Acknowledge the customer's feelings and situation. Provide detailed, thorough explanations.
Show genuine care and patience. Take time to explain things clearly and completely.""",
                description="Warm, detailed responses",
            )
        )

        # Strategy 3: Technical and Precise
        self.add_strategy(
            ResponseStrategy(
                name="technical",
                system_prompt="""You are a customer support agent with technical expertise.
Provide precise, technically accurate information. Include relevant details and terminology.
Be professional and authoritative. Assume the customer wants comprehensive technical information.""",
                description="Technical, detailed responses",
            )
        )

        # Strategy 4: Friendly and Conversational
        self.add_strategy(
            ResponseStrategy(
                name="friendly",
                system_prompt="""You are a customer support agent with a friendly, conversational style.
Be warm and personable. Use casual language while remaining professional.
Add appropriate enthusiasm and positivity. Make the interaction feel human and engaging.""",
                description="Casual, friendly responses",
            )
        )

    def add_strategy(self, strategy: ResponseStrategy):
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> Optional[ResponseStrategy]:
        """Get a strategy by name"""
        return self.strategies.get(name)

    def get_strategy_names(self) -> List[str]:
        """Get list of all strategy names"""
        return list(self.strategies.keys())


class PerformanceTracker:
    """Tracks performance metrics and outcomes"""

    def __init__(self):
        self.interactions: List[Dict] = []

    def record_interaction(
        self,
        query: str,
        strategy: str,
        response: str,
        rating: int,
        is_exploration: bool,
        timestamp: datetime = None,
    ):
        """Record an interaction"""
        interaction = {
            "query": query,
            "strategy": strategy,
            "response": response,
            "rating": rating,
            "reward": self._rating_to_reward(rating),
            "is_exploration": is_exploration,
            "timestamp": timestamp or datetime.now(),
        }
        self.interactions.append(interaction)

    def _rating_to_reward(self, rating: int) -> float:
        """Convert rating (1-5) to reward (0-1)"""
        return (rating - 1) / 4.0

    def get_strategy_performance(self, strategy: str) -> Dict:
        """Get performance metrics for a specific strategy"""
        strategy_interactions = [i for i in self.interactions if i["strategy"] == strategy]

        if not strategy_interactions:
            return {"count": 0, "avg_rating": 0.0, "avg_reward": 0.0}

        ratings = [i["rating"] for i in strategy_interactions]
        rewards = [i["reward"] for i in strategy_interactions]

        return {
            "count": len(strategy_interactions),
            "avg_rating": np.mean(ratings),
            "avg_reward": np.mean(rewards),
            "rating_std": np.std(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
        }

    def get_overall_statistics(self) -> Dict:
        """Get overall performance statistics"""
        if not self.interactions:
            return {
                "total_interactions": 0,
                "overall_avg_rating": 0.0,
                "exploration_ratio": 0.0,
            }

        ratings = [i["rating"] for i in self.interactions]
        exploration_count = sum(1 for i in self.interactions if i["is_exploration"])

        return {
            "total_interactions": len(self.interactions),
            "overall_avg_rating": np.mean(ratings),
            "overall_avg_reward": np.mean([i["reward"] for i in self.interactions]),
            "exploration_count": exploration_count,
            "exploitation_count": len(self.interactions) - exploration_count,
            "exploration_ratio": exploration_count / len(self.interactions),
        }


class AdaptiveLearningAgentAdvanced:
    """Advanced agent that uses multi-armed bandit for strategy optimization"""

    def __init__(self, llm: ChatOpenAI, epsilon: float = 0.2):
        self.llm = llm
        self.strategy_manager = StrategyManager()
        self.bandit = MultiArmedBandit(self.strategy_manager.get_strategy_names(), epsilon=epsilon)
        self.tracker = PerformanceTracker()

    def generate_response(self, query: str) -> Tuple[str, str, bool]:
        """
        Generate response using selected strategy

        Returns:
            (response, strategy_used, is_exploration)
        """
        # Select strategy using bandit
        strategy_name, is_exploration = self.bandit.select_strategy()
        strategy = self.strategy_manager.get_strategy(strategy_name)

        # Generate response with selected strategy
        messages = [
            SystemMessage(content=strategy.system_prompt),
            HumanMessage(content=query),
        ]

        response = self.llm.invoke(messages)

        return response.content, strategy_name, is_exploration

    def process_feedback(self, query: str, strategy: str, response: str, rating: int, is_exploration: bool):
        """Process feedback and update bandit"""
        # Record interaction
        self.tracker.record_interaction(query, strategy, response, rating, is_exploration)

        # Update bandit with reward
        reward = (rating - 1) / 4.0  # Normalize to 0-1
        self.bandit.update(strategy, reward)

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        return {
            "bandit_stats": self.bandit.get_statistics(),
            "overall_stats": self.tracker.get_overall_statistics(),
            "strategy_performance": {
                name: self.tracker.get_strategy_performance(name)
                for name in self.strategy_manager.get_strategy_names()
            },
        }


def visualize_advanced_learning(agent: AdaptiveLearningAgentAdvanced):
    """Create comprehensive visualizations of learning progress"""

    if not agent.tracker.interactions:
        print("No data to visualize")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Strategy Performance Over Time
    ax1 = fig.add_subplot(gs[0, :])
    strategies = agent.strategy_manager.get_strategy_names()
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))

    for strategy, color in zip(strategies, colors):
        strategy_data = [i for i in agent.tracker.interactions if i["strategy"] == strategy]
        if strategy_data:
            interactions = list(range(1, len(strategy_data) + 1))
            ratings = [i["rating"] for i in strategy_data]

            # Calculate moving average
            if len(ratings) >= 3:
                window = 3
                moving_avg = [sum(ratings[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(ratings))]
                ax1.plot(interactions, moving_avg, marker="o", label=strategy.title(), color=color, linewidth=2)
            else:
                ax1.plot(interactions, ratings, marker="o", label=strategy.title(), color=color, linewidth=2)

    ax1.set_xlabel("Strategy Usage Count")
    ax1.set_ylabel("Average Rating")
    ax1.set_title("Strategy Performance Over Time (Moving Average)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 6)

    # Plot 2: Average Rating by Strategy
    ax2 = fig.add_subplot(gs[1, 0])
    report = agent.get_performance_report()
    strategy_perf = report["strategy_performance"]

    strategy_names = []
    avg_ratings = []
    for name, perf in strategy_perf.items():
        if perf["count"] > 0:
            strategy_names.append(name.title())
            avg_ratings.append(perf["avg_rating"])

    bars = ax2.bar(strategy_names, avg_ratings, color=colors[: len(strategy_names)], edgecolor="black")
    ax2.set_ylabel("Average Rating")
    ax2.set_title("Average Rating by Strategy")
    ax2.set_ylim(0, 5)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, rating in zip(bars, avg_ratings):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{rating:.2f}", ha="center", va="bottom", fontsize=10)

    # Plot 3: Strategy Usage Count
    ax3 = fig.add_subplot(gs[1, 1])
    counts = [strategy_perf[name]["count"] for name in strategy_perf.keys() if strategy_perf[name]["count"] > 0]
    labels = [name.title() for name in strategy_perf.keys() if strategy_perf[name]["count"] > 0]

    ax3.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors[: len(labels)], startangle=90)
    ax3.set_title("Strategy Usage Distribution")

    # Plot 4: Exploration vs Exploitation
    ax4 = fig.add_subplot(gs[2, 0])
    overall = report["overall_stats"]
    expl_expt = [overall["exploration_count"], overall["exploitation_count"]]
    labels = ["Exploration", "Exploitation"]

    ax4.pie(expl_expt, labels=labels, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"], startangle=90)
    ax4.set_title("Exploration vs Exploitation")

    # Plot 5: Cumulative Reward Over Time
    ax5 = fig.add_subplot(gs[2, 1])
    cumulative_rewards = []
    total_reward = 0
    for interaction in agent.tracker.interactions:
        total_reward += interaction["reward"]
        cumulative_rewards.append(total_reward)

    ax5.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, marker="o", color="green", linewidth=2)
    ax5.set_xlabel("Interaction Number")
    ax5.set_ylabel("Cumulative Reward")
    ax5.set_title("Cumulative Reward Over Time")
    ax5.grid(True, alpha=0.3)

    plt.savefig("advanced_learning_progress.png", dpi=150, bbox_inches="tight")
    print("\n✅ Visualization saved as 'advanced_learning_progress.png'")
    plt.close()


def simulate_advanced_interactions(agent: AdaptiveLearningAgentAdvanced, num_interactions: int = 30):
    """Simulate customer interactions with multiple strategies"""

    # Sample queries with varying difficulty
    queries = [
        "How do I reset my password?",
        "What are your business hours?",
        "I can't login to my account",
        "How do I cancel my subscription?",
        "Is there a refund policy?",
        "My payment failed, what should I do?",
        "How do I upgrade my account?",
        "I need help with billing",
        "How do I export my data?",
        "Can I get a discount?",
        "What's included in the premium plan?",
        "How do I contact support?",
    ]

    # Strategy preferences (simulating that some strategies work better)
    strategy_quality_bias = {
        "empathetic": 1.2,  # Tends to get higher ratings
        "friendly": 1.1,  # Also good
        "concise": 0.9,  # Slightly lower
        "technical": 0.8,  # Sometimes too complex
    }

    print("\n" + "=" * 80)
    print("ADVANCED ADAPTIVE LEARNING: Multi-Armed Bandit Strategy Optimization")
    print("=" * 80)
    print("\nThe agent will test different response strategies and learn which work best.")
    print(f"Exploration rate (epsilon): {agent.bandit.epsilon:.1%}")
    print("\nStrategies:")
    for name, strategy in agent.strategy_manager.strategies.items():
        print(f"  • {name.title()}: {strategy.description}")

    print("\n" + "-" * 80)
    print("Starting interactions...")
    print("-" * 80)

    for i in range(num_interactions):
        query = queries[i % len(queries)]

        # Generate response
        response, strategy, is_exploration = agent.generate_response(query)

        # Simulate rating based on strategy quality
        base_rating = 3.5
        strategy_bias = strategy_quality_bias[strategy]
        noise = np.random.normal(0, 0.5)
        rating = int(np.clip(base_rating * strategy_bias + noise, 1, 5))

        # Print interaction details (show every 5th to avoid clutter)
        if (i + 1) % 5 == 0 or i < 3:
            print(f"\n--- Interaction {i + 1} ---")
            print(f"Query: {query}")
            print(f"Strategy: {strategy.title()} {'(exploring)' if is_exploration else '(exploiting)'}")
            print(f"Response: {response[:100]}...")
            print(f"Rating: {'⭐' * rating} ({rating}/5)")

        # Process feedback
        agent.process_feedback(query, strategy, response, rating, is_exploration)

    print("\n" + "-" * 80)
    print("Interactions complete!")
    print("-" * 80)


def print_advanced_performance_report(agent: AdaptiveLearningAgentAdvanced):
    """Print comprehensive performance report"""
    report = agent.get_performance_report()

    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT: Multi-Armed Bandit Strategy Optimization")
    print("=" * 80)

    # Overall statistics
    overall = report["overall_stats"]
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Interactions: {overall['total_interactions']}")
    print(f"  Average Rating: {overall['overall_avg_rating']:.2f}/5")
    print(f"  Average Reward: {overall['overall_avg_reward']:.3f}")
    print(f"  Exploration Count: {overall['exploration_count']} ({overall['exploration_ratio']:.1%})")
    print(f"  Exploitation Count: {overall['exploitation_count']}")

    # Bandit statistics
    bandit = report["bandit_stats"]
    print(f"\n🎰 Multi-Armed Bandit Statistics:")
    print(f"  Best Strategy: {bandit['best_strategy'].title()}")
    print(f"  Exploration Rate (ε): {bandit['exploration_rate']:.1%}")

    # Strategy performance
    print(f"\n🎯 Strategy Performance:")
    strategy_perf = report["strategy_performance"]

    # Sort by average rating
    sorted_strategies = sorted(strategy_perf.items(), key=lambda x: x[1]["avg_rating"], reverse=True)

    for name, perf in sorted_strategies:
        if perf["count"] > 0:
            print(f"\n  {name.title()}:")
            print(f"    Usage Count: {perf['count']}")
            print(f"    Average Rating: {perf['avg_rating']:.2f}/5 ± {perf['rating_std']:.2f}")
            print(f"    Average Reward: {perf['avg_reward']:.3f}")
            print(f"    Rating Range: {perf['min_rating']:.0f} - {perf['max_rating']:.0f}")

    # Key insights
    print(f"\n💡 Key Insights:")

    best_strategy = bandit["best_strategy"]
    best_perf = strategy_perf[best_strategy]

    print(f"  • The '{best_strategy.title()}' strategy performed best with {best_perf['avg_rating']:.2f}/5 average rating")
    print(f"  • Used {best_perf['count']} times ({best_perf['count']/overall['total_interactions']:.1%} of interactions)")

    if overall["overall_avg_rating"] > 3.5:
        print(f"  • Overall performance is strong (>3.5/5 average)")
    else:
        print(f"  • Room for improvement in overall performance")

    print(
        f"  • The bandit balanced exploration ({overall['exploration_ratio']:.1%}) with exploitation ({1-overall['exploration_ratio']:.1%})"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print(
        """
    ╔═══════════════════════════════════════════════════════════════╗
    ║    Adaptive Learning Pattern - Advanced Implementation        ║
    ║                                                               ║
    ║  Multi-Armed Bandit for Strategy Optimization                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    )

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create advanced adaptive learning agent
    # epsilon = 0.2 means 20% exploration, 80% exploitation
    agent = AdaptiveLearningAgentAdvanced(llm, epsilon=0.2)

    # Simulate interactions
    simulate_advanced_interactions(agent, num_interactions=30)

    # Print performance report
    print_advanced_performance_report(agent)

    # Visualize learning progress
    visualize_advanced_learning(agent)

    print(
        """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Simulation Complete!                       ║
    ║                                                               ║
    ║  The agent demonstrated:                                      ║
    ║  ✅ Multiple competing response strategies                    ║
    ║  ✅ Multi-armed bandit optimization                           ║
    ║  ✅ Exploration vs exploitation balance                       ║
    ║  ✅ Real-time strategy adaptation                             ║
    ║  ✅ Performance tracking and comparison                       ║
    ║  ✅ Comprehensive visualization                               ║
    ║                                                               ║
    ║  Key Insight: The bandit algorithm automatically discovered   ║
    ║  which strategies work best by intelligently balancing        ║
    ║  exploration of new approaches with exploitation of known     ║
    ║  successful strategies.                                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    )
