"""
Exploration and Discovery Pattern: Basic Implementation
This example demonstrates the epsilon-greedy exploration strategy for creative
discovery tasks like brainstorming business ideas or generating research directions.
"""

import os
import random
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import numpy as np

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0.9, model="gpt-4o-mini")  # High temperature for creativity


class NoveltyDetector:
    """Simple novelty detection using string similarity"""

    def __init__(self):
        self.discovered_ideas: List[str] = []

    def compute_novelty(self, new_idea: str) -> float:
        """
        Compute how novel an idea is compared to existing discoveries.

        Args:
            new_idea: The new idea to evaluate

        Returns:
            Novelty score between 0.0 (duplicate) and 1.0 (completely novel)
        """
        if not self.discovered_ideas:
            return 1.0  # First idea is maximally novel

        # Compute word-level Jaccard similarity to existing ideas
        new_words = set(new_idea.lower().split())
        max_similarity = 0.0

        for existing_idea in self.discovered_ideas:
            existing_words = set(existing_idea.lower().split())

            # Jaccard similarity: intersection / union
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)

            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity
        return novelty

    def add_discovery(self, idea: str):
        """Add a new discovery to the tracking list"""
        self.discovered_ideas.append(idea)


class EpsilonGreedyExplorer:
    """
    Epsilon-Greedy exploration strategy.

    Balances exploration (trying new random ideas) with exploitation
    (refining the best ideas found so far).
    """

    def __init__(
        self,
        epsilon: float = 0.9,
        epsilon_decay: float = 0.95,
        min_epsilon: float = 0.1,
    ):
        """
        Initialize the explorer.

        Args:
            epsilon: Initial exploration rate (0.0-1.0, typically 0.9)
            epsilon_decay: Decay rate per iteration (typically 0.95)
            min_epsilon: Minimum exploration rate (typically 0.1)
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discoveries: List[Dict] = []
        self.novelty_detector = NoveltyDetector()

    def should_explore(self) -> bool:
        """
        Decide whether to explore (random) or exploit (refine best).

        Returns:
            True if should explore, False if should exploit
        """
        return random.random() < self.epsilon

    def evaluate_idea(self, idea: str) -> Dict[str, float]:
        """
        Evaluate an idea on multiple dimensions.

        Args:
            idea: The idea to evaluate

        Returns:
            Dictionary with scores for novelty, feasibility, impact, and overall
        """
        # Compute novelty using detector
        novelty = self.novelty_detector.compute_novelty(idea)

        # Simple heuristic evaluations (in production, use LLM-based evaluation)
        # Feasibility: inversely related to length (simpler = more feasible)
        feasibility = max(0.3, min(1.0, 1.0 - len(idea) / 500))

        # Impact: based on presence of impactful keywords
        impact_keywords = ["sustainable", "efficient", "innovative", "scalable", "revolutionary", "breakthrough"]
        impact_score = sum(1 for keyword in impact_keywords if keyword.lower() in idea.lower())
        impact = min(1.0, 0.5 + (impact_score * 0.1))

        # Overall score: weighted combination
        overall = 0.40 * novelty + 0.30 * feasibility + 0.30 * impact

        return {
            "novelty": novelty,
            "feasibility": feasibility,
            "impact": impact,
            "overall": overall,
        }

    def generate_idea(self, prompt: str, mode: str, best_idea: str = None) -> str:
        """
        Generate a new idea based on exploration mode.

        Args:
            prompt: The base prompt for idea generation
            mode: "explore" or "exploit"
            best_idea: The best idea so far (used in exploit mode)

        Returns:
            Generated idea string
        """
        if mode == "explore":
            # EXPLORE: Generate novel, creative idea
            explore_prompt = f"""{prompt}

Generate a highly creative, novel, and unconventional idea. Think outside the box.
Be specific and concrete. Aim for something unique that hasn't been thought of before.

Provide just the idea in 1-2 sentences, nothing else."""

            response = llm.invoke(explore_prompt)
            return response.content.strip()

        else:
            # EXPLOIT: Refine the best idea found so far
            exploit_prompt = f"""{prompt}

Here is a promising idea that has been discovered:
"{best_idea}"

Generate a refined, improved version of this idea. Make it more practical,
scalable, or impactful while maintaining its core innovation.

Provide just the refined idea in 1-2 sentences, nothing else."""

            response = llm.invoke(exploit_prompt)
            return response.content.strip()

    def explore(
        self,
        prompt: str,
        max_iterations: int = 20,
        novelty_threshold: float = 0.6,
    ) -> Dict:
        """
        Run the exploration process.

        Args:
            prompt: The exploration prompt (what to generate ideas about)
            max_iterations: Maximum number of exploration iterations
            novelty_threshold: Minimum novelty score to accept an idea

        Returns:
            Dictionary with all discoveries and statistics
        """
        print(f"\n{'='*80}")
        print("EPSILON-GREEDY EXPLORATION")
        print(f"{'='*80}\n")
        print(f"Prompt: {prompt}")
        print(f"Max Iterations: {max_iterations}")
        print(f"Initial ε: {self.epsilon:.2f}\n")

        best_overall_score = 0.0
        best_idea = None
        explore_count = 0
        exploit_count = 0

        for iteration in range(1, max_iterations + 1):
            # Decide mode
            mode = "explore" if self.should_explore() else "exploit"

            # Update counters
            if mode == "explore":
                explore_count += 1
            else:
                exploit_count += 1

            # Generate idea
            if mode == "exploit" and best_idea is None:
                # Can't exploit without a best idea yet, force exploration
                mode = "explore"
                explore_count += 1
                exploit_count -= 1

            idea = self.generate_idea(prompt, mode, best_idea)

            # Evaluate
            scores = self.evaluate_idea(idea)

            # Check novelty threshold
            if scores["novelty"] < novelty_threshold:
                print(f"\nIteration {iteration}/{max_iterations} (ε={self.epsilon:.2f})")
                print("─" * 80)
                print(f"🔄 Mode: {mode.upper()}")
                print(f"💡 Idea: {idea}")
                print(f"\n✗ Rejected: Novelty ({scores['novelty']:.2f}) below threshold ({novelty_threshold:.2f})")
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                continue

            # Record discovery
            discovery = {
                "iteration": iteration,
                "mode": mode,
                "idea": idea,
                **scores,
            }
            self.discoveries.append(discovery)
            self.novelty_detector.add_discovery(idea)

            # Update best
            if scores["overall"] > best_overall_score:
                best_overall_score = scores["overall"]
                best_idea = idea

            # Display iteration
            self._display_iteration(iteration, max_iterations, discovery)

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Display final results
        self._display_results(explore_count, exploit_count)

        return {
            "discoveries": self.discoveries,
            "best_idea": best_idea,
            "best_score": best_overall_score,
            "explore_count": explore_count,
            "exploit_count": exploit_count,
        }

    def _display_iteration(self, iteration: int, max_iterations: int, discovery: Dict):
        """Display iteration information"""
        print(f"\nIteration {iteration}/{max_iterations} (ε={self.epsilon:.2f})")
        print("━" * 80)
        print(f"🔍 Mode: {discovery['mode'].upper()}")
        print(f"💡 Idea: {discovery['idea']}")
        print(f"\n📊 Evaluation:")
        print(f"  Novelty:     {'█' * int(discovery['novelty'] * 10)}{'░' * (10 - int(discovery['novelty'] * 10))} {discovery['novelty']:.2f}")
        print(f"  Feasibility: {'█' * int(discovery['feasibility'] * 10)}{'░' * (10 - int(discovery['feasibility'] * 10))} {discovery['feasibility']:.2f}")
        print(f"  Impact:      {'█' * int(discovery['impact'] * 10)}{'░' * (10 - int(discovery['impact'] * 10))} {discovery['impact']:.2f}")
        print(f"  Overall:     {'█' * int(discovery['overall'] * 10)}{'░' * (10 - int(discovery['overall'] * 10))} {discovery['overall']:.2f}")
        print(f"\n✓ New Discovery Added")
        print(f"\nCurrent Portfolio:")
        print(f"  - Total Discoveries: {len(self.discoveries)}")
        print(f"  - Best Overall Score: {max(d['overall'] for d in self.discoveries):.2f}")

    def _display_results(self, explore_count: int, exploit_count: int):
        """Display final exploration results"""
        print(f"\n\n{'='*80}")
        print("EXPLORATION COMPLETE")
        print(f"{'='*80}\n")

        # Sort discoveries by overall score
        sorted_discoveries = sorted(self.discoveries, key=lambda x: x["overall"], reverse=True)

        print("🏆 Top 5 Discoveries by Overall Score:")
        print("─" * 80)
        for i, discovery in enumerate(sorted_discoveries[:5], 1):
            print(f"\n{i}. Overall Score: {discovery['overall']:.2f} | Iteration: {discovery['iteration']}")
            print(f"   {discovery['idea']}")
            print(f"   Novelty: {discovery['novelty']:.2f} | Feasibility: {discovery['feasibility']:.2f} | Impact: {discovery['impact']:.2f}")

        # Top by dimension
        print(f"\n\n📊 Top Discovery by Each Dimension:")
        print("─" * 80)

        top_novelty = max(self.discoveries, key=lambda x: x["novelty"])
        print(f"\n🌟 Most Novel (Score: {top_novelty['novelty']:.2f}):")
        print(f"   {top_novelty['idea']}")

        top_feasibility = max(self.discoveries, key=lambda x: x["feasibility"])
        print(f"\n🛠️  Most Feasible (Score: {top_feasibility['feasibility']:.2f}):")
        print(f"   {top_feasibility['idea']}")

        top_impact = max(self.discoveries, key=lambda x: x["impact"])
        print(f"\n💥 Highest Impact (Score: {top_impact['impact']:.2f}):")
        print(f"   {top_impact['idea']}")

        # Statistics
        print(f"\n\n📈 Exploration Statistics:")
        print("─" * 80)
        print(f"Total Discoveries: {len(self.discoveries)}")
        print(f"Exploration Iterations: {explore_count} ({explore_count / (explore_count + exploit_count) * 100:.1f}%)")
        print(f"Exploitation Iterations: {exploit_count} ({exploit_count / (explore_count + exploit_count) * 100:.1f}%)")
        print(f"\nAverage Scores:")
        print(f"  Novelty:     {np.mean([d['novelty'] for d in self.discoveries]):.2f}")
        print(f"  Feasibility: {np.mean([d['feasibility'] for d in self.discoveries]):.2f}")
        print(f"  Impact:      {np.mean([d['impact'] for d in self.discoveries]):.2f}")
        print(f"  Overall:     {np.mean([d['overall'] for d in self.discoveries]):.2f}")

        # Diversity analysis
        novelty_scores = [d["novelty"] for d in self.discoveries]
        diversity_score = np.mean(novelty_scores)
        print(f"\n🎨 Diversity Score: {diversity_score:.2f} (avg novelty across all discoveries)")

        print(f"\n{'='*80}\n")


def run_example(prompt: str, max_iterations: int = 15):
    """Run a basic exploration example"""
    explorer = EpsilonGreedyExplorer(
        epsilon=0.9,  # Start with 90% exploration
        epsilon_decay=0.95,  # Decay 5% per iteration
        min_epsilon=0.1,  # Never go below 10% exploration
    )

    results = explorer.explore(
        prompt=prompt,
        max_iterations=max_iterations,
        novelty_threshold=0.6,  # Only accept ideas with novelty > 0.6
    )

    return results


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║           Exploration and Discovery - Basic Implementation                    ║
    ║                                                                               ║
    ║  This example demonstrates epsilon-greedy exploration for creative            ║
    ║  discovery tasks. Watch as the agent balances exploring novel ideas          ║
    ║  with exploiting (refining) the best discoveries.                            ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Business idea generation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Creative Business Idea Generation")
    print("=" * 80)

    run_example(
        prompt="Generate innovative business ideas for sustainable urban living. "
        "Focus on practical solutions that improve quality of life while reducing environmental impact.",
        max_iterations=15,
    )

    # Example 2: Research directions
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Research Hypothesis Discovery")
    print("=" * 80)

    run_example(
        prompt="Generate research hypotheses about the impact of remote work on employee productivity and wellbeing. "
        "Focus on testable hypotheses that explore different factors and mechanisms.",
        max_iterations=12,
    )

    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                          Examples Complete!                                   ║
    ║                                                                               ║
    ║  The Epsilon-Greedy Explorer demonstrated:                                   ║
    ║  • Balancing exploration (novel ideas) with exploitation (refinement)        ║
    ║  • Multi-dimensional evaluation (novelty, feasibility, impact)               ║
    ║  • Novelty detection to avoid duplicates                                     ║
    ║  • Adaptive exploration rate (epsilon decay)                                 ║
    ║  • Diverse portfolio of discoveries                                          ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
