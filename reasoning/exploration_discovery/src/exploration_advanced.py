"""
Exploration and Discovery Pattern: Advanced Implementation
This example demonstrates the UCB (Upper Confidence Bound) exploration strategy
for optimized discovery with multi-dimensional evaluation and clustering.
"""

import os
import random
from typing import List, Dict, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0.9, model="gpt-4o-mini")  # High temperature for creativity


class SemanticNoveltyDetector:
    """Advanced novelty detection using TF-IDF and cosine similarity"""

    def __init__(self):
        self.discovered_ideas: List[str] = []
        self.vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )
        self.idea_vectors = None

    def compute_novelty(self, new_idea: str) -> float:
        """
        Compute semantic novelty using TF-IDF vectors.

        Args:
            new_idea: The new idea to evaluate

        Returns:
            Novelty score between 0.0 and 1.0
        """
        if not self.discovered_ideas:
            return 1.0

        # Add new idea temporarily to vectorize
        all_ideas = self.discovered_ideas + [new_idea]

        # Vectorize
        try:
            vectors = self.vectorizer.fit_transform(all_ideas)
            new_vector = vectors[-1]
            existing_vectors = vectors[:-1]

            # Compute cosine similarity
            similarities = cosine_similarity(new_vector, existing_vectors)
            max_similarity = similarities.max()

            # Novelty is inverse of similarity
            novelty = float(1.0 - max_similarity)
            return novelty
        except Exception:
            # Fallback to simple word overlap
            new_words = set(new_idea.lower().split())
            max_similarity = 0.0
            for existing_idea in self.discovered_ideas:
                existing_words = set(existing_idea.lower().split())
                intersection = len(new_words & existing_words)
                union = len(new_words | existing_words)
                if union > 0:
                    similarity = intersection / union
                    max_similarity = max(max_similarity, similarity)
            return 1.0 - max_similarity

    def add_discovery(self, idea: str):
        """Add a new discovery"""
        self.discovered_ideas.append(idea)

    def get_clusters(self, n_clusters: int = 5) -> List[List[int]]:
        """
        Cluster discoveries to analyze diversity.

        Args:
            n_clusters: Target number of clusters

        Returns:
            List of clusters (each cluster is a list of discovery indices)
        """
        if len(self.discovered_ideas) < n_clusters:
            # Each idea is its own cluster
            return [[i] for i in range(len(self.discovered_ideas))]

        # Vectorize all ideas
        vectors = self.vectorizer.fit_transform(self.discovered_ideas)

        # Cluster using agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(self.discovered_ideas)),
            metric="cosine",
            linkage="average",
        )

        # Fit and get labels
        dense_vectors = vectors.toarray()
        labels = clustering.fit_predict(dense_vectors)

        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        return list(clusters.values())


class UCBExplorer:
    """
    Upper Confidence Bound (UCB) exploration strategy.

    Optimizes exploration efficiency by balancing average reward with uncertainty.
    Clusters with high uncertainty (few visits) get exploration bonuses.
    """

    def __init__(self, c: float = 1.414, n_clusters: int = 5):
        """
        Initialize UCB explorer.

        Args:
            c: Exploration constant (typically sqrt(2) ≈ 1.414)
            n_clusters: Number of solution clusters to track
        """
        self.c = c
        self.n_clusters = n_clusters
        self.total_iterations = 0
        self.discoveries: List[Dict] = []
        self.cluster_stats: Dict[str, Dict] = {}
        self.novelty_detector = SemanticNoveltyDetector()

    def compute_ucb(self, cluster_id: str) -> float:
        """
        Compute UCB score for a cluster.

        UCB = average_reward + c * sqrt(ln(total_iterations) / cluster_visits)

        Args:
            cluster_id: The cluster to compute UCB for

        Returns:
            UCB score (higher = more attractive for exploration)
        """
        if cluster_id not in self.cluster_stats:
            return float("inf")  # New cluster has infinite UCB

        stats = self.cluster_stats[cluster_id]
        avg_reward = stats["total_reward"] / stats["visits"]

        # Exploration bonus
        exploration_bonus = self.c * np.sqrt(
            np.log(self.total_iterations + 1) / stats["visits"]
        )

        return avg_reward + exploration_bonus

    def select_cluster(self) -> Tuple[str, float]:
        """
        Select which cluster to explore based on UCB scores.

        Returns:
            Tuple of (cluster_id, ucb_score)
        """
        if not self.cluster_stats or random.random() < 0.2:  # 20% chance to explore new
            return "new_cluster", float("inf")

        # Compute UCB for all clusters
        ucb_scores = {
            cluster_id: self.compute_ucb(cluster_id)
            for cluster_id in self.cluster_stats.keys()
        }

        # Select cluster with highest UCB
        best_cluster = max(ucb_scores.items(), key=lambda x: x[1])
        return best_cluster

    def evaluate_idea(self, idea: str) -> Dict[str, float]:
        """
        Multi-dimensional evaluation of an idea.

        Args:
            idea: The idea to evaluate

        Returns:
            Dictionary with scores for novelty, feasibility, impact, and overall
        """
        # Novelty from detector
        novelty = self.novelty_detector.compute_novelty(idea)

        # Feasibility: Use LLM to evaluate
        feasibility = self._evaluate_feasibility(idea)

        # Impact: Use LLM to evaluate
        impact = self._evaluate_impact(idea)

        # Overall score
        overall = 0.35 * novelty + 0.35 * feasibility + 0.30 * impact

        return {
            "novelty": novelty,
            "feasibility": feasibility,
            "impact": impact,
            "overall": overall,
        }

    def _evaluate_feasibility(self, idea: str) -> float:
        """Evaluate how feasible an idea is"""
        prompt = f"""Evaluate the feasibility of this idea on a scale of 0.0 to 1.0:

Idea: {idea}

Consider:
- Technical feasibility
- Resource requirements
- Implementation complexity
- Time to market

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            # Fallback: length-based heuristic
            return max(0.3, min(1.0, 1.0 - len(idea) / 500))

    def _evaluate_impact(self, idea: str) -> float:
        """Evaluate the potential impact of an idea"""
        prompt = f"""Evaluate the potential impact/value of this idea on a scale of 0.0 to 1.0:

Idea: {idea}

Consider:
- Market size / target audience
- Problem significance
- Potential for positive change
- Competitive advantage

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            # Fallback: keyword-based heuristic
            impact_keywords = [
                "sustainable",
                "efficient",
                "innovative",
                "scalable",
                "revolutionary",
                "breakthrough",
                "transformative",
                "disruptive",
            ]
            impact_score = sum(
                1 for keyword in impact_keywords if keyword.lower() in idea.lower()
            )
            return min(1.0, 0.5 + (impact_score * 0.1))

    def generate_idea(self, prompt: str, cluster_id: str, cluster_examples: List[str] = None) -> str:
        """
        Generate an idea, optionally guided by a cluster.

        Args:
            prompt: Base exploration prompt
            cluster_id: Target cluster to explore
            cluster_examples: Example ideas from the cluster

        Returns:
            Generated idea string
        """
        if cluster_id == "new_cluster" or not cluster_examples:
            # Explore entirely new territory
            generation_prompt = f"""{prompt}

Generate a highly creative, novel, and unconventional solution. Think outside the box.
Explore unexplored territory and come up with something truly unique.

Provide just the idea in 1-2 sentences, nothing else."""

        else:
            # Explore within the selected cluster
            examples_text = "\n".join([f"- {ex}" for ex in cluster_examples[:3]])
            generation_prompt = f"""{prompt}

Here are some related ideas in a promising direction:
{examples_text}

Generate a NEW idea that explores this same general direction but with a unique twist.
Build on these themes but don't repeat them.

Provide just the idea in 1-2 sentences, nothing else."""

        response = llm.invoke(generation_prompt)
        return response.content.strip()

    def update_cluster_stats(self, cluster_id: str, reward: float, idea: str):
        """Update cluster statistics with new observation"""
        if cluster_id not in self.cluster_stats:
            self.cluster_stats[cluster_id] = {
                "visits": 0,
                "total_reward": 0.0,
                "examples": [],
            }

        stats = self.cluster_stats[cluster_id]
        stats["visits"] += 1
        stats["total_reward"] += reward
        stats["examples"].append(idea)

        # Keep only recent examples
        if len(stats["examples"]) > 5:
            stats["examples"] = stats["examples"][-5:]

    def assign_cluster(self, idea: str) -> str:
        """
        Assign an idea to a cluster based on semantic similarity.

        Args:
            idea: The idea to assign

        Returns:
            Cluster ID (or "new_cluster")
        """
        if not self.cluster_stats:
            return "cluster_0"

        # Check similarity to each cluster's examples
        best_cluster = None
        best_similarity = 0.0

        for cluster_id, stats in self.cluster_stats.items():
            if not stats["examples"]:
                continue

            # Compare to cluster examples
            for example in stats["examples"]:
                try:
                    # Simple word overlap similarity
                    idea_words = set(idea.lower().split())
                    example_words = set(example.lower().split())
                    intersection = len(idea_words & example_words)
                    union = len(idea_words | example_words)
                    if union > 0:
                        similarity = intersection / union
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_cluster = cluster_id
                except Exception:
                    pass

        # If similarity is high enough, assign to cluster
        if best_similarity > 0.3 and best_cluster:
            return best_cluster

        # Otherwise, create new cluster
        new_cluster_id = f"cluster_{len(self.cluster_stats)}"
        return new_cluster_id

    def explore(self, prompt: str, max_iterations: int = 25, novelty_threshold: float = 0.6) -> Dict:
        """
        Run UCB-guided exploration.

        Args:
            prompt: Exploration prompt
            max_iterations: Maximum iterations
            novelty_threshold: Minimum novelty to accept

        Returns:
            Dictionary with discoveries and statistics
        """
        print(f"\n{'='*80}")
        print("UCB (UPPER CONFIDENCE BOUND) EXPLORATION")
        print(f"{'='*80}\n")
        print(f"Prompt: {prompt}")
        print(f"Max Iterations: {max_iterations}")
        print(f"UCB Constant (c): {self.c}\n")

        for iteration in range(1, max_iterations + 1):
            self.total_iterations = iteration

            # Select cluster using UCB
            cluster_id, ucb_score = self.select_cluster()

            # Get cluster examples if available
            cluster_examples = None
            if cluster_id in self.cluster_stats:
                cluster_examples = self.cluster_stats[cluster_id]["examples"]

            # Generate idea
            idea = self.generate_idea(prompt, cluster_id, cluster_examples)

            # Evaluate
            scores = self.evaluate_idea(idea)

            # Check novelty threshold
            if scores["novelty"] < novelty_threshold:
                print(f"\nIteration {iteration}/{max_iterations}")
                print("─" * 80)
                print(f"Target Cluster: {cluster_id} (UCB: {ucb_score:.3f})")
                print(f"💡 Idea: {idea}")
                print(f"✗ Rejected: Novelty ({scores['novelty']:.2f}) below threshold")
                continue

            # Assign to actual cluster (may differ from target)
            actual_cluster = self.assign_cluster(idea)

            # Record discovery
            discovery = {
                "iteration": iteration,
                "cluster": actual_cluster,
                "target_cluster": cluster_id,
                "ucb_score": ucb_score,
                "idea": idea,
                **scores,
            }
            self.discoveries.append(discovery)
            self.novelty_detector.add_discovery(idea)

            # Update cluster statistics
            self.update_cluster_stats(actual_cluster, scores["overall"], idea)

            # Display iteration
            self._display_iteration(iteration, max_iterations, discovery)

        # Display final results
        self._display_results()

        return {
            "discoveries": self.discoveries,
            "cluster_stats": self.cluster_stats,
            "total_iterations": self.total_iterations,
        }

    def _display_iteration(self, iteration: int, max_iterations: int, discovery: Dict):
        """Display iteration information"""
        print(f"\nIteration {iteration}/{max_iterations}")
        print("━" * 80)
        print(f"🎯 Target Cluster: {discovery['target_cluster']}")
        print(f"   UCB Score: {discovery['ucb_score']:.3f}")

        if discovery["cluster"] in self.cluster_stats:
            stats = self.cluster_stats[discovery["cluster"]]
            avg_reward = stats["total_reward"] / stats["visits"]
            exploration_bonus = discovery["ucb_score"] - avg_reward if discovery["ucb_score"] != float("inf") else 0
            print(f"   Avg Reward: {avg_reward:.2f} | Visits: {stats['visits']} | Exploration Bonus: {exploration_bonus:.3f}")

        print(f"\n💡 Generated Idea:")
        print(f"   {discovery['idea']}")

        print(f"\n📊 Evaluation:")
        print(f"  Novelty:     {'█' * int(discovery['novelty'] * 10)}{'░' * (10 - int(discovery['novelty'] * 10))} {discovery['novelty']:.2f}")
        print(f"  Feasibility: {'█' * int(discovery['feasibility'] * 10)}{'░' * (10 - int(discovery['feasibility'] * 10))} {discovery['feasibility']:.2f}")
        print(f"  Impact:      {'█' * int(discovery['impact'] * 10)}{'░' * (10 - int(discovery['impact'] * 10))} {discovery['impact']:.2f}")
        print(f"  Overall:     {'█' * int(discovery['overall'] * 10)}{'░' * (10 - int(discovery['overall'] * 10))} {discovery['overall']:.2f}")

        print(f"\n✓ Assigned to: {discovery['cluster']}")
        print(f"   Total Discoveries: {len(self.discoveries)}")
        print(f"   Active Clusters: {len(self.cluster_stats)}")

    def _display_results(self):
        """Display final exploration results"""
        print(f"\n\n{'='*80}")
        print("UCB EXPLORATION COMPLETE")
        print(f"{'='*80}\n")

        # Top discoveries
        sorted_discoveries = sorted(self.discoveries, key=lambda x: x["overall"], reverse=True)

        print("🏆 Top 5 Discoveries by Overall Score:")
        print("─" * 80)
        for i, discovery in enumerate(sorted_discoveries[:5], 1):
            print(f"\n{i}. Overall: {discovery['overall']:.2f} | Cluster: {discovery['cluster']} | Iteration: {discovery['iteration']}")
            print(f"   {discovery['idea']}")
            print(f"   Novelty: {discovery['novelty']:.2f} | Feasibility: {discovery['feasibility']:.2f} | Impact: {discovery['impact']:.2f}")

        # Cluster analysis
        print(f"\n\n📊 Cluster Analysis:")
        print("─" * 80)
        print(f"Total Clusters Discovered: {len(self.cluster_stats)}\n")

        for cluster_id, stats in sorted(
            self.cluster_stats.items(), key=lambda x: x[1]["total_reward"] / x[1]["visits"], reverse=True
        ):
            avg_reward = stats["total_reward"] / stats["visits"]
            print(f"\n{cluster_id}:")
            print(f"  Visits: {stats['visits']}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Example: {stats['examples'][0] if stats['examples'] else 'None'}")

        # Diversity metrics
        print(f"\n\n🎨 Diversity Metrics:")
        print("─" * 80)

        novelty_scores = [d["novelty"] for d in self.discoveries]
        print(f"Average Novelty: {np.mean(novelty_scores):.2f}")
        print(f"Novelty Std Dev: {np.std(novelty_scores):.2f}")

        # Cluster distribution
        cluster_counts = defaultdict(int)
        for d in self.discoveries:
            cluster_counts[d["cluster"]] += 1

        entropy = -sum(
            (count / len(self.discoveries)) * np.log(count / len(self.discoveries))
            for count in cluster_counts.values()
        )
        print(f"Cluster Entropy: {entropy:.2f} (higher = more diverse)")

        # UCB efficiency
        print(f"\n\n⚡ UCB Exploration Efficiency:")
        print("─" * 80)
        print(f"Total Discoveries: {len(self.discoveries)}")
        print(f"Discoveries per Cluster: {len(self.discoveries) / len(self.cluster_stats):.1f}")

        avg_scores = [d["overall"] for d in self.discoveries]
        print(f"Average Overall Score: {np.mean(avg_scores):.3f}")
        print(f"Best Overall Score: {max(avg_scores):.3f}")

        # Score trajectory
        early_avg = np.mean([d["overall"] for d in self.discoveries[: len(self.discoveries) // 2]])
        late_avg = np.mean([d["overall"] for d in self.discoveries[len(self.discoveries) // 2 :]])
        print(f"\nScore Trajectory:")
        print(f"  Early Average (first half): {early_avg:.3f}")
        print(f"  Late Average (second half): {late_avg:.3f}")
        print(f"  Improvement: {'+' if late_avg > early_avg else ''}{(late_avg - early_avg):.3f}")

        print(f"\n{'='*80}\n")


def run_advanced_example(prompt: str, max_iterations: int = 20):
    """Run an advanced UCB exploration example"""
    explorer = UCBExplorer(
        c=1.414,  # Standard exploration constant (sqrt(2))
        n_clusters=5,  # Track up to 5 solution clusters
    )

    results = explorer.explore(
        prompt=prompt,
        max_iterations=max_iterations,
        novelty_threshold=0.6,
    )

    return results


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║        Exploration and Discovery - Advanced UCB Implementation                ║
    ║                                                                               ║
    ║  This example demonstrates UCB (Upper Confidence Bound) exploration,          ║
    ║  which optimizes exploration efficiency by balancing average reward           ║
    ║  with uncertainty. Watch as the agent discovers solution clusters             ║
    ║  and strategically explores high-potential areas.                             ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Product feature discovery
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Product Feature Discovery")
    print("=" * 80)

    run_advanced_example(
        prompt="Explore innovative features for a next-generation project management tool. "
        "Focus on features that leverage AI, improve collaboration, or enhance productivity.",
        max_iterations=20,
    )

    # Example 2: Research hypothesis generation
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Research Hypothesis Discovery")
    print("=" * 80)

    run_advanced_example(
        prompt="Generate research hypotheses about the impact of artificial intelligence on creative work. "
        "Focus on testable hypotheses exploring different aspects: cognitive effects, workflow changes, "
        "skill development, or human-AI collaboration.",
        max_iterations=18,
    )

    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                          Examples Complete!                                   ║
    ║                                                                               ║
    ║  The UCB Explorer demonstrated:                                              ║
    ║  • Upper Confidence Bound algorithm for optimized exploration                ║
    ║  • Multi-dimensional evaluation with LLM-based scoring                       ║
    ║  • Semantic similarity for advanced novelty detection                        ║
    ║  • Automatic solution clustering and diversity analysis                      ║
    ║  • Adaptive exploration based on cluster uncertainty                         ║
    ║  • Efficient discovery with theoretical guarantees                           ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
