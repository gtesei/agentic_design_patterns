"""
Basic Graph of Thoughts Implementation

Demonstrates:
- Multi-perspective thought generation (ethical, practical, economic, legal)
- DAG (Directed Acyclic Graph) structure
- Thought connections (support, critique, build-on)
- PageRank-based aggregation
- Simple ASCII visualization
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import os
from dataclasses import dataclass
from typing import Literal

import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from parent directory
load_dotenv("../../.env")


@dataclass
class ThoughtNode:
    """Represents a single thought in the graph"""

    id: str
    content: str
    perspective: str
    score: float = 0.0


@dataclass
class ConnectionEvaluation:
    """Evaluation of connection between two thoughts"""

    exists: bool
    relationship_type: Literal["support", "critique", "build-on", "prerequisite"] | None
    strength: float  # 0.0 to 1.0
    explanation: str


def print_header(text: str) -> None:
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str) -> None:
    """Print a formatted section"""
    print("\n" + "-" * 70)
    print(f"{text}")
    print("-" * 70)


def generate_perspective(problem: str, perspective: str, llm: ChatOpenAI) -> str:
    """Generate a thought from a specific perspective"""
    prompt = f"""You are analyzing a problem from a {perspective} perspective.

Problem: {problem}

Provide your analysis from the {perspective} viewpoint. Be specific and concise (2-3 sentences).
Focus on the key considerations relevant to this perspective.

Your analysis:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def evaluate_connection(
    thought_a: ThoughtNode, thought_b: ThoughtNode, problem: str, llm: ChatOpenAI
) -> ConnectionEvaluation:
    """Evaluate if and how two thoughts are connected"""

    # Quick heuristic: don't connect thoughts from same perspective to each other
    if thought_a.perspective == thought_b.perspective:
        return ConnectionEvaluation(
            exists=False, relationship_type=None, strength=0.0, explanation="Same perspective"
        )

    prompt = f"""Analyze the relationship between these two thoughts about a problem.

Problem: {problem}

Thought A ({thought_a.perspective}): {thought_a.content}

Thought B ({thought_b.perspective}): {thought_b.content}

Does Thought A have a meaningful relationship to Thought B?

If yes, classify the relationship as one of:
- "support": Thought A reinforces or strengthens Thought B
- "critique": Thought A challenges or questions Thought B
- "build-on": Thought A extends or develops Thought B's idea
- "prerequisite": Thought B depends on or requires Thought A

Respond in this exact format:
RELATIONSHIP: [support/critique/build-on/prerequisite/none]
STRENGTH: [0.0-1.0]
EXPLANATION: [one sentence explanation]"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse response
    lines = content.split("\n")
    relationship = None
    strength = 0.0
    explanation = ""

    for line in lines:
        if line.startswith("RELATIONSHIP:"):
            rel = line.split(":", 1)[1].strip().lower()
            if rel in ["support", "critique", "build-on", "prerequisite"]:
                relationship = rel
            else:
                relationship = None
        elif line.startswith("STRENGTH:"):
            try:
                strength = float(line.split(":", 1)[1].strip())
            except ValueError:
                strength = 0.0
        elif line.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    exists = relationship is not None and strength > 0.3

    return ConnectionEvaluation(
        exists=exists, relationship_type=relationship, strength=strength, explanation=explanation
    )


def evaluate_thought(thought: str, problem: str, llm: ChatOpenAI) -> float:
    """Evaluate the quality of a thought"""
    prompt = f"""Evaluate the quality of this thought regarding the problem.

Problem: {problem}

Thought: {thought}

Rate this thought on a scale of 1-10 based on:
- Relevance: How relevant is it to the problem?
- Insight: Does it provide valuable perspective?
- Clarity: Is it clear and well-articulated?
- Actionability: Does it contribute to a solution?

Respond with only a number between 1.0 and 10.0."""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(1.0, min(10.0, score))  # Clamp to 1-10
    except ValueError:
        return 5.0  # Default if parsing fails


def synthesize_solution(
    graph: nx.DiGraph, top_thought_ids: list[str], thoughts: dict[str, ThoughtNode], problem: str, llm: ChatOpenAI
) -> str:
    """Synthesize final solution from top thoughts"""

    # Prepare context from top thoughts
    thought_summaries = []
    for tid in top_thought_ids:
        thought = thoughts[tid]
        thought_summaries.append(f"- [{thought.perspective.upper()}] {thought.content}")

    # Include graph structure info
    connections = []
    for tid in top_thought_ids:
        edges = list(graph.out_edges(tid, data=True))
        if edges:
            for _, target, data in edges:
                if target in top_thought_ids:
                    connections.append(
                        f"  {tid} --{data['type']}--> {target} (strength: {data['weight']:.2f})"
                    )

    prompt = f"""Synthesize a comprehensive answer to the problem by integrating these key perspectives.

Problem: {problem}

Key Perspectives:
{chr(10).join(thought_summaries)}

Thought Connections:
{chr(10).join(connections) if connections else "  (perspectives analyzed independently)"}

Provide a comprehensive, balanced answer that:
1. Synthesizes insights from multiple perspectives
2. Acknowledges connections and relationships between viewpoints
3. Addresses potential conflicts or trade-offs
4. Offers actionable recommendations

Your synthesis:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def visualize_graph(graph: nx.DiGraph, thoughts: dict[str, ThoughtNode], scores: dict[str, float]) -> None:
    """Display ASCII visualization of the thought graph"""
    print_section("Graph Structure Visualization")

    # Show nodes with scores
    print("\nThought Nodes:")
    for tid in sorted(thoughts.keys()):
        thought = thoughts[tid]
        pr_score = scores.get(tid, 0)
        print(f"\n  [{tid}] {thought.perspective.upper()} (Score: {thought.score:.1f}/10, PageRank: {pr_score:.3f})")
        print(f"    {thought.content[:100]}{'...' if len(thought.content) > 100 else ''}")

    # Show edges
    print("\n\nThought Connections:")
    if graph.number_of_edges() > 0:
        for source, target, data in graph.edges(data=True):
            rel_type = data["type"]
            weight = data["weight"]
            # Visual representation
            arrow = "→"
            if rel_type == "support":
                symbol = "✓"
            elif rel_type == "critique":
                symbol = "✗"
            elif rel_type == "build-on":
                symbol = "⊕"
            else:
                symbol = "◆"

            print(f"  {source} {arrow} {target}")
            print(f"    {symbol} {rel_type} (strength: {weight:.2f})")
    else:
        print("  No connections formed (thoughts analyzed independently)")


def run_basic_got(problem: str) -> str:
    """
    Run basic Graph of Thoughts with DAG structure

    Args:
        problem: The problem to analyze

    Returns:
        Final synthesized solution
    """
    print_header("Graph of Thoughts - Basic Implementation")
    print(f"\nProblem: {problem}\n")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Initialize graph and data structures
    graph = nx.DiGraph()
    thoughts: dict[str, ThoughtNode] = {}

    # Step 1: Generate thoughts from different perspectives
    print_section("Step 1: Generating Perspectives")
    perspectives = ["ethical", "practical", "economic", "legal"]

    for i, perspective in enumerate(perspectives):
        print(f"\n  Generating {perspective} perspective...")
        content = generate_perspective(problem, perspective, llm)

        thought_id = f"T{i}_{perspective}"
        thought = ThoughtNode(id=thought_id, content=content, perspective=perspective)

        thoughts[thought_id] = thought
        graph.add_node(thought_id, thought=thought)

        print(f"  [{thought_id}] {content[:120]}{'...' if len(content) > 120 else ''}")

    # Step 2: Form connections between thoughts
    print_section("Step 2: Forming Connections")
    print("\n  Evaluating potential connections...")

    connection_count = 0
    for tid_a in thoughts:
        for tid_b in thoughts:
            if tid_a < tid_b:  # Avoid duplicate checks
                evaluation = evaluate_connection(thoughts[tid_a], thoughts[tid_b], problem, llm)

                if evaluation.exists:
                    # Add edge from A to B
                    graph.add_edge(
                        tid_a,
                        tid_b,
                        type=evaluation.relationship_type,
                        weight=evaluation.strength,
                        explanation=evaluation.explanation,
                    )
                    connection_count += 1
                    print(f"\n  ✓ {tid_a} --{evaluation.relationship_type}--> {tid_b}")
                    print(f"    Strength: {evaluation.strength:.2f} | {evaluation.explanation}")

    print(f"\n  Total connections formed: {connection_count}")

    # Step 3: Evaluate thoughts
    print_section("Step 3: Evaluating Thoughts")
    print("\n  Scoring individual thoughts...")

    for tid, thought in thoughts.items():
        score = evaluate_thought(thought.content, problem, llm)
        thought.score = score
        graph.nodes[tid]["score"] = score
        print(f"  [{tid}] {thought.perspective}: {score:.1f}/10")

    # Step 4: Calculate graph metrics
    print_section("Step 4: Graph Analysis")

    # PageRank (if there are edges)
    if graph.number_of_edges() > 0:
        pagerank = nx.pagerank(graph, weight="weight")
        print("\n  PageRank scores (influence in graph):")
        for tid in sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True):
            print(f"    {tid}: {pagerank[tid]:.3f}")
    else:
        # Equal weights if no connections
        pagerank = {tid: 1.0 / len(thoughts) for tid in thoughts}
        print("\n  No connections formed - using equal weights")

    # Combine PageRank with quality scores
    combined_scores = {tid: 0.6 * (thoughts[tid].score / 10.0) + 0.4 * pagerank[tid] for tid in thoughts}

    # Step 5: Visualize graph
    visualize_graph(graph, thoughts, pagerank)

    # Step 6: Synthesize solution
    print_section("Step 5: Synthesizing Solution")

    # Select top thoughts based on combined scores
    top_k = min(3, len(thoughts))
    top_thought_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

    print(f"\n  Using top {top_k} thoughts for synthesis:")
    for tid in top_thought_ids:
        print(f"    {tid} ({thoughts[tid].perspective}): combined score = {combined_scores[tid]:.3f}")

    print("\n  Generating final synthesis...\n")
    final_answer = synthesize_solution(graph, top_thought_ids, thoughts, problem, llm)

    # Display final answer
    print_header("Final Synthesized Solution")
    print(f"\n{final_answer}\n")

    # Summary statistics
    print_section("Summary Statistics")
    print(f"  Total perspectives analyzed: {len(thoughts)}")
    print(f"  Connections formed: {graph.number_of_edges()}")
    print(f"  Graph density: {nx.density(graph):.2%}")
    if graph.number_of_edges() > 0:
        print(f"  Average path length: {nx.average_shortest_path_length(graph):.2f}")

    return final_answer


def main():
    """Main execution"""

    # Example problem: Ethical dilemma requiring multiple perspectives
    problem = (
        "Should autonomous vehicles prioritize passenger safety or pedestrian safety "
        "in unavoidable accident scenarios?"
    )

    # Alternative problems to try:
    # problem = "Should social media platforms be legally liable for user-generated content?"
    # problem = "Is it ethical for employers to monitor employee productivity through AI?"
    # problem = "Should we prioritize colonizing Mars or solving Earth's climate crisis?"

    try:
        result = run_basic_got(problem)

        print("\n" + "=" * 70)
        print("  Graph of Thoughts Completed Successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
