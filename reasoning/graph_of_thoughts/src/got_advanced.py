"""
Advanced Graph of Thoughts Implementation

Demonstrates:
- Multi-agent consensus building (optimist, critic, pragmatist, innovator)
- Multiple refinement rounds (iterative improvement)
- Critique and support relationships
- Weighted edge voting
- Round-by-round evolution visualization
- Convergence detection
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv("../../.env")


@dataclass
class ThoughtNode:
    """Represents a single thought in the graph"""

    id: str
    content: str
    perspective: str
    round_num: int
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ThoughtGraph:
    """Manages the graph of thoughts"""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    current_round: int = 1

    def add_thought(self, thought: ThoughtNode) -> None:
        """Add a thought to the graph"""
        self.nodes[thought.id] = thought
        self.graph.add_node(thought.id, thought=thought, score=thought.score)

    def add_connection(
        self, source_id: str, target_id: str, relationship_type: str, weight: float, explanation: str = ""
    ) -> None:
        """Add a connection between thoughts"""
        self.graph.add_edge(
            source_id, target_id, type=relationship_type, weight=weight, explanation=explanation
        )

    def get_thoughts_from_round(self, round_num: int) -> list[ThoughtNode]:
        """Get all thoughts from a specific round"""
        return [node for node in self.nodes.values() if node.round_num == round_num]

    def get_recent_thoughts(self, max_round: int, exclude_perspective: str = None) -> list[ThoughtNode]:
        """Get recent thoughts, optionally excluding a perspective"""
        thoughts = []
        for node in self.nodes.values():
            if node.round_num <= max_round:
                if exclude_perspective is None or node.perspective != exclude_perspective:
                    thoughts.append(node)
        return thoughts


def print_header(text: str) -> None:
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_round_header(round_num: int, title: str) -> None:
    """Print a round header"""
    print("\n" + "=" * 80)
    print(f"  ROUND {round_num}: {title}")
    print("=" * 80)


def print_section(text: str) -> None:
    """Print a formatted section"""
    print("\n" + "-" * 80)
    print(f"{text}")
    print("-" * 80)


def generate_initial_thought(problem: str, persona: str, llm: ChatOpenAI) -> str:
    """Generate initial thought from a persona's perspective"""

    persona_descriptions = {
        "optimist": "You see opportunities and potential. You focus on positive outcomes and possibilities.",
        "critic": "You identify risks and challenges. You focus on what could go wrong and potential problems.",
        "pragmatist": "You focus on practical implementation. You consider what's realistic and feasible.",
        "innovator": "You think creatively and unconventionally. You suggest novel approaches and solutions.",
    }

    prompt = f"""You are participating in a collaborative problem-solving session as the {persona}.

Your character: {persona_descriptions.get(persona, 'You provide balanced analysis.')}

Problem: {problem}

Provide your initial perspective on this problem. Be specific and authentic to your role.
(2-3 sentences)

Your perspective:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def generate_response_thought(
    problem: str, persona: str, context_thoughts: list[ThoughtNode], round_num: int, llm: ChatOpenAI
) -> str:
    """Generate response thought considering previous thoughts"""

    persona_descriptions = {
        "optimist": "You see opportunities and potential. You focus on positive outcomes and possibilities.",
        "critic": "You identify risks and challenges. You focus on what could go wrong and potential problems.",
        "pragmatist": "You focus on practical implementation. You consider what's realistic and feasible.",
        "innovator": "You think creatively and unconventionally. You suggest novel approaches and solutions.",
    }

    # Format context thoughts
    context_text = []
    for thought in context_thoughts[-6:]:  # Last 6 thoughts for context
        context_text.append(f"- [{thought.perspective.upper()}]: {thought.content}")

    prompt = f"""You are participating in round {round_num} of a collaborative problem-solving session as the {persona}.

Your character: {persona_descriptions.get(persona, 'You provide balanced analysis.')}

Problem: {problem}

Previous thoughts from the group:
{chr(10).join(context_text)}

Based on what others have said, provide your response. You can:
- Support and build on ideas you agree with
- Critique and challenge ideas you disagree with
- Offer refinements or alternatives
- Synthesize multiple perspectives

Stay authentic to your role but engage with the previous ideas.
(2-3 sentences)

Your response:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def evaluate_relationship(
    source: ThoughtNode, target: ThoughtNode, problem: str, llm: ChatOpenAI
) -> tuple[str | None, float, str]:
    """
    Evaluate relationship between two thoughts

    Returns:
        (relationship_type, strength, explanation)
    """

    # Don't connect thoughts from same round and perspective
    if source.round_num == target.round_num and source.perspective == target.perspective:
        return None, 0.0, "Same round and perspective"

    # Only form edges where source comes after target (can reference earlier thoughts)
    if source.round_num < target.round_num:
        return None, 0.0, "Cannot reference future thoughts"

    prompt = f"""Analyze if there is a meaningful relationship between these thoughts.

Problem context: {problem}

Earlier thought [{target.perspective}, Round {target.round_num}]:
{target.content}

Later thought [{source.perspective}, Round {source.round_num}]:
{source.content}

Does the later thought have a relationship to the earlier thought?

Classify as:
- "support": Later thought agrees with or reinforces earlier thought
- "critique": Later thought challenges or questions earlier thought
- "build-on": Later thought extends or develops the earlier thought's idea
- "none": No meaningful relationship

Format:
TYPE: [support/critique/build-on/none]
STRENGTH: [0.0-1.0]
REASON: [brief explanation]"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse response
    rel_type = None
    strength = 0.0
    reason = ""

    for line in content.split("\n"):
        if line.startswith("TYPE:"):
            t = line.split(":", 1)[1].strip().lower()
            if t in ["support", "critique", "build-on"]:
                rel_type = t
        elif line.startswith("STRENGTH:"):
            try:
                strength = float(line.split(":", 1)[1].strip())
            except ValueError:
                strength = 0.0
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return rel_type, strength, reason


def evaluate_thought_quality(thought: str, problem: str, llm: ChatOpenAI) -> float:
    """Evaluate thought quality"""
    prompt = f"""Rate this thought on a scale of 1-10.

Problem: {problem}
Thought: {thought}

Consider: relevance, insight, clarity, and actionability.

Respond with only a number (1.0-10.0):"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(1.0, min(10.0, score))
    except ValueError:
        return 5.0


def connect_round_thoughts(graph: ThoughtGraph, current_round: int, problem: str, llm: ChatOpenAI) -> int:
    """Connect thoughts from current round to previous thoughts"""

    current_thoughts = graph.get_thoughts_from_round(current_round)
    previous_thoughts = graph.get_recent_thoughts(current_round - 1)

    connections_formed = 0

    for curr_thought in current_thoughts:
        # Each current thought can reference previous thoughts
        for prev_thought in previous_thoughts:
            rel_type, strength, reason = evaluate_relationship(curr_thought, prev_thought, problem, llm)

            if rel_type and strength > 0.3:
                graph.add_connection(curr_thought.id, prev_thought.id, rel_type, strength, reason)
                connections_formed += 1

    return connections_formed


def calculate_convergence_score(graph: ThoughtGraph) -> float:
    """Calculate convergence score (0-1) based on agreement in graph"""

    if graph.graph.number_of_edges() == 0:
        return 0.0

    # Count support edges vs critique edges
    support_edges = 0
    critique_edges = 0
    total_weight = 0

    for _, _, data in graph.graph.edges(data=True):
        weight = data.get("weight", 0)
        if data["type"] == "support":
            support_edges += 1
            total_weight += weight
        elif data["type"] == "critique":
            critique_edges += 1
            total_weight -= weight * 0.5  # Critiques reduce convergence

    if support_edges + critique_edges == 0:
        return 0.0

    # Convergence is high when there are many high-weight support edges
    convergence = (support_edges / (support_edges + critique_edges + 1)) * (
        total_weight / graph.graph.number_of_edges()
    )
    return max(0.0, min(1.0, convergence))


def synthesize_final_answer(graph: ThoughtGraph, problem: str, llm: ChatOpenAI) -> str:
    """Synthesize final answer from the thought graph"""

    # Use PageRank to find most influential thoughts
    if graph.graph.number_of_edges() > 0:
        pagerank = nx.pagerank(graph.graph, weight="weight")
    else:
        pagerank = {nid: 1.0 / len(graph.nodes) for nid in graph.nodes}

    # Get top thoughts
    top_thought_ids = sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True)[:5]

    # Build synthesis prompt
    thought_summaries = []
    for tid in top_thought_ids:
        thought = graph.nodes[tid]
        thought_summaries.append(
            f"- [{thought.perspective.upper()}, Round {thought.round_num}] {thought.content} "
            f"(influence: {pagerank[tid]:.3f})"
        )

    # Include key relationships
    relationships = []
    for source, target, data in graph.graph.edges(data=True):
        if source in top_thought_ids or target in top_thought_ids:
            relationships.append(
                f"  {source} --{data['type']}--> {target} (strength: {data['weight']:.2f})"
            )

    prompt = f"""Synthesize a comprehensive answer by integrating these collaborative perspectives.

Problem: {problem}

Key Thoughts (by influence):
{chr(10).join(thought_summaries)}

Key Relationships:
{chr(10).join(relationships[:10]) if relationships else "  (independent perspectives)"}

Provide a comprehensive answer that:
1. Integrates insights from multiple perspectives
2. Acknowledges where perspectives agree (support) and disagree (critique)
3. Builds on the collaborative refinement across rounds
4. Offers balanced, actionable recommendations

Your synthesis:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def visualize_round(graph: ThoughtGraph, round_num: int) -> None:
    """Visualize thoughts and connections for a specific round"""

    thoughts = graph.get_thoughts_from_round(round_num)

    for thought in thoughts:
        print(f"\n  [{thought.id}] {thought.perspective.upper()}")
        print(f"    {thought.content}")

        # Show outgoing connections (what this thought references)
        edges = list(graph.graph.out_edges(thought.id, data=True))
        if edges:
            print(f"    Connections:")
            for _, target, data in edges:
                target_thought = graph.nodes[target]
                symbol = "✓" if data["type"] == "support" else "✗" if data["type"] == "critique" else "⊕"
                print(
                    f"      {symbol} {data['type']} → {target} "
                    f"[{target_thought.perspective}, R{target_thought.round_num}] "
                    f"(weight: {data['weight']:.2f})"
                )


def visualize_graph_summary(graph: ThoughtGraph) -> None:
    """Visualize overall graph structure"""

    print_section("Graph Structure Summary")

    print("\n  Nodes by Round:")
    for round_num in range(1, graph.current_round + 1):
        thoughts = graph.get_thoughts_from_round(round_num)
        print(f"    Round {round_num}: {len(thoughts)} thoughts")

    print("\n  Edge Types:")
    edge_types = {}
    for _, _, data in graph.graph.edges(data=True):
        et = data["type"]
        edge_types[et] = edge_types.get(et, 0) + 1

    for et, count in sorted(edge_types.items()):
        print(f"    {et}: {count}")

    print(f"\n  Graph Metrics:")
    print(f"    Total nodes: {graph.graph.number_of_nodes()}")
    print(f"    Total edges: {graph.graph.number_of_edges()}")
    print(f"    Density: {nx.density(graph.graph):.2%}")

    if graph.graph.number_of_edges() > 0:
        # PageRank analysis
        pagerank = nx.pagerank(graph.graph, weight="weight")
        top_3 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n  Most Influential Thoughts (PageRank):")
        for nid, score in top_3:
            thought = graph.nodes[nid]
            print(f"    {nid} [{thought.perspective}, R{thought.round_num}]: {score:.3f}")


def run_advanced_got(problem: str, max_rounds: int = 3) -> str:
    """
    Run advanced Graph of Thoughts with multi-round consensus building

    Args:
        problem: Problem to solve
        max_rounds: Maximum number of refinement rounds

    Returns:
        Final synthesized answer
    """

    print_header("Graph of Thoughts - Advanced Implementation")
    print(f"\nProblem: {problem}\n")
    print(f"Configuration: {max_rounds} rounds, 4 agent personas\n")

    # Initialize
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    graph = ThoughtGraph()
    agent_personas = ["optimist", "critic", "pragmatist", "innovator"]

    # Round 1: Initial proposals
    print_round_header(1, "Initial Proposals")

    for persona in agent_personas:
        print(f"\n  Generating {persona}'s perspective...")
        content = generate_initial_thought(problem, persona, llm)

        thought = ThoughtNode(id=f"R1_{persona}", content=content, perspective=persona, round_num=1)

        graph.add_thought(thought)

        print(f"\n  {persona.upper()}:")
        print(f"    {content}")

    graph.current_round = 1

    # Subsequent rounds: Refinement and consensus building
    for round_num in range(2, max_rounds + 1):
        print_round_header(round_num, "Refinement and Consensus")

        # Generate responses
        for persona in agent_personas:
            # Get context (exclude own previous thoughts to avoid echo chamber)
            context = graph.get_recent_thoughts(round_num - 1, exclude_perspective=persona)

            print(f"\n  {persona} responding to previous thoughts...")
            content = generate_response_thought(problem, persona, context, round_num, llm)

            thought = ThoughtNode(
                id=f"R{round_num}_{persona}", content=content, perspective=persona, round_num=round_num
            )

            graph.add_thought(thought)

            print(f"\n  {persona.upper()}:")
            print(f"    {content}")

        # Form connections for this round
        print(f"\n  Forming connections for Round {round_num}...")
        connections = connect_round_thoughts(graph, round_num, problem, llm)
        print(f"  ✓ {connections} connections formed")

        # Show connections for this round
        visualize_round(graph, round_num)

        # Check convergence
        convergence = calculate_convergence_score(graph)
        print(f"\n  Convergence score: {convergence:.2%}")

        if convergence > 0.75 and round_num >= 2:
            print(f"  ✓ High convergence reached, stopping early")
            break

        graph.current_round = round_num

    # Graph analysis
    visualize_graph_summary(graph)

    # Synthesize final answer
    print_section("Final Synthesis")
    print("\n  Integrating all perspectives across rounds...\n")

    final_answer = synthesize_final_answer(graph, problem, llm)

    print_header("Final Answer")
    print(f"\n{final_answer}\n")

    # Statistics
    print_section("Session Statistics")
    print(f"  Rounds completed: {graph.current_round}")
    print(f"  Total thoughts generated: {len(graph.nodes)}")
    print(f"  Total connections formed: {graph.graph.number_of_edges()}")
    print(f"  Final convergence: {calculate_convergence_score(graph):.2%}")

    return final_answer


def main():
    """Main execution"""

    # Example problem: Complex decision requiring consensus
    problem = (
        "Should our company adopt a 4-day work week? Consider impact on "
        "productivity, employee satisfaction, costs, and competitive advantage."
    )

    # Alternative problems:
    # problem = "Should we prioritize AI safety regulations or AI innovation and competition?"
    # problem = "Is universal basic income a viable solution to automation-driven unemployment?"
    # problem = "Should gene editing be allowed for human enhancement or only disease prevention?"

    try:
        result = run_advanced_got(problem, max_rounds=3)

        print("\n" + "=" * 80)
        print("  Advanced Graph of Thoughts Completed Successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
