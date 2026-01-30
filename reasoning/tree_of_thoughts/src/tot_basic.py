"""
Tree of Thoughts Pattern: Basic Implementation with BFS
This example demonstrates the Tree of Thoughts pattern using Breadth-First Search
to solve the Game of 24 puzzle by exploring multiple reasoning paths.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from project root (note: different path!)
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")


# --- Data Structures ---

@dataclass
class ThoughtNode:
    """Represents a node in the thought tree"""
    content: str  # The thought/reasoning step
    state: str  # Current state of the problem
    score: float  # Evaluation score (0-10)
    depth: int  # Distance from root
    parent: Optional['ThoughtNode'] = None  # Previous step
    children: list['ThoughtNode'] = None  # Next steps
    is_solution: bool = False  # Is this a final solution?

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def path_from_root(self) -> list['ThoughtNode']:
        """Reconstruct the path from root to this node"""
        path = []
        node = self
        while node is not None:
            path.insert(0, node)
            node = node.parent
        return path

    def __repr__(self) -> str:
        return f"ThoughtNode(content='{self.content[:30]}...', score={self.score:.1f}, depth={self.depth})"


# --- Core Tree of Thoughts Functions ---

def generate_thoughts(current_state: str, problem: str, num_thoughts: int = 3) -> list[str]:
    """
    Generate multiple possible next reasoning steps from current state.

    This is the "branching" phase where we create alternative paths to explore.
    """
    prompt = f"""You are solving the Game of 24 puzzle.

Problem: {problem}
Current state: {current_state}

Generate {num_thoughts} different next steps to progress toward the solution.
Each step should:
1. Try a different mathematical operation
2. Be concrete and actionable
3. Show the calculation

Format each thought as: "Try [operation]: [calculation] = [result], leaving [remaining numbers]"

Example: "Try multiplication: 4 * 6 = 24, leaving [8, 2]"

Provide exactly {num_thoughts} different approaches:"""

    response = llm.invoke(prompt)
    thoughts = response.content.strip().split('\n')

    # Clean and filter thoughts
    thoughts = [t.strip() for t in thoughts if t.strip()]
    thoughts = [t.lstrip('0123456789.-) ') for t in thoughts]  # Remove numbering

    return thoughts[:num_thoughts]


def evaluate_thought(thought: str, goal: str, current_depth: int, max_depth: int) -> float:
    """
    Evaluate how promising a thought is toward solving the problem.

    Returns a score from 0-10 where:
    - 10 = Definitely solves the problem
    - 7-9 = Very promising direction
    - 5-6 = Moderately promising
    - 3-4 = Questionable direction
    - 0-2 = Poor direction
    """
    prompt = f"""Evaluate this reasoning step for solving the Game of 24 puzzle.

Goal: {goal}
Current step: {thought}
Progress: Depth {current_depth}/{max_depth}

Rate this step from 0-10 based on:
1. Does it move closer to making 24?
2. Are the remaining numbers easy to work with?
3. Is this a logical mathematical approach?

If the step reaches 24, give it a 10.
If the step gets very close (like 20-28), give it 8-9.
If the step makes progress but not close, give it 5-7.
If the step seems unhelpful, give it 0-4.

Provide ONLY a number from 0-10, nothing else."""

    try:
        response = llm.invoke(prompt)
        score_text = response.content.strip()

        # Extract number from response
        import re
        match = re.search(r'\d+\.?\d*', score_text)
        if match:
            score = float(match.group())
            return min(10.0, max(0.0, score))
        else:
            return 5.0  # Default if can't parse
    except Exception as e:
        print(f"Error evaluating thought: {e}")
        return 5.0


def check_solution(thought: str, target: int = 24) -> bool:
    """
    Check if a thought represents a solution to the problem.
    """
    # Look for the target number (24) appearing in the thought
    import re

    # Check for patterns like "= 24" or "equals 24" or "result: 24"
    patterns = [
        rf'=\s*{target}(?:\s|,|\.|\)|\]|$)',
        rf'equals\s+{target}',
        rf'result[:\s]+{target}',
        rf'answer[:\s]+{target}',
    ]

    for pattern in patterns:
        if re.search(pattern, thought, re.IGNORECASE):
            return True

    return False


# --- BFS Tree of Thoughts Implementation ---

def tree_of_thoughts_bfs(
    problem: str,
    max_depth: int = 4,
    beam_width: int = 5,
    branching_factor: int = 3,
    score_threshold: float = 5.0
) -> tuple[Optional[ThoughtNode], list[ThoughtNode]]:
    """
    Tree of Thoughts using Breadth-First Search.

    Args:
        problem: The problem to solve
        max_depth: Maximum tree depth to explore
        beam_width: Number of best nodes to keep at each level
        branching_factor: Number of thoughts to generate per node
        score_threshold: Minimum score to keep exploring a path

    Returns:
        Tuple of (solution_node, all_explored_nodes)
    """
    print(f"\n{'='*80}")
    print("STARTING TREE OF THOUGHTS (BFS)")
    print(f"{'='*80}")
    print(f"Problem: {problem}")
    print(f"Max Depth: {max_depth}, Beam Width: {beam_width}, Branching: {branching_factor}")
    print(f"Score Threshold: {score_threshold}")
    print(f"{'='*80}\n")

    # Initialize root node
    root = ThoughtNode(
        content=f"Initial problem: {problem}",
        state=problem,
        score=10.0,
        depth=0
    )

    # Track all nodes for visualization
    all_nodes = [root]

    # BFS queue starts with root
    current_level = [root]
    solution = None

    # Explore level by level
    for depth in range(1, max_depth + 1):
        print(f"\n{'─'*80}")
        print(f"LEVEL {depth}/{max_depth} - Exploring {len(current_level)} nodes")
        print(f"{'─'*80}")

        next_level = []

        # Expand each node in current level
        for node_idx, node in enumerate(current_level):
            print(f"\n  Node {node_idx + 1}/{len(current_level)} (Score: {node.score:.1f}):")
            print(f"  State: {node.state}")

            # Generate thoughts from this node
            print(f"  Generating {branching_factor} thoughts...")
            thoughts = generate_thoughts(node.state, problem, branching_factor)

            # Evaluate each thought
            for thought_idx, thought in enumerate(thoughts):
                # Evaluate the thought
                score = evaluate_thought(thought, problem, depth, max_depth)

                print(f"    └─ Thought {thought_idx + 1}: {thought[:60]}... (Score: {score:.1f})")

                # Check if below threshold
                if score < score_threshold:
                    print(f"       ✗ Pruned (below threshold {score_threshold})")
                    continue

                # Create child node
                child = ThoughtNode(
                    content=thought,
                    state=thought,  # Simplified: thought becomes new state
                    score=score,
                    depth=depth,
                    parent=node
                )

                node.children.append(child)
                all_nodes.append(child)

                # Check if this is a solution
                if check_solution(thought, target=24):
                    print(f"       ✓✓ SOLUTION FOUND!")
                    child.is_solution = True
                    solution = child
                    return solution, all_nodes

                # Add to next level
                next_level.append(child)

        # No candidates for next level? Dead end
        if not next_level:
            print(f"\n  ⚠️  No viable paths at depth {depth}. Stopping search.")
            break

        # Prune: Keep only top beam_width nodes for next iteration
        next_level.sort(key=lambda x: x.score, reverse=True)
        pruned_count = len(next_level) - beam_width
        current_level = next_level[:beam_width]

        if pruned_count > 0:
            print(f"\n  Pruning: Keeping top {beam_width} nodes, pruned {pruned_count} nodes")

    # No solution found, return best node
    if not solution and current_level:
        print(f"\n{'='*80}")
        print("No complete solution found. Returning best path.")
        print(f"{'='*80}")
        solution = max(current_level, key=lambda x: x.score)

    return solution, all_nodes


# --- Visualization Functions ---

def visualize_tree(root: ThoughtNode, max_content_length: int = 50):
    """
    Display the thought tree in ASCII format.
    """
    print(f"\n{'='*80}")
    print("THOUGHT TREE VISUALIZATION")
    print(f"{'='*80}\n")

    def print_node(node: ThoughtNode, prefix: str = "", is_last: bool = True):
        """Recursively print tree structure"""
        # Connector symbols
        connector = "└─" if is_last else "├─"
        extension = "  " if is_last else "│ "

        # Format node content
        content = node.content[:max_content_length]
        if len(node.content) > max_content_length:
            content += "..."

        # Node symbol
        symbol = "✓✓" if node.is_solution else "✓" if node.score >= 7 else "○"

        # Print this node
        print(f"{prefix}{connector} {symbol} {content} (Score: {node.score:.1f})")

        # Print children
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            print_node(child, prefix + extension, is_last_child)

    print_node(root)
    print()


def display_solution_path(solution_node: ThoughtNode):
    """
    Display the path from root to solution.
    """
    path = solution_node.path_from_root()

    print(f"\n{'='*80}")
    print("SOLUTION PATH")
    print(f"{'='*80}\n")

    for i, node in enumerate(path):
        if i == 0:
            print(f"START: {node.content}")
        elif i == len(path) - 1:
            print(f"  ↓")
            print(f"SOLUTION: {node.content} ✓")
            print(f"Final Score: {node.score:.1f}/10")
        else:
            print(f"  ↓")
            print(f"Step {i}: {node.content}")

    print(f"\n{'='*80}\n")


def display_statistics(all_nodes: list[ThoughtNode], solution: Optional[ThoughtNode]):
    """
    Display search statistics.
    """
    print(f"\n{'='*80}")
    print("SEARCH STATISTICS")
    print(f"{'='*80}")

    total_nodes = len(all_nodes)
    max_depth = max(node.depth for node in all_nodes)
    avg_score = sum(node.score for node in all_nodes) / total_nodes if total_nodes > 0 else 0

    nodes_by_depth = {}
    for node in all_nodes:
        nodes_by_depth[node.depth] = nodes_by_depth.get(node.depth, 0) + 1

    print(f"\nTotal nodes explored: {total_nodes}")
    print(f"Maximum depth reached: {max_depth}")
    print(f"Average node score: {avg_score:.2f}/10")
    print(f"\nNodes by depth:")
    for depth in sorted(nodes_by_depth.keys()):
        print(f"  Level {depth}: {nodes_by_depth[depth]} nodes")

    if solution:
        path_length = solution.depth
        print(f"\nSolution found: {'Yes' if solution.is_solution else 'Best effort'}")
        print(f"Solution path length: {path_length} steps")
        print(f"Solution score: {solution.score:.1f}/10")
    else:
        print(f"\nNo solution found")

    print(f"{'='*80}\n")


# --- Example Usage ---

def run_game_of_24_example(numbers: list[int]):
    """
    Run the Game of 24 example with given numbers.
    """
    problem = f"Use the numbers {numbers} with operations (+, -, *, /) to make 24"

    print(f"""
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                    Tree of Thoughts: Game of 24                                ║
    ║                                                                                ║
    ║  The agent will explore multiple reasoning paths to solve the puzzle          ║
    ║  Numbers: {str(numbers):^64} ║
    ║  Target: 24                                                                    ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run Tree of Thoughts BFS
    solution, all_nodes = tree_of_thoughts_bfs(
        problem=problem,
        max_depth=4,
        beam_width=5,
        branching_factor=3,
        score_threshold=5.0
    )

    # Visualize the tree
    if all_nodes:
        visualize_tree(all_nodes[0])  # Root node

    # Display solution path
    if solution:
        display_solution_path(solution)

    # Display statistics
    display_statistics(all_nodes, solution)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║              Tree of Thoughts Pattern - Basic Implementation                   ║
    ║                                                                                ║
    ║  Using Breadth-First Search to explore multiple reasoning paths               ║
    ║  Problem: Game of 24 Puzzle                                                   ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Classic Game of 24
    print("\n" + "="*80)
    print("EXAMPLE 1: Classic Numbers")
    print("="*80)
    run_game_of_24_example([4, 6, 8, 2])

    # Example 2: Harder puzzle
    print("\n" + "="*80)
    print("EXAMPLE 2: More Challenging")
    print("="*80)
    run_game_of_24_example([3, 3, 8, 8])

    print("""
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                          Examples Complete!                                    ║
    ║                                                                                ║
    ║  The Basic Tree of Thoughts implementation demonstrated:                      ║
    ║  • Multiple reasoning path exploration (branching)                            ║
    ║  • Breadth-First Search strategy                                              ║
    ║  • Thought evaluation and scoring                                             ║
    ║  • Beam pruning to control costs                                              ║
    ║  • Solution path reconstruction                                               ║
    ║  • Tree visualization                                                         ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """)
