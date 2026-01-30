"""
Tree of Thoughts Pattern: Advanced Implementation with Beam Search
This example demonstrates an advanced Tree of Thoughts pattern using Beam Search
with aggressive pruning, backtracking, and path optimization for creative writing tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0.8, model="gpt-4o-mini")


# --- Advanced Data Structures ---

@dataclass
class AdvancedThoughtNode:
    """Enhanced node with additional metadata for advanced search"""
    content: str  # The thought/reasoning step
    state: dict  # Rich state representation
    score: float  # Evaluation score (0-10)
    depth: int  # Distance from root
    parent: Optional['AdvancedThoughtNode'] = None
    children: list['AdvancedThoughtNode'] = field(default_factory=list)
    is_solution: bool = False
    is_dead_end: bool = False  # Marked as dead end
    visit_count: int = 0  # Times this node was considered
    metadata: dict = field(default_factory=dict)  # Additional info

    def path_from_root(self) -> list['AdvancedThoughtNode']:
        """Reconstruct path from root to this node"""
        path = []
        node = self
        while node is not None:
            path.insert(0, node)
            node = node.parent
        return path

    def get_path_scores(self) -> list[float]:
        """Get all scores along the path"""
        return [node.score for node in self.path_from_root()]

    def average_path_score(self) -> float:
        """Calculate average score along the path"""
        scores = self.get_path_scores()
        return sum(scores) / len(scores) if scores else 0.0

    def __repr__(self) -> str:
        status = "SOLUTION" if self.is_solution else "DEAD_END" if self.is_dead_end else "ACTIVE"
        return f"Node(depth={self.depth}, score={self.score:.1f}, status={status})"


# --- Advanced Thought Generation ---

def generate_creative_thoughts(
    current_state: dict,
    problem: str,
    num_thoughts: int = 5
) -> list[dict]:
    """
    Generate diverse creative thoughts for story writing.

    Returns list of dicts with 'content' and 'reasoning' keys.
    """
    context = current_state.get('story_so_far', 'Starting a new story')

    prompt = f"""You are writing a creative sci-fi mystery story.

Task: {problem}
Story so far: {context}

Generate {num_thoughts} DIFFERENT next directions for the story.
Each direction should:
1. Be creative and engaging
2. Explore a different narrative path
3. Maintain mystery and intrigue
4. Be distinct from the others

For each direction, provide:
- The next story content (2-3 sentences)
- Brief reasoning for why this direction is interesting

Format:
DIRECTION 1:
Content: [story content here]
Reasoning: [why this works]

DIRECTION 2:
Content: [story content here]
Reasoning: [why this works]

...and so on for {num_thoughts} directions."""

    response = llm.invoke(prompt)
    text = response.content.strip()

    # Parse the response
    thoughts = []
    directions = text.split('DIRECTION ')[1:]  # Split by direction markers

    for direction in directions:
        lines = direction.strip().split('\n')
        content = ""
        reasoning = ""

        for line in lines:
            if line.startswith('Content:'):
                content = line.replace('Content:', '').strip()
            elif line.startswith('Reasoning:'):
                reasoning = line.replace('Reasoning:', '').strip()

        if content:
            thoughts.append({
                'content': content,
                'reasoning': reasoning
            })

    return thoughts[:num_thoughts]


def evaluate_creative_thought(
    thought: dict,
    goal: str,
    current_depth: int,
    max_depth: int,
    parent_score: float = 8.0
) -> dict:
    """
    Evaluate a creative thought across multiple criteria.

    Returns dict with overall score and breakdown.
    """
    content = thought['content']
    reasoning = thought.get('reasoning', '')

    prompt = f"""Evaluate this creative story direction:

Goal: {goal}
Story direction: {content}
Author's reasoning: {reasoning}
Progress: Step {current_depth}/{max_depth}
Parent quality: {parent_score:.1f}/10

Rate this direction on these criteria (0-10 for each):

1. ENGAGEMENT: How compelling and interesting is this direction?
2. CREATIVITY: How original and unexpected is this?
3. COHERENCE: How well does it fit and make sense?
4. POTENTIAL: How much story potential does this create?
5. MYSTERY: Does it maintain/enhance the mystery element?

Provide scores in this exact format:
ENGAGEMENT: [0-10]
CREATIVITY: [0-10]
COHERENCE: [0-10]
POTENTIAL: [0-10]
MYSTERY: [0-10]
OVERALL: [0-10]"""

    try:
        response = llm.invoke(prompt)
        scores_text = response.content.strip()

        # Parse scores
        import re
        criteria = ['ENGAGEMENT', 'CREATIVITY', 'COHERENCE', 'POTENTIAL', 'MYSTERY', 'OVERALL']
        scores = {}

        for criterion in criteria:
            pattern = rf'{criterion}:\s*(\d+\.?\d*)'
            match = re.search(pattern, scores_text, re.IGNORECASE)
            if match:
                scores[criterion.lower()] = float(match.group(1))
            else:
                scores[criterion.lower()] = 6.0  # Default

        return {
            'overall': scores.get('overall', 6.0),
            'breakdown': scores
        }

    except Exception as e:
        print(f"Error evaluating thought: {e}")
        return {
            'overall': 6.0,
            'breakdown': {}
        }


def check_creative_solution(node: AdvancedThoughtNode, min_depth: int = 3) -> bool:
    """
    Check if we have a complete story (reached sufficient depth with high quality).
    """
    if node.depth < min_depth:
        return False

    # Check if score is high enough
    if node.score >= 8.5:
        return True

    # Check if average path score is high
    if node.average_path_score() >= 8.0 and node.depth >= min_depth:
        return True

    return False


# --- Beam Search with Advanced Features ---

def tree_of_thoughts_beam_search(
    problem: str,
    max_depth: int = 5,
    beam_width: int = 10,
    branching_factor: int = 5,
    score_threshold: float = 6.5,
    enable_backtracking: bool = True
) -> tuple[Optional[AdvancedThoughtNode], list[AdvancedThoughtNode]]:
    """
    Advanced Tree of Thoughts using Beam Search with:
    - Aggressive pruning
    - Backtracking when stuck
    - Dynamic score thresholds
    - Path quality optimization

    Args:
        problem: The creative task to solve
        max_depth: Maximum exploration depth
        beam_width: Number of best paths to maintain
        branching_factor: Thoughts to generate per node
        score_threshold: Minimum score to continue exploration
        enable_backtracking: Allow backtracking on dead ends

    Returns:
        Tuple of (best_solution, all_explored_nodes)
    """
    print(f"\n{'='*100}")
    print("ADVANCED TREE OF THOUGHTS WITH BEAM SEARCH")
    print(f"{'='*100}")
    print(f"Problem: {problem}")
    print(f"Config: Depth={max_depth}, Beam={beam_width}, Branch={branching_factor}, Threshold={score_threshold}")
    print(f"Backtracking: {'Enabled' if enable_backtracking else 'Disabled'}")
    print(f"{'='*100}\n")

    # Initialize root
    root = AdvancedThoughtNode(
        content="Starting creative story development",
        state={'story_so_far': '', 'step': 0},
        score=10.0,
        depth=0,
        metadata={'is_root': True}
    )

    # Tracking
    all_nodes = [root]
    beam = [root]
    best_solution = None
    backtrack_history = []

    # Beam search iteration
    for depth in range(1, max_depth + 1):
        print(f"\n{'─'*100}")
        print(f"DEPTH {depth}/{max_depth} | Beam size: {len(beam)} | Total explored: {len(all_nodes)}")
        print(f"{'─'*100}")

        candidates = []

        # Expand each node in beam
        for beam_idx, node in enumerate(beam):
            node.visit_count += 1

            print(f"\n  [{beam_idx + 1}/{len(beam)}] Expanding node (Score: {node.score:.1f}, Avg path: {node.average_path_score():.1f})")
            print(f"  Content: {node.content[:80]}...")

            # Build current state
            current_state = {
                'story_so_far': node.content,
                'step': depth
            }

            # Generate thoughts
            print(f"  Generating {branching_factor} creative directions...")
            thoughts = generate_creative_thoughts(current_state, problem, branching_factor)

            if not thoughts:
                print(f"  ⚠️  No thoughts generated. Marking as dead end.")
                node.is_dead_end = True
                continue

            # Evaluate each thought
            for thought_idx, thought in enumerate(thoughts):
                # Evaluate
                evaluation = evaluate_creative_thought(
                    thought,
                    problem,
                    depth,
                    max_depth,
                    node.score
                )

                score = evaluation['overall']
                breakdown = evaluation['breakdown']

                print(f"\n    Direction {thought_idx + 1}:")
                print(f"    Content: {thought['content'][:70]}...")
                print(f"    Score: {score:.1f}/10 (Engagement:{breakdown.get('engagement', 0):.1f}, "
                      f"Creativity:{breakdown.get('creativity', 0):.1f}, "
                      f"Mystery:{breakdown.get('mystery', 0):.1f})")

                # Check threshold
                if score < score_threshold:
                    print(f"    ✗ Pruned (below threshold {score_threshold})")
                    continue

                # Create child node
                child = AdvancedThoughtNode(
                    content=thought['content'],
                    state=current_state,
                    score=score,
                    depth=depth,
                    parent=node,
                    metadata={
                        'reasoning': thought.get('reasoning', ''),
                        'score_breakdown': breakdown
                    }
                )

                node.children.append(child)
                all_nodes.append(child)

                # Check if solution
                if check_creative_solution(child, min_depth=3):
                    print(f"    ✓✓ High-quality solution found!")
                    child.is_solution = True
                    if not best_solution or child.average_path_score() > best_solution.average_path_score():
                        best_solution = child

                candidates.append(child)

        # Check if we have candidates
        if not candidates:
            print(f"\n  ⚠️  No viable candidates at depth {depth}.")

            if enable_backtracking and beam:
                print(f"  🔄 Attempting backtracking...")
                # Try lower threshold
                score_threshold = max(4.0, score_threshold - 1.0)
                print(f"  Lowered threshold to {score_threshold}")
                backtrack_history.append(depth)
                continue
            else:
                print(f"  Stopping search (no candidates, backtracking disabled).")
                break

        # Sort candidates by combined score (node score + path average)
        def combined_score(node):
            return 0.6 * node.score + 0.4 * node.average_path_score()

        candidates.sort(key=combined_score, reverse=True)

        # Show top candidates
        print(f"\n  Top candidates:")
        for i, cand in enumerate(candidates[:5]):
            print(f"    {i+1}. Score: {cand.score:.1f} (Path avg: {cand.average_path_score():.1f})")

        # Update beam (keep top beam_width)
        pruned_count = len(candidates) - beam_width
        beam = candidates[:beam_width]

        if pruned_count > 0:
            print(f"\n  Pruned {pruned_count} candidates, keeping top {len(beam)} in beam")

        # Adaptive threshold: raise if we have many good candidates
        if len([c for c in candidates if c.score >= 8.0]) > beam_width:
            score_threshold = min(8.0, score_threshold + 0.5)
            print(f"  📈 Raised threshold to {score_threshold} (many high-quality candidates)")

    # If no explicit solution, return best node
    if not best_solution and beam:
        best_solution = max(beam, key=lambda n: n.average_path_score())
        print(f"\n{'='*100}")
        print("No explicit solution found. Selecting best path based on average score.")
        print(f"{'='*100}")

    return best_solution, all_nodes


# --- Visualization Functions ---

def visualize_decision_tree(root: AdvancedThoughtNode, max_depth: int = 3, max_width: int = 3):
    """
    Visualize the decision tree with detailed information.
    """
    print(f"\n{'='*100}")
    print("DECISION TREE VISUALIZATION")
    print(f"{'='*100}\n")

    def print_node(node: AdvancedThoughtNode, prefix: str = "", is_last: bool = True, current_depth: int = 0):
        """Recursively print tree with depth limit"""
        if current_depth > max_depth:
            return

        # Connector
        connector = "└─" if is_last else "├─"
        extension = "  " if is_last else "│ "

        # Node symbol
        if node.is_solution:
            symbol = "✓✓"
        elif node.is_dead_end:
            symbol = "✗✗"
        elif node.score >= 8.5:
            symbol = "⭐"
        elif node.score >= 7.0:
            symbol = "✓"
        else:
            symbol = "○"

        # Content preview
        content = node.content[:60]
        if len(node.content) > 60:
            content += "..."

        # Print node
        print(f"{prefix}{connector} {symbol} [{node.score:.1f}|{node.average_path_score():.1f}] {content}")

        # Print children (limited)
        children_to_show = sorted(node.children, key=lambda c: c.score, reverse=True)[:max_width]
        for i, child in enumerate(children_to_show):
            is_last_child = (i == len(children_to_show) - 1)
            print_node(child, prefix + extension, is_last_child, current_depth + 1)

        # Show if there are more children
        if len(node.children) > max_width:
            print(f"{prefix}{extension}  ... and {len(node.children) - max_width} more")

    print_node(root)
    print(f"\nLegend: [node_score|path_average] ✓✓=Solution ⭐=Excellent ✓=Good ○=Okay ✗✗=Dead End")
    print()


def display_solution_narrative(solution: AdvancedThoughtNode):
    """
    Display the complete story narrative from root to solution.
    """
    path = solution.path_from_root()

    print(f"\n{'='*100}")
    print("COMPLETE STORY NARRATIVE")
    print(f"{'='*100}\n")

    story_parts = []

    for i, node in enumerate(path):
        if i == 0:
            continue  # Skip root

        print(f"{'─'*100}")
        print(f"STEP {i} (Score: {node.score:.1f}/10, Path Average: {node.average_path_score():.1f})")
        print(f"{'─'*100}")
        print(f"{node.content}\n")

        if node.metadata.get('reasoning'):
            print(f"💭 Reasoning: {node.metadata['reasoning']}\n")

        story_parts.append(node.content)

    print(f"{'='*100}")
    print("FINAL STORY")
    print(f"{'='*100}\n")
    print(' '.join(story_parts))
    print(f"\n{'='*100}")
    print(f"Story Quality: {solution.average_path_score():.1f}/10")
    print(f"Total Steps: {solution.depth}")
    print(f"{'='*100}\n")


def display_advanced_statistics(all_nodes: list[AdvancedThoughtNode], solution: Optional[AdvancedThoughtNode]):
    """
    Display comprehensive search statistics.
    """
    print(f"\n{'='*100}")
    print("ADVANCED SEARCH STATISTICS")
    print(f"{'='*100}")

    total_nodes = len(all_nodes)
    max_depth = max(node.depth for node in all_nodes) if all_nodes else 0
    avg_score = sum(node.score for node in all_nodes) / total_nodes if total_nodes > 0 else 0

    solution_nodes = [n for n in all_nodes if n.is_solution]
    dead_end_nodes = [n for n in all_nodes if n.is_dead_end]
    high_quality_nodes = [n for n in all_nodes if n.score >= 8.0]

    nodes_by_depth = {}
    for node in all_nodes:
        nodes_by_depth[node.depth] = nodes_by_depth.get(node.depth, 0) + 1

    print(f"\n📊 Exploration Metrics:")
    print(f"  Total nodes explored: {total_nodes}")
    print(f"  Maximum depth reached: {max_depth}")
    print(f"  Average node score: {avg_score:.2f}/10")
    print(f"  High-quality nodes (≥8.0): {len(high_quality_nodes)} ({len(high_quality_nodes)/total_nodes*100:.1f}%)")
    print(f"  Solution nodes: {len(solution_nodes)}")
    print(f"  Dead-end nodes: {len(dead_end_nodes)}")

    print(f"\n📈 Nodes by depth:")
    for depth in sorted(nodes_by_depth.keys()):
        count = nodes_by_depth[depth]
        bar = "█" * min(50, count * 2)
        print(f"  Level {depth}: {count:3d} {bar}")

    if solution:
        path = solution.path_from_root()
        path_scores = [n.score for n in path[1:]]  # Exclude root

        print(f"\n🎯 Solution Quality:")
        print(f"  Solution found: Yes")
        print(f"  Path length: {solution.depth} steps")
        print(f"  Final score: {solution.score:.1f}/10")
        print(f"  Average path score: {solution.average_path_score():.1f}/10")
        print(f"  Min score in path: {min(path_scores):.1f}/10")
        print(f"  Max score in path: {max(path_scores):.1f}/10")

    print(f"{'='*100}\n")


# --- Example Usage ---

def run_creative_writing_example():
    """
    Run creative writing example with advanced ToT.
    """
    problem = "Write an engaging opening for a sci-fi mystery story about an AI detective investigating an impossible crime"

    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                    Tree of Thoughts: Creative Story Writing                                  ║
    ║                                                                                              ║
    ║  The agent will explore multiple narrative paths using beam search                          ║
    ║  Task: Write compelling sci-fi mystery opening                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run advanced beam search
    solution, all_nodes = tree_of_thoughts_beam_search(
        problem=problem,
        max_depth=4,
        beam_width=8,
        branching_factor=4,
        score_threshold=6.5,
        enable_backtracking=True
    )

    # Visualize decision tree
    if all_nodes:
        visualize_decision_tree(all_nodes[0], max_depth=3, max_width=3)

    # Display solution narrative
    if solution:
        display_solution_narrative(solution)

    # Display statistics
    display_advanced_statistics(all_nodes, solution)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║              Tree of Thoughts Pattern - Advanced Implementation                             ║
    ║                                                                                              ║
    ║  Using Beam Search with pruning, backtracking, and quality optimization                     ║
    ║  Problem: Creative Story Writing                                                            ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    run_creative_writing_example()

    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                          Example Complete!                                                   ║
    ║                                                                                              ║
    ║  The Advanced Tree of Thoughts implementation demonstrated:                                 ║
    ║  • Beam search with dynamic width                                                           ║
    ║  • Multi-criteria evaluation (engagement, creativity, coherence, mystery)                   ║
    ║  • Aggressive pruning with adaptive thresholds                                              ║
    ║  • Backtracking when stuck                                                                  ║
    ║  • Path quality optimization (combining node and path scores)                               ║
    ║  • Rich visualization with decision trees                                                   ║
    ║  • Comprehensive statistics and metrics                                                     ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
