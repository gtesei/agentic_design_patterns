# Tree of Thoughts (ToT)

## Overview

The **Tree of Thoughts (ToT) Pattern** is an advanced reasoning framework that extends beyond linear chain-of-thought prompting by enabling LLMs to explore multiple reasoning paths simultaneously. Instead of following a single chain of reasoning, ToT constructs a tree where each node represents a partial solution or "thought," and the agent systematically explores, evaluates, and prunes branches to find the optimal solution.

ToT transforms LLMs from linear reasoners into strategic problem-solvers that can backtrack, compare alternatives, and make informed decisions about which reasoning paths to pursue—much like how humans approach complex problems that require deliberation and exploration.

## Why Use This Pattern?

Traditional reasoning approaches have significant limitations:

- **Chain-of-Thought (CoT)**: Follows a single linear path, can't explore alternatives or backtrack when stuck
- **Direct prompting**: Limited reasoning depth, prone to getting stuck in local optima
- **ReAct**: Sequential exploration without systematic comparison of alternative paths
- **Greedy search**: Commits to decisions without considering multiple options

Tree of Thoughts solves these by:
- **Multi-path exploration**: Generates and evaluates multiple reasoning paths simultaneously
- **Systematic evaluation**: Scores each thought to identify most promising directions
- **Strategic backtracking**: Returns to earlier states when current path proves unproductive
- **Deliberate decision-making**: Compares alternatives before committing to a path
- **Global optimization**: Searches for best overall solution rather than first acceptable one

### Example: Solving the Game of 24

```
Without ToT (Linear):
Prompt: "Use 4, 6, 8, 2 to make 24"
Response: "Let me try: 4 + 6 = 10, 10 + 8 = 18, 18 + 2 = 20... doesn't work"
→ Gets stuck, can't try alternative approaches

With ToT (Exploration):
Root: [4, 6, 8, 2] → Make 24

Branch 1: "Combine 4 and 6"
  ├─ 4 + 6 = 10 → [10, 8, 2] (Score: 6/10 - possible)
  ├─ 4 * 6 = 24 → [24, 8, 2] (Score: 9/10 - very promising!)
  └─ 6 - 4 = 2  → [2, 8, 2] (Score: 3/10 - limited options)

Branch 2: "Combine 8 and 2"
  ├─ 8 * 2 = 16 → [4, 6, 16] (Score: 7/10 - good potential)
  ├─ 8 + 2 = 10 → [4, 6, 10] (Score: 6/10 - possible)
  └─ 8 - 2 = 6  → [4, 6, 6] (Score: 4/10 - limited)

Select highest scored path: 4 * 6 = 24
Verification: 24 is already the target!
Final solution: 4 * 6 = 24 (unused: 8, 2)
Alternative: (4 * 6) * (8 / 2) / (8 / 2) = 24
```

## How It Works

The Tree of Thoughts process consists of five key phases:

```
┌─────────────────────────────────────────────────────────────┐
│                    Initial Problem                           │
│                    (Root Node)                               │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
            ┌───────────────┐
            │  1. GENERATE  │  Create multiple thought branches
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │  2. EVALUATE  │  Score each thought (1-10)
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │  3. EXPAND    │  Develop promising branches
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   4. PRUNE    │  Discard low-scoring branches
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │  5. ITERATE   │  Repeat until solution found
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ Final Solution │
            └───────────────┘
```

### Phase 1: Generate Thoughts

From each node, generate multiple possible next steps:
- **Breadth**: Create 2-5 alternative thoughts per node
- **Diversity**: Ensure thoughts explore different approaches
- **Coherence**: Each thought must be a logical progression

```python
def generate_thoughts(current_state: str, num_thoughts: int = 3) -> list[str]:
    """Generate multiple possible next reasoning steps"""
    prompt = f"""Given current state: {current_state}
    Generate {num_thoughts} different next steps to solve this problem.
    Each should explore a different approach."""
    return llm.generate(prompt)
```

### Phase 2: Evaluate Thoughts

Score each thought on its promise toward the solution:
- **Feasibility**: Can this path lead to a solution? (1-10)
- **Quality**: How good is this intermediate result? (1-10)
- **Progress**: Does it move us closer to the goal? (1-10)

```python
def evaluate_thought(thought: str, goal: str) -> float:
    """Score a thought based on its potential"""
    prompt = f"""Evaluate this reasoning step toward the goal.
    Thought: {thought}
    Goal: {goal}
    Rate from 1-10 how promising this direction is."""
    return float(llm.generate(prompt))
```

### Phase 3: Expand Promising Branches

Select the best thoughts and develop them further:
- **Selection criteria**: Choose top-k scored thoughts
- **Recursive expansion**: Generate new thoughts from selected nodes
- **Depth management**: Track tree depth to prevent infinite expansion

### Phase 4: Prune Low-Scoring Branches

Remove unpromising paths to conserve resources:
- **Threshold pruning**: Eliminate thoughts below score threshold
- **Relative pruning**: Keep only top-k thoughts at each level
- **Dead-end detection**: Identify and remove stuck branches

### Phase 5: Iterate Until Solution

Continue the generate-evaluate-expand-prune cycle:
- **Termination conditions**: Solution found OR max depth reached OR all paths pruned
- **Solution verification**: Confirm final answer satisfies constraints
- **Path reconstruction**: Trace back through tree to show reasoning

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Complex problem-solving**: Math puzzles, logic problems, optimization tasks
- **Creative tasks**: Story writing with multiple plot options, brainstorming
- **Strategic planning**: Game playing (chess, Go), decision-making under uncertainty
- **Multi-step reasoning**: Problems requiring exploring dead-ends before finding solution
- **Constraint satisfaction**: Sudoku, crosswords, scheduling problems
- **Code generation**: Exploring multiple implementation approaches
- **Proof generation**: Mathematical or logical proofs requiring backtracking

### ❌ When NOT to Use

- **Simple queries**: Questions with obvious direct answers
- **Time-critical tasks**: ToT has higher latency due to exploration
- **Limited computation**: Exploring multiple branches increases cost significantly
- **Well-defined algorithms**: When optimal sequence of steps is known
- **Factual retrieval**: When you just need to look up information
- **Straightforward reasoning**: When Chain-of-Thought suffices

## Rule of Thumb

**Use Tree of Thoughts when:**
1. Problem requires **exploring multiple approaches** before finding solution
2. **Dead-ends are expected** and backtracking is necessary
3. **Quality matters more than speed** (willing to pay computational cost)
4. Need to **compare alternatives** before committing to a path
5. Problem has **multiple valid solutions** and you want the best one

**Don't use Tree of Thoughts when:**
1. Problem has obvious direct solution path (use CoT instead)
2. Latency is critical (exploration adds 5-20x overhead)
3. Budget is constrained (ToT uses many LLM calls)
4. Task is purely sequential with no branching decisions
5. Simple trial-and-error would work

## Core Components

### 1. Thought Generation

The mechanism for creating alternative reasoning paths:

```python
class ThoughtGenerator:
    def generate(self, state: State, num_thoughts: int) -> list[Thought]:
        """Generate diverse next steps from current state"""
        prompt = self._build_generation_prompt(state)
        thoughts = []
        for _ in range(num_thoughts):
            thought = llm.generate(prompt)
            thoughts.append(Thought(content=thought, parent=state))
        return thoughts
```

**Key considerations:**
- **Diversity**: Ensure thoughts explore different directions
- **Relevance**: Keep thoughts focused on the goal
- **Granularity**: Right level of detail (not too broad/narrow)

### 2. Thought Evaluation

Scoring mechanism to assess thought quality:

```python
class ThoughtEvaluator:
    def evaluate(self, thought: Thought, goal: str) -> float:
        """Score thought on scale of 0-10"""
        criteria = [
            "Feasibility: Can this lead to solution?",
            "Progress: Does this move closer to goal?",
            "Quality: Is this a good intermediate state?"
        ]
        scores = [self._score_criterion(thought, c) for c in criteria]
        return sum(scores) / len(scores)
```

**Evaluation strategies:**
- **Heuristic-based**: Use problem-specific rules
- **LLM-based**: Ask LLM to score each thought
- **Hybrid**: Combine multiple evaluation methods

### 3. Search Strategy

Algorithm for traversing the thought tree:

**Breadth-First Search (BFS)**
- Explores all thoughts at depth N before depth N+1
- Finds shortest solution path
- Higher memory usage (keeps all nodes at current level)

**Depth-First Search (DFS)**
- Explores one path fully before trying alternatives
- Lower memory usage
- May find longer paths but faster to first solution

**Beam Search**
- Hybrid: BFS but keeps only top-k best nodes at each level
- Balances exploration with resource constraints
- Most common choice for ToT

**Best-First Search**
- Always expands highest-scoring node regardless of depth
- Can find optimal solution faster
- Risk of getting stuck in local optima

### 4. Tree Structure

Data structure to maintain reasoning state:

```python
@dataclass
class ThoughtNode:
    content: str          # The thought itself
    score: float         # Evaluation score
    depth: int          # Distance from root
    parent: ThoughtNode # Previous step
    children: list[ThoughtNode]  # Next steps
    is_solution: bool   # Terminal node?

    def path_from_root(self) -> list[ThoughtNode]:
        """Reconstruct reasoning path"""
        path = []
        node = self
        while node:
            path.insert(0, node)
            node = node.parent
        return path
```

## Implementation Approaches

### Approach 1: Basic BFS with Iteration Limit

Simplest implementation using breadth-first exploration:

```python
def tree_of_thoughts_bfs(problem: str, max_depth: int = 5) -> str:
    """Basic ToT with breadth-first search"""

    root = ThoughtNode(content=problem, depth=0)
    queue = [root]

    for depth in range(max_depth):
        # Generate thoughts for all nodes at current depth
        current_level = queue
        next_level = []

        for node in current_level:
            # Generate multiple next steps
            thoughts = generate_thoughts(node.content, num=3)

            for thought in thoughts:
                # Evaluate each thought
                score = evaluate_thought(thought, problem)
                child = ThoughtNode(
                    content=thought,
                    score=score,
                    depth=depth + 1,
                    parent=node
                )

                # Check if solution
                if is_solution(child.content, problem):
                    return child.path_from_root()

                next_level.append(child)

        # Prune: keep only top-k thoughts
        next_level.sort(key=lambda x: x.score, reverse=True)
        queue = next_level[:5]  # Beam width of 5

    # Return best thought found
    return max(queue, key=lambda x: x.score).path_from_root()
```

### Approach 2: Beam Search with Pruning

More sophisticated with dynamic pruning:

```python
def tree_of_thoughts_beam(
    problem: str,
    beam_width: int = 5,
    branching_factor: int = 3,
    max_depth: int = 10,
    score_threshold: float = 6.0
) -> Solution:
    """ToT with beam search and aggressive pruning"""

    root = ThoughtNode(content=problem, depth=0, score=10.0)
    beam = [root]

    for depth in range(max_depth):
        candidates = []

        # Expand each node in beam
        for node in beam:
            thoughts = generate_thoughts(node.content, branching_factor)

            for thought in thoughts:
                score = evaluate_thought(thought, problem)

                # Prune low-scoring thoughts immediately
                if score < score_threshold:
                    continue

                child = ThoughtNode(
                    content=thought,
                    score=score,
                    depth=depth + 1,
                    parent=node
                )

                # Check for solution
                if is_solution(child.content, problem):
                    return Solution(path=child.path_from_root(), found=True)

                candidates.append(child)

        # No candidates? Dead end
        if not candidates:
            break

        # Keep top-k for next iteration
        candidates.sort(key=lambda x: x.score, reverse=True)
        beam = candidates[:beam_width]

    # Return best effort
    return Solution(
        path=max(beam, key=lambda x: x.score).path_from_root(),
        found=False
    )
```

### Approach 3: Monte Carlo Tree Search (Advanced)

Uses simulation and exploration/exploitation balance:

```python
def monte_carlo_tree_search(
    problem: str,
    num_simulations: int = 100,
    exploration_weight: float = 1.4
) -> Solution:
    """MCTS variant of Tree of Thoughts"""

    root = MCTSNode(content=problem)

    for _ in range(num_simulations):
        # Selection: Use UCB1 to select promising nodes
        node = select_node(root, exploration_weight)

        # Expansion: Add new thoughts
        if not node.is_fully_expanded():
            node = expand_node(node)

        # Simulation: Random playout from node
        reward = simulate_playout(node, problem)

        # Backpropagation: Update scores up the tree
        backpropagate(node, reward)

    # Return best path
    return best_child(root).path_from_root()
```

## Key Benefits

### 🧠 Superior Problem-Solving

- **Exploration**: Systematically explores multiple solution strategies
- **Comparison**: Evaluates alternatives before committing
- **Optimization**: Finds better solutions than greedy approaches
- **Robustness**: Less likely to get stuck in dead ends

### 🎯 Better Solution Quality

- **Global search**: Not limited to first acceptable solution
- **Quality-driven**: Scores guide search toward best paths
- **Verification**: Can verify and compare multiple solutions
- **Confidence**: Provides score-based confidence in solution

### 🔄 Strategic Backtracking

- **Recovery**: Can backtrack when path proves unproductive
- **Efficiency**: Prunes bad paths early to save computation
- **Memory**: Keeps track of attempted approaches
- **Learning**: Incorporates evaluation feedback

### 🔍 Transparency and Interpretability

- **Tree visualization**: Shows entire exploration process
- **Decision rationale**: Scores explain why paths were chosen
- **Alternative paths**: Shows other options that were considered
- **Debugging**: Easy to identify where reasoning went wrong

## Trade-offs

### ⚠️ High Computational Cost

**Issue**: Multiple LLM calls per step (generation + evaluation + expansion)

**Impact**: 10-100x more expensive than Chain-of-Thought

**Cost breakdown**:
- Depth 3, branching factor 3, beam width 5: ~30-50 LLM calls
- Depth 5, branching factor 5, beam width 10: ~200-300 LLM calls

**Mitigation**:
- Use smaller models for evaluation (GPT-4o-mini, Claude Haiku)
- Aggressive pruning with score thresholds
- Cache similar thought evaluations
- Limit tree depth and branching factor
- Early termination when good solution found

### ⚠️ Increased Latency

**Issue**: Sequential exploration of multiple branches takes time

**Impact**: 5-20x slower than direct prompting

**Mitigation**:
- Parallelize thought generation and evaluation
- Use streaming for incremental results
- Implement time limits
- Adjust beam width based on urgency

### ⚠️ Complexity

**Issue**: More complex to implement and debug than linear reasoning

**Challenges**:
- Managing tree state
- Implementing search strategies
- Tuning hyperparameters (beam width, depth, scoring)
- Preventing infinite loops

**Mitigation**:
- Start with simple BFS implementation
- Use existing frameworks (LangGraph, guidance)
- Comprehensive logging and visualization
- Clear termination conditions

### ⚠️ Evaluation Quality Dependency

**Issue**: Entire approach depends on accurate thought scoring

**Risk**: Poor evaluation leads to exploring wrong branches

**Mitigation**:
- Use domain-specific heuristics when possible
- Combine multiple evaluation methods
- Calibrate scores on example problems
- Include verification steps
- Human-in-the-loop for critical evaluations

## Best Practices

### 1. Thought Generation Design

```python
# ❌ BAD: Vague, undirected generation
prompt = "Give me ideas for solving this problem"

# ✅ GOOD: Specific, diverse generation
prompt = """Given the current state: {state}
Generate {n} distinct approaches to progress toward the goal.
Each approach should:
1. Be concrete and actionable
2. Explore a different strategy
3. Build logically on the current state

Goal: {goal}
Current state: {state}

Provide {n} different next steps:"""
```

### 2. Evaluation Criteria

```python
class ThoughtEvaluator:
    def evaluate(self, thought: Thought) -> EvaluationResult:
        """Multi-criteria evaluation"""
        return EvaluationResult(
            feasibility=self._check_feasibility(thought),  # Can this work?
            progress=self._measure_progress(thought),      # Are we closer?
            quality=self._assess_quality(thought),         # Is this good?
            novelty=self._check_novelty(thought),         # Is this new?

            # Combine scores
            overall=weighted_average([...])
        )
```

### 3. Hyperparameter Tuning

```python
# Problem complexity determines parameters
if problem_type == "simple_puzzle":
    config = ToTConfig(
        max_depth=3,
        beam_width=3,
        branching_factor=3,
        score_threshold=6.0
    )
elif problem_type == "complex_strategy":
    config = ToTConfig(
        max_depth=7,
        beam_width=10,
        branching_factor=5,
        score_threshold=7.0
    )
```

### 4. Early Termination

```python
def should_terminate(state: ToTState) -> bool:
    """Check if search should stop"""
    if state.solution_found:
        return True
    if state.depth >= state.max_depth:
        return True
    if not state.beam:  # No more candidates
        return True
    if state.best_score < state.min_acceptable_score:
        return True  # Give up, no good paths
    return False
```

### 5. Solution Verification

```python
def verify_solution(solution: Solution, problem: Problem) -> bool:
    """Verify solution satisfies all constraints"""

    # Check solution completeness
    if not solution.is_complete():
        return False

    # Verify constraints
    for constraint in problem.constraints:
        if not constraint.satisfied_by(solution):
            return False

    # Test correctness
    if problem.has_test_cases():
        return all(solution.passes(test) for test in problem.test_cases)

    return True
```

## Performance Metrics

Track these metrics to optimize ToT performance:

### Effectiveness Metrics
- **Solution quality**: How good is the final answer?
- **Success rate**: % of problems solved correctly
- **Solution diversity**: Number of distinct solutions found
- **Optimality**: How close to optimal solution?

### Efficiency Metrics
- **LLM calls**: Total number of generation + evaluation calls
- **Average tree depth**: How deep did search go?
- **Branching factor**: Average children per node
- **Pruning rate**: % of thoughts pruned before expansion
- **Time to solution**: Latency from start to answer

### Cost Metrics
- **Token usage**: Total input + output tokens
- **Cost per problem**: Dollar cost of solving
- **Cost vs. quality trade-off**: Is better solution worth the cost?

### Search Metrics
- **Beam utilization**: How many beam slots used per level?
- **Dead-end rate**: % of branches that hit dead ends
- **Backtrack frequency**: How often did we backtrack?
- **Exploration breadth**: Number of unique strategies tried

## Example Scenarios

### Scenario 1: Game of 24

```
Problem: Use numbers [4, 9, 3, 2] with operations (+, -, *, /) to make 24

Root: [4, 9, 3, 2] → Target: 24

Level 1: First operation (9 candidates, keep top 3)
├─ 9 * 3 = 27 → [4, 27, 2] (Score: 8.5/10) ✓
├─ 9 * 2 = 18 → [4, 3, 18] (Score: 8.0/10) ✓
├─ 4 * 2 = 8  → [9, 3, 8]  (Score: 7.5/10) ✓
├─ 9 + 3 = 12 → [4, 12, 2] (Score: 6.5/10) ✗ pruned
└─ 4 + 3 = 7  → [9, 7, 2]  (Score: 5.0/10) ✗ pruned

Level 2: Expand top 3 (9 candidates, keep top 3)
From [4, 27, 2]:
├─ 27 - 2 = 25 → [4, 25] (Score: 7.5/10) ✓
├─ 27 - 4 = 23 → [2, 23] (Score: 7.0/10) ✓
└─ 4 + 2 = 6  → [27, 6] (Score: 6.0/10) ✗ pruned

From [4, 3, 18]:
├─ 18 + 4 = 22 → [3, 22] (Score: 7.8/10) ✓
└─ 18 + 3 = 21 → [4, 21] (Score: 7.2/10) ✗ pruned

Level 3: Final operations
From [4, 25]:
└─ 25 - 4 = 21 → ✗ Not 24

From [2, 23]:
└─ 23 + 2 = 25 → ✗ Not 24

From [3, 22]:
└─ 22 + 3 = 25 → ✗ Not 24

Backtrack to Level 1, expand 4th best option...
From [4, 12, 2]:
├─ 12 * 2 = 24 → [4, 24] (Score: 9.5/10) ✓✓
  └─ SOLUTION FOUND: (9 + 3) * 2 = 24 (unused: 4)

Alternative path: 4 * (9 - 3) = 24
```

### Scenario 2: Creative Writing - Story Opening

```
Problem: Write an engaging opening for a sci-fi mystery novel

Root: "Write opening for sci-fi mystery"

Level 1: Story hooks (Score range: 5-9)
├─ Branch A: "Woke up in spaceship with no memory" (Score: 7/10) ✓
├─ Branch B: "Message from the future warns of disaster" (Score: 8/10) ✓
├─ Branch C: "AI detective investigates impossible murder" (Score: 9/10) ✓
└─ Branch D: "Routine mission discovers alien artifact" (Score: 6/10) ✗

Level 2: Opening lines for top branches

Branch C (AI detective): Expanding...
├─ "Detective Algorithm-7 stared at the body that shouldn't exist" (9/10) ✓
├─ "In 2157, murder was supposed to be impossible" (8/10) ✓
└─ "The crime scene made no sense, even to a quantum processor" (7/10) ✗

Branch B (future message): Expanding...
├─ "The transmission arrived exactly at midnight, from forty years forward" (8.5/10) ✓
└─ "WARNING: DO NOT OPEN THIS MESSAGE" (7.5/10) ✗

Level 3: Full paragraph development

Top candidate: "Detective Algorithm-7 stared at the body that shouldn't exist"
Expansion:
"Detective Algorithm-7 stared at the body that shouldn't exist. In the thirty-two
years since the Consciousness Upload Act, murder had become a technical
impossibility—you couldn't kill what wasn't technically alive. Yet here lay Dr.
Sarah Chen, very much dead, in a sealed room that no one could have entered.
The logical contradictions cascaded through my neural net like dominoes, each
impossible fact triggering alarms I didn't know I could feel."

Score: 9.5/10 ✓ SELECTED AS BEST OPENING
```

### Scenario 3: Strategic Planning - Chess Move

```
Problem: Find best move as White (Board state given)

Root: Current board position → Find optimal move

Level 1: Candidate moves (Generated 8, evaluate all)
├─ Qh5+ (Queen check)     → Score: 8.5/10 ✓
├─ Nf6+ (Knight fork)     → Score: 9.0/10 ✓
├─ Bc4 (Bishop development) → Score: 7.0/10 ✓
├─ O-O (Castle kingside)  → Score: 6.5/10 ✗
└─ h3 (Prevent pin)       → Score: 5.0/10 ✗

Level 2: Opponent responses (Top 3 moves, 3 responses each)

Nf6+ Branch: What if opponent plays...
├─ King to g8    → White: Nxe8 (Score: 9.5/10) ✓✓
├─ King to f7    → White: Nxd7 (Score: 8.0/10) ✓
└─ Blocks with Bd7 → White: Nxd7 (Score: 7.5/10) ✗

Qh5+ Branch: What if opponent plays...
├─ King to d8    → White: Qf7 (Score: 7.5/10) ✓
├─ g6 blocks     → White: Qxh7 (Score: 6.5/10) ✗
└─ Nf6 blocks    → White: Qxf7+ (Score: 8.0/10) ✓

Level 3: 2-move lookahead for top paths

Nf6+ → Kg8 → Nxe8 → Opponent must respond
├─ Qxe8 (Queen takes)  → Material: +5 (Score: 9.0/10) ✓
├─ Rxe8 (Rook takes)   → Material: +3 (Score: 8.5/10) ✓
└─ Ignores (develops)  → Material: +5 (Score: 9.5/10) ✓✓

BEST MOVE FOUND: Nf6+ (Knight fork)
Reasoning path: Forces king move → Captures rook → Win material
Expected outcome: +5 material advantage
```

## Advanced Patterns

### 1. Hybrid Search Strategies

Combine multiple search approaches:

```python
class HybridToTSearch:
    def search(self, problem: Problem) -> Solution:
        # Start with BFS for broad exploration
        candidates = self.bfs_exploration(problem, depth=2)

        # Switch to DFS for deep investigation of top candidates
        best_candidate = max(candidates, key=lambda x: x.score)
        solution = self.dfs_exploitation(best_candidate, depth=5)

        # If no solution, try beam search with different parameters
        if not solution:
            solution = self.beam_search(problem, beam_width=10)

        return solution
```

### 2. Dynamic Beam Width

Adjust exploration based on progress:

```python
def adaptive_beam_search(problem: Problem) -> Solution:
    beam_width = 5  # Start conservative

    for depth in range(max_depth):
        candidates = expand_beam(beam, beam_width)

        # If making good progress, expand search
        if avg_score(candidates) > 8.0:
            beam_width = min(beam_width + 2, 15)

        # If struggling, focus on best paths
        elif avg_score(candidates) < 6.0:
            beam_width = max(beam_width - 1, 3)

        beam = prune_to_width(candidates, beam_width)
```

### 3. Thought Refinement

Iteratively improve thoughts:

```python
def refine_thought(thought: Thought, goal: str) -> Thought:
    """Refine low-scoring thought before pruning"""
    if thought.score < 7.0 and thought.score > 5.0:
        # Ask LLM to improve the thought
        refined = llm.generate(f"""
        This reasoning step scored {thought.score}/10.
        Thought: {thought.content}
        Goal: {goal}

        How could this reasoning step be improved to better progress toward the goal?
        """)

        refined_score = evaluate_thought(refined, goal)
        if refined_score > thought.score:
            return Thought(content=refined, score=refined_score)

    return thought
```

### 4. Multi-Objective Optimization

Balance multiple criteria:

```python
class MultiObjectiveEvaluator:
    def evaluate(self, thought: Thought) -> float:
        """Score thought across multiple objectives"""
        scores = {
            'correctness': self.score_correctness(thought),
            'efficiency': self.score_efficiency(thought),
            'elegance': self.score_elegance(thought),
            'safety': self.score_safety(thought)
        }

        # Weighted combination based on problem priorities
        weights = self.problem.objective_weights
        return sum(scores[obj] * weights[obj] for obj in scores)
```

### 5. Ensemble Evaluation

Use multiple evaluators for robust scoring:

```python
def ensemble_evaluate(thought: Thought, goal: str) -> float:
    """Combine multiple evaluation methods"""
    scores = []

    # LLM-based evaluation
    scores.append(llm_evaluate(thought, goal))

    # Heuristic-based evaluation
    scores.append(heuristic_evaluate(thought, goal))

    # Similarity to known good solutions
    scores.append(similarity_evaluate(thought, solution_db))

    # Weighted average with confidence weighting
    return weighted_average(scores, confidence_weights)
```

## Comparison with Related Patterns

| Pattern | Exploration | Evaluation | Backtracking | Best For |
|---------|-------------|------------|--------------|----------|
| **Tree of Thoughts** | Multi-path, breadth/depth | Explicit scoring | Yes, strategic | Complex problem-solving |
| **Chain of Thought** | Single path | Implicit | No | Sequential reasoning |
| **Graph of Thoughts** | Arbitrary graph | Flexible | Yes, via graph | Complex dependencies |
| **ReAct** | Sequential tools | Observation-based | Limited | Tool-based tasks |
| **Prompt Chaining** | Fixed sequence | None | No | Known workflows |
| **Self-Consistency** | Multiple samples | Vote-based | No | Uncertain reasoning |

### When to Choose ToT

- **vs CoT**: When problem requires exploring dead-ends and alternative paths
- **vs GoT**: When tree structure suffices (no need for cycles/merges)
- **vs ReAct**: When reasoning about internal problem structure, not external tools
- **vs Self-Consistency**: When you need systematic exploration, not sampling

## Common Pitfalls

### 1. Explosion of Branches

**Problem**: Tree grows exponentially, consuming excessive resources

**Symptoms**: Out of memory, extreme latency, high costs

**Solution**:
- Set aggressive pruning thresholds
- Limit branching factor (2-5 thoughts per node)
- Use beam search with narrow width
- Implement depth limits

```python
# ❌ BAD: Unbounded growth
for node in current_level:
    children.extend(generate_thoughts(node, num=10))  # 10^depth nodes!

# ✅ GOOD: Bounded with pruning
for node in beam[:beam_width]:  # Limited beam
    candidates = generate_thoughts(node, num=3)  # Moderate branching
    children.extend([c for c in candidates if c.score > threshold])
```

### 2. Poor Evaluation Function

**Problem**: Scores don't correlate with solution quality

**Symptoms**: Good paths pruned, bad paths explored, no progress

**Solution**:
- Use domain-specific heuristics
- Validate evaluator on known problems
- Combine multiple evaluation methods
- Include human feedback loop

### 3. Premature Convergence

**Problem**: All beams converge to similar thoughts, losing diversity

**Symptoms**: Exploring same ideas repeatedly, missing alternatives

**Solution**:
- Add diversity bonus to scores
- Penalize similarity to existing nodes
- Maintain multiple diverse beams
- Periodically inject random exploration

```python
def diversity_adjusted_score(thought: Thought, beam: list[Thought]) -> float:
    base_score = evaluate_thought(thought)

    # Penalize similarity to existing beam thoughts
    similarity_penalty = max(
        similarity(thought, existing) for existing in beam
    )

    return base_score - (0.3 * similarity_penalty)
```

### 4. Ignoring Intermediate Quality

**Problem**: Focus only on reaching goal, ignore thought quality along the way

**Symptoms**: Poor reasoning paths, incoherent logic

**Solution**:
- Evaluate thought quality independently
- Include coherence and logic checks
- Verify each step builds on previous
- Add intermediate checkpoints

### 5. No Solution Verification

**Problem**: Accept first thought marked as solution without verification

**Symptoms**: Wrong answers accepted, constraints violated

**Solution**:
- Always verify solutions against constraints
- Test solutions with test cases
- Compare multiple candidate solutions
- Include verification in scoring

## Conclusion

The Tree of Thoughts pattern represents a significant advancement in LLM reasoning capabilities, enabling systematic exploration of solution spaces that single-path methods cannot navigate effectively. By combining thought generation, evaluation, and strategic search, ToT empowers LLMs to solve complex problems that require deliberation, comparison, and backtracking.

**Use Tree of Thoughts when:**
- Problem requires exploring multiple approaches before finding solution
- Dead-ends are expected and backtracking is essential
- Quality of solution matters more than speed
- Need to compare alternatives systematically
- Problem has complex constraint satisfaction requirements

**Implementation checklist:**
- ✅ Define clear thought generation prompts
- ✅ Implement robust evaluation function (multi-criteria)
- ✅ Choose appropriate search strategy (BFS/DFS/Beam)
- ✅ Set hyperparameters (depth, width, branching, threshold)
- ✅ Implement pruning to control costs
- ✅ Add solution verification logic
- ✅ Include visualization of thought tree
- ✅ Log metrics (LLM calls, depth, scores)
- ✅ Set iteration and time limits
- ✅ Test on representative problems

**Key Takeaways:**
- 🌳 ToT explores multiple reasoning paths in tree structure
- 🎯 Explicit evaluation guides search toward best solutions
- 🔄 Strategic backtracking prevents getting stuck
- 💰 Trade-off: Better solutions at higher computational cost
- 🔍 Tree visualization provides transparency
- ⚖️ Balance exploration breadth vs. depth vs. cost
- 🛠️ Hyperparameter tuning critical for performance

**Performance Guidelines:**
- **Simple puzzles**: Depth 3-5, beam width 3-5, branching 3
- **Complex problems**: Depth 5-10, beam width 5-10, branching 3-5
- **Creative tasks**: Depth 3-7, beam width 10-15, branching 5-7
- **Cost-sensitive**: Aggressive pruning (threshold 7+), narrow beam (3-5)

---

*Tree of Thoughts transforms LLMs from linear reasoners into strategic problem-solvers that can explore, evaluate, and optimize—unlocking solutions to problems that require deliberation, backtracking, and systematic exploration of alternatives.*
