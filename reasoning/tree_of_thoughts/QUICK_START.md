# Tree of Thoughts - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Tree of Thoughts Directory
```bash
cd reasoning/tree_of_thoughts
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic ToT with BFS (Game of 24 puzzle)
- **Option 2**: Advanced ToT with Beam Search (Creative writing)
- **Option 3**: Run all examples

---

## 📖 Understanding Tree of Thoughts in 30 Seconds

**Tree of Thoughts** = Explore multiple reasoning paths like a tree

Instead of following one path:
```
Problem → Step 1 → Step 2 → Step 3 → Solution
```

ToT explores multiple alternatives:
```
                    ┌─ Path A → Dead End
Problem → Branch ──┼─ Path B → Partial Solution → Expand
                    └─ Path C → Best Solution ✓
```

The agent:
1. **Generate**: Create multiple next steps (branches)
2. **Evaluate**: Score each step (1-10)
3. **Expand**: Develop promising branches
4. **Prune**: Remove low-scoring paths
5. **Iterate**: Repeat until solution found

---

## 🛠️ What's in Each Implementation

### Basic Implementation (tot_basic.py)
- **Problem**: Game of 24 puzzle - use 4 numbers to make 24
- **Search**: Breadth-First Search (BFS)
- **Exploration**: Systematic level-by-level
- **Visualization**: ASCII tree in console
- **Good for**: Understanding the core ToT concept

### Advanced Implementation (tot_advanced.py)
- **Problem**: Creative story writing with branching plots
- **Search**: Beam Search with pruning
- **Features**: Backtracking, adaptive scoring, path reconstruction
- **Visualization**: Detailed decision tree with scores
- **Good for**: Complex problem-solving, production use

---

## 💡 Example Problems

### Game of 24 (Basic)
```
Input: Use [4, 6, 8, 2] to make 24
ToT explores:
  - Try 4 * 6 = 24 ✓ (Found it!)
  - Try 8 + 6 = 14 → 14 + 4 = 18 ✗
  - Try 8 * 2 = 16 → 16 + 6 = 22 ✗
```

### Creative Writing (Advanced)
```
Task: Write engaging sci-fi opening
ToT explores:
  - "Woke up with no memory" (Score: 7/10)
  - "AI detective impossible murder" (Score: 9/10) ✓
  - "Routine mission alien artifact" (Score: 6/10)
Best path selected based on scores
```

---

## 🎯 Key Concepts

### Thought Generation
Generate multiple possible next steps:
```python
generate_thoughts(current_state, num=3)
# Returns: [thought1, thought2, thought3]
```

### Thought Evaluation
Score each thought on its promise (1-10):
```python
evaluate_thought(thought, goal)
# Returns: 8.5  (high score = promising)
```

### Search Strategies

**Breadth-First Search (BFS)**
- Explores all options at depth N before depth N+1
- Finds shortest path
- Uses more memory

**Beam Search**
- Keeps only top-k best nodes at each level
- Balances exploration with resource constraints
- Most practical for real applications

**Depth-First Search (DFS)**
- Explores one path fully before trying others
- Uses less memory
- May miss better solutions

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic (BFS) | Advanced (Beam) |
|---------|-------------|-----------------|
| Search Strategy | Breadth-First | Beam Search |
| Pruning | Simple (keep top-k) | Threshold + Top-k |
| Backtracking | No | Yes |
| Problem Type | Logic puzzle | Creative task |
| Complexity | Low | Medium-High |
| Output | Simple tree | Detailed visualization |
| Best For | Learning | Production |

**Recommendation**: Start with Basic to understand concepts, use Advanced for real problems.

---

## 🔧 Configuration Options

### Hyperparameters You Can Adjust

**max_depth** (default: 5)
- How many levels deep to explore
- Higher = more thorough, more expensive
- Typical range: 3-10

**beam_width** (default: 5)
- How many paths to keep at each level
- Higher = more exploration, higher cost
- Typical range: 3-15

**branching_factor** (default: 3)
- How many thoughts to generate per node
- Higher = more alternatives, more expensive
- Typical range: 2-7

**score_threshold** (default: 6.0)
- Minimum score to keep exploring a path
- Higher = more aggressive pruning, lower cost
- Typical range: 5.0-8.0

### Example Configuration

```python
# For simple puzzles
config = {
    'max_depth': 3,
    'beam_width': 3,
    'branching_factor': 3,
    'score_threshold': 6.0
}

# For complex problems
config = {
    'max_depth': 7,
    'beam_width': 10,
    'branching_factor': 5,
    'score_threshold': 7.0
}
```

---

## ⚡ Common Issues & Solutions

### Issue: "Tree grows too large"
**Symptoms**: Out of memory, very slow
**Solution**:
- Reduce beam_width (try 3-5)
- Increase score_threshold (try 7.0+)
- Reduce max_depth (try 3-5)

### Issue: "All paths pruned early"
**Symptoms**: No solution found, beam becomes empty
**Solution**:
- Lower score_threshold (try 5.0-6.0)
- Increase branching_factor (try 5-7)
- Check evaluation function

### Issue: "Poor solutions found"
**Symptoms**: Solution doesn't solve problem correctly
**Solution**:
- Improve evaluation function
- Add solution verification step
- Try different search strategy

### Issue: "Too expensive/slow"
**Symptoms**: High costs, long wait times
**Solution**:
- Use smaller model for evaluation
- Aggressive pruning (high threshold)
- Reduce depth and beam width
- Enable early termination

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)
- **Related Patterns**:
  - Chain of Thought: Linear reasoning
  - Graph of Thoughts: More complex dependencies
  - ReAct: Reasoning with external tools

---

## 🎓 Learning Path

1. ✅ **Start**: Run basic example (Game of 24)
2. ✅ **Observe**: Watch how tree explores multiple paths
3. ✅ **Understand**: See how evaluation guides search
4. ✅ **Explore**: Run advanced example (creative writing)
5. ✅ **Experiment**: Modify hyperparameters and observe effects
6. ✅ **Apply**: Try your own problems

---

## 🌟 Pro Tips

### When to Use ToT
✅ Use when:
- Problem requires exploring dead-ends
- Need to compare multiple approaches
- Quality matters more than speed
- Backtracking is necessary

❌ Don't use when:
- Problem has obvious solution path
- Speed is critical
- Budget is very limited
- Chain-of-Thought would suffice

### Optimization Tips
1. **Start Conservative**: Small depth, narrow beam
2. **Profile First**: Measure cost before scaling up
3. **Prune Aggressively**: High threshold saves money
4. **Verify Solutions**: Always check correctness
5. **Visualize Trees**: Helps debug and optimize
6. **Cache Evaluations**: Reuse scores when possible

### Debugging Tips
1. **Check Evaluation**: Print scores to verify reasonable
2. **Visualize Tree**: See what paths are explored
3. **Track Depth**: Monitor how deep search goes
4. **Count LLM Calls**: Understand cost breakdown
5. **Test Small First**: Start with depth=2, width=2

---

## 📈 Performance Expectations

### Basic Example (Game of 24)
- Depth: 3 levels
- Beam Width: 5
- LLM Calls: ~30-40
- Time: 10-20 seconds
- Cost: ~$0.05-0.10

### Advanced Example (Creative Writing)
- Depth: 5 levels
- Beam Width: 10
- LLM Calls: ~80-120
- Time: 30-60 seconds
- Cost: ~$0.20-0.40

*Note: Times and costs vary based on model choice and API speed*

---

## 🔍 Understanding the Output

### Tree Visualization
```
Root: [Problem]
├─ Thought 1 (Score: 8.5) ✓ Expanded
│  ├─ Thought 1.1 (Score: 9.0) ✓✓ Solution!
│  └─ Thought 1.2 (Score: 7.0) ✗ Pruned
├─ Thought 2 (Score: 7.5) ✓ Expanded
│  └─ Thought 2.1 (Score: 6.0) ✗ Pruned
└─ Thought 3 (Score: 5.5) ✗ Pruned (below threshold)
```

### Score Meanings
- **9-10**: Excellent, very promising path
- **7-8**: Good, worth exploring
- **5-6**: Marginal, might prune
- **3-4**: Poor, likely pruned
- **1-2**: Very poor, definitely pruned

### Path Reconstruction
Shows the reasoning steps from root to solution:
```
Step 1: [Initial state] → Action: Try approach A
Step 2: [After A] → Action: Refine with B
Step 3: [After B] → Action: Complete with C
Solution: [Final answer] ✓
```

---

## 🎯 Quick Reference Commands

```bash
# Install dependencies
uv sync

# Run basic example
uv run python src/tot_basic.py

# Run advanced example
uv run python src/tot_advanced.py

# Run all examples
bash run.sh
# Select option 3

# Check your setup
python -c "import langchain; print('Setup OK!')"
```

---

**Happy Exploring! 🌳**

For questions or issues, refer to the full [README.md](./README.md).
