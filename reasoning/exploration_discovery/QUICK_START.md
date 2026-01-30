# Exploration and Discovery Pattern - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Exploration Discovery Directory
```bash
cd reasoning/exploration_discovery
```

### Step 2: Install Dependencies (if not already installed)
```bash
uv sync
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic Epsilon-Greedy Exploration
- **Option 2**: Advanced UCB (Upper Confidence Bound) Exploration
- **Option 3**: Run all examples

---

## 📖 Understanding Exploration & Discovery in 30 Seconds

**Exploration and Discovery** = Systematically searching solution spaces to find novel, diverse options

The core mechanism is the **Exploration-Exploitation Trade-off**:
- **Exploration**: Try new, untested ideas (maximize novelty and learning)
- **Exploitation**: Refine known good ideas (maximize immediate value)

The agent balances these using strategies like:
- **Epsilon-Greedy**: Random probability (ε) determines explore vs. exploit
- **UCB**: Mathematical approach balancing reward and uncertainty
- **Curiosity-Driven**: Follow information gain and surprise

---

## 🎯 Key Concepts

### Epsilon (ε) Parameter
Controls exploration vs. exploitation balance:
- **ε = 1.0**: Pure exploration (completely random)
- **ε = 0.5**: Balanced (50% explore, 50% exploit)
- **ε = 0.0**: Pure exploitation (only refine best)

Most implementations use **epsilon decay**: start high (0.9), gradually decrease (0.95 decay rate).

### Multi-Dimensional Evaluation
Each discovery is scored on multiple dimensions:
- **Novelty**: How different from existing ideas (0.0-1.0)
- **Feasibility**: How practical to implement (0.0-1.0)
- **Impact**: Expected value or benefit (0.0-1.0)

Combined into overall score with weighted sum.

### Diversity Metrics
- **Cluster count**: Number of distinct idea categories
- **Pairwise distance**: Average similarity between all ideas
- **Coverage**: Percentage of solution space explored

---

## 🛠️ Available Implementations

### Basic Implementation (Epsilon-Greedy)
- Simple exploration-exploitation balance
- Epsilon decay over iterations
- Novelty detection with semantic similarity
- Good for: Creative brainstorming, idea generation

### Advanced Implementation (UCB)
- Upper Confidence Bound algorithm
- Optimized exploration efficiency
- Multi-dimensional clustering
- Adaptive exploration based on uncertainty
- Good for: Complex discovery tasks, hypothesis generation

---

## 💡 Example Queries to Try

### Creative Brainstorming
```
"Generate innovative business ideas for sustainable living"
```
Expected: 10-20 diverse ideas across multiple categories (energy, food, transportation, etc.)

### Research Hypothesis Discovery
```
"Discover research hypotheses about remote work productivity"
```
Expected: Multiple testable hypotheses exploring different factors (environment, technology, social dynamics)

### Product Feature Discovery
```
"Explore potential features for a project management tool"
```
Expected: Diverse feature ideas across different aspects (collaboration, automation, analytics)

### Market Opportunity Analysis
```
"Identify market opportunities in the education technology space"
```
Expected: Various opportunity areas with different risk-reward profiles

---

## 📊 Understanding the Output

### Basic Example Output
```
Iteration 5/20 (ε=0.73)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Mode: EXPLORE
💡 Idea: "Community-owned solar microgrids for apartment buildings"

📊 Evaluation:
  Novelty:     ████████░░ 0.88
  Feasibility: ███████░░░ 0.76
  Impact:      █████████░ 0.91
  Overall:     ████████░░ 0.85

✓ New Discovery Added

Current Portfolio:
  - Total Discoveries: 5
  - Diversity Score: 0.72
  - Best Overall: 0.85 (current)
```

### Advanced Example Output
```
UCB Selection - Iteration 8/25
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cluster Selection:
  Cluster: "Technology Integration"
  UCB Score: 1.89
  Avg Reward: 0.78 | Visits: 3 | Exploration Bonus: 0.45

💡 Hypothesis: "AI-powered context switching reduces cognitive load"

📊 Evaluation:
  Novelty:     ████████░░ 0.84
  Feasibility: ████████░░ 0.82
  Impact:      █████████░ 0.89
  Overall:     ████████░░ 0.85

Cluster Stats Updated:
  Total Visits: 4
  Average Reward: 0.80 (+0.02)
```

---

## 🔧 Customization Tips

### Adjust Exploration Rate

```python
# In src/exploration_basic.py
explorer = EpsilonGreedyExplorer(
    epsilon=0.95,        # Start with 95% exploration
    epsilon_decay=0.98,  # Slower decay (was 0.95)
    min_epsilon=0.1      # Don't go below 10% exploration
)
```

### Modify Evaluation Weights

```python
# Change importance of different dimensions
score = (
    0.40 * novelty +      # Increase if creativity is most important
    0.30 * feasibility +  # Increase if practicality is key
    0.30 * impact         # Increase if value is critical
)
```

### Set Iteration Limits

```python
# In run scripts
max_iterations = 25  # Increase for more thorough exploration
```

### Adjust Convergence Criteria

```python
convergence_detector = ConvergenceDetector(
    patience=8,          # Wait 8 iterations for improvement
    threshold=0.03       # Consider converged if improvement < 0.03
)
```

---

## ⚡ Common Issues & Solutions

### Issue: "All ideas are similar/not diverse"
**Solution**:
- Increase `epsilon` (start at 0.95 instead of 0.9)
- Decrease `epsilon_decay` (0.98 instead of 0.95)
- Increase `max_iterations` for more exploration time

### Issue: "Ideas are creative but impractical"
**Solution**:
- Increase feasibility weight in evaluation
- Add feasibility threshold filter
- Start with lower epsilon (0.7) for more exploitation

### Issue: "Exploration never converges"
**Solution**:
- Set strict `max_iterations` limit
- Adjust convergence `patience` (reduce from 5 to 3)
- Use diversity saturation as stopping criterion

### Issue: "Duplicate discoveries"
**Solution**:
- Lower novelty threshold for acceptance (e.g., must be > 0.7)
- Improve semantic similarity detection
- Use better embedding model for novelty calculation

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ Start: Run the basic epsilon-greedy example
2. ✅ Understand: Watch how epsilon decays and mode switches between explore/exploit
3. ✅ Explore: Run the advanced UCB example to see optimized exploration
4. ✅ Experiment: Modify epsilon, decay rate, and weights
5. ✅ Customize: Try your own exploration problem
6. ✅ Integrate: Use exploration in your applications

---

## 🌟 Pro Tips

### 1. Start with High Exploration
Begin with ε ≥ 0.9 to ensure broad coverage before narrowing down.

### 2. Monitor Diversity
Check diversity metrics regularly. If diversity stops increasing, you may have converged.

### 3. Multi-Dimensional Evaluation
Don't rely on a single score. Look at novelty, feasibility, and impact separately.

### 4. Use Clustering
Group similar discoveries to understand coverage and identify gaps.

### 5. Adaptive Strategies
Let epsilon adjust based on success rate for more efficient exploration.

### 6. Set Clear Stopping Criteria
Use multiple signals: iteration limit, diversity plateau, quality threshold.

---

## 🔄 Exploration vs. Exploitation Examples

### Pure Exploration (ε=1.0)
```
✓ Maximum novelty and diversity
✗ May find impractical ideas
Use: Initial discovery phase
```

### Balanced (ε=0.5)
```
✓ Good mix of new and refined ideas
✓ Explores while improving
Use: Mid-exploration phase
```

### Heavy Exploitation (ε=0.2)
```
✓ Refines best ideas found
✗ Less likely to discover new territory
Use: Final refinement phase
```

---

## 📈 Success Metrics to Watch

- **Novelty Rate**: % of genuinely novel discoveries (target: >70%)
- **Diversity Score**: Coverage of solution space (target: >0.7)
- **Quality Trajectory**: Is overall score improving? (should increase)
- **Cluster Count**: Number of distinct idea categories (target: 5-10)
- **Convergence Speed**: Iterations until plateau (typical: 15-25)

---

## 🚦 When to Use Each Strategy

### Use Epsilon-Greedy When:
- ✅ You want simplicity and interpretability
- ✅ Problem is moderately complex
- ✅ You can tune epsilon manually
- ✅ Good default choice

### Use UCB When:
- ✅ You want optimized exploration efficiency
- ✅ Problem has clear reward signals
- ✅ You need theoretical guarantees
- ✅ Resources are limited (fewer iterations)

### Use Curiosity-Driven When:
- ✅ Learning about domain is as valuable as solutions
- ✅ Surprises and anomalies are interesting
- ✅ You have world models to update
- ✅ Long-term exploration with no time pressure

---

**Happy Exploring! 🔍**

For questions or issues, refer to the full [README.md](./README.md).
