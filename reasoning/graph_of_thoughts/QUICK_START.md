# Graph of Thoughts - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Graph of Thoughts Directory
```bash
cd reasoning/graph_of_thoughts
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# or use the provided script
bash run.sh install
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic GoT (DAG with single round)
- **Option 2**: Advanced GoT (multi-round consensus)
- **Option 3**: Run all examples

---

## 📖 Understanding GoT in 30 Seconds

**GoT** = **Graph of Thoughts**

Unlike Tree of Thoughts (hierarchical parent-child), GoT allows:
1. **Any thought can connect to any other** (not just parent-child)
2. **Multiple perspectives** generate initial thoughts
3. **Connections** form between related thoughts (support, critique, build-on)
4. **Graph algorithms** identify key insights (PageRank, centrality)
5. **Synthesis** aggregates across connected thoughts
6. **Iteration** refines through multiple rounds (optional)

---

## 🌟 Key Difference from Tree of Thoughts

### Tree of Thoughts:
```
        Root
       /  |  \
      A   B   C     ← A, B, C can only reference Root
     / \  |
    D   E F         ← D, E reference A; F references B
```

### Graph of Thoughts:
```
        Root
       /  |  \
      A   B   C
     /|\ /|\ /|\
    | +-+-+-+ |    ← A, B, C can reference each other!
    |   | |   |
    D---E-F---G    ← D, E, F, G can reference any node
```

**GoT enables:**
- Thought B can critique Thought A
- Thought C can build on both A and B
- Thoughts can merge into synthesis nodes
- Non-hierarchical relationships

---

## 🛠️ What's Included

### Basic Implementation (`got_basic.py`)
- Multi-perspective analysis (ethical, practical, economic, legal)
- DAG structure (Directed Acyclic Graph)
- Single-round generation
- Connection evaluation
- PageRank-based aggregation
- ASCII graph visualization

### Advanced Implementation (`got_advanced.py`)
- Multi-agent consensus building (optimist, critic, pragmatist, innovator)
- Multiple refinement rounds (3 rounds default)
- Critique and support relationships
- Iterative convergence
- Rich graph visualization
- Weighted edge support

---

## 💡 Example Problems to Try

### Multi-Perspective Analysis
```
"Should remote work be mandatory, optional, or prohibited for our company?"
```

### Ethical Dilemma
```
"Is it ethical to use AI for hiring decisions?"
```

### Strategic Planning
```
"Should we prioritize rapid growth or sustainable profitability?"
```

### Product Decision
```
"Should our app be free with ads or paid without ads?"
```

### Policy Analysis
```
"Should cities invest more in public transit or road infrastructure?"
```

---

## 🎯 Key Concepts

### 1. Thought Nodes
Individual reasoning units representing perspectives:
```python
ThoughtNode(
    id="T1_ethical",
    content="From an ethical standpoint...",
    perspective="ethical",
    score=8.5
)
```

### 2. Edges (Relationships)
Connections between thoughts:
- **Support**: Thought A reinforces Thought B
- **Critique**: Thought A challenges Thought B
- **Build-on**: Thought A extends Thought B
- **Merge**: Thoughts should be combined

### 3. Graph Structure
NetworkX DiGraph enabling:
- Arbitrary connections (not just parent-child)
- Graph algorithms (PageRank, centrality)
- Community detection
- Path analysis

### 4. Aggregation
Synthesis using graph structure:
- **Centrality-based**: Use PageRank to find influential thoughts
- **Clustering-based**: Group related thoughts
- **Consensus-based**: Identify widely-supported ideas

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic GoT | Advanced GoT |
|---------|-----------|--------------|
| Structure | DAG | DAG with rounds |
| Perspectives | 4 fixed | 4 agent personas |
| Rounds | 1 | 3 (configurable) |
| Refinement | No | Yes, iterative |
| Edge Types | All types | Emphasis on critique/support |
| Visualization | Simple ASCII | Detailed round-by-round |
| Use Case | Quick analysis | Deep exploration |

**Recommendation**: Start with Basic for understanding, use Advanced for complex problems.

---

## 🔧 Customization Tips

### Add Your Own Perspectives

**Basic GoT:**
```python
perspectives = [
    "ethical",      # Moral considerations
    "practical",    # Feasibility and implementation
    "economic",     # Financial implications
    "technical",    # Technical constraints
    "legal",        # Regulatory/compliance
    "environmental" # Sustainability
]
```

**Advanced GoT:**
```python
agent_personas = [
    "optimist",     # Sees opportunities
    "critic",       # Identifies risks
    "pragmatist",   # Focuses on what's practical
    "innovator",    # Suggests creative solutions
    "analyst",      # Data-driven perspective
    "ethicist"      # Moral framework
]
```

### Adjust Iteration Rounds

In `got_advanced.py`:
```python
max_rounds: int = 5  # Increase for deeper exploration
```

### Modify Connection Threshold

Control how many edges are formed:
```python
connection_threshold: float = 0.3  # Lower = more connections
                                   # Higher = fewer, stronger connections
```

### Change Aggregation Strategy

In `got_basic.py`:
```python
# Option 1: PageRank (default)
pagerank = nx.pagerank(G)

# Option 2: Betweenness centrality (bridge thoughts)
centrality = nx.betweenness_centrality(G)

# Option 3: In-degree (most supported)
in_degree = dict(G.in_degree())
```

---

## ⚡ Common Issues & Solutions

### Issue: "No connections formed"
**Cause**: Connection threshold too high or thoughts too dissimilar
**Solution**: Lower `connection_threshold` to 0.2 or generate more diverse thoughts

### Issue: "Graph is too dense"
**Cause**: Too many connections making it hard to interpret
**Solution**: Increase `connection_threshold` to 0.5 or implement pruning

### Issue: "All thoughts have similar scores"
**Cause**: Evaluation criteria not discriminative enough
**Solution**: Use more specific evaluation prompts or multi-criteria scoring

### Issue: "Synthesis doesn't capture key insights"
**Cause**: Aggregation method not appropriate for graph structure
**Solution**: Try different aggregation strategies (centrality, clustering, consensus)

### Issue: "Takes too long to run"
**Cause**: Evaluating all possible connections (O(N²))
**Solution**: Reduce number of perspectives or use heuristics to pre-filter connections

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Implementation Code**: See `src/got_basic.py` and `src/got_advanced.py`
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ **Start**: Run basic example with simple problem
2. ✅ **Understand**: Observe how thoughts connect (not just parent-child)
3. ✅ **Visualize**: Study the ASCII graph output
4. ✅ **Experiment**: Try advanced example with multiple rounds
5. ✅ **Compare**: Run same problem with basic vs advanced
6. ✅ **Customize**: Add your own perspectives and problems
7. ✅ **Integrate**: Use GoT in your own applications

---

## 🌟 Pro Tips

### 1. Diverse Perspectives
The quality of GoT depends on perspective diversity. Include:
- **Conflicting viewpoints** (ethical vs. economic)
- **Different expertise** (technical vs. business)
- **Various stakeholders** (users, developers, executives)

### 2. Connection Evaluation
Good connections are:
- **Specific**: Clear relationship between thoughts
- **Relevant**: Meaningfully advance the reasoning
- **Weighted**: Stronger connections carry more influence

### 3. Graph Visualization
Always visualize your graph to:
- Identify central thoughts (high PageRank)
- Find bridge thoughts (high betweenness)
- Spot isolated thoughts (low connectivity)
- Validate relationships make sense

### 4. Iterative Refinement
Use multiple rounds when:
- Initial perspectives incomplete
- Conflicts need resolution
- Consensus needs to emerge
- Quality can be improved

### 5. Synthesis Quality
Good synthesis should:
- Reference multiple perspectives
- Acknowledge conflicts
- Build on connections
- Preserve key insights

### 6. Performance Optimization
For faster execution:
- Pre-filter potential connections with heuristics
- Use cheaper models for connection evaluation
- Parallelize independent evaluations
- Set maximum connection limits

---

## 🔍 Example Output Walkthrough

### Basic GoT Output:
```
Generating perspectives...
  T0_ethical: Privacy is a fundamental right...
  T1_practical: Implementation requires clear policies...
  T2_economic: Cost-benefit analysis shows...
  T3_legal: Regulatory compliance demands...

Forming connections...
  T0_ethical --supports--> T3_legal
  T1_practical --builds-on--> T2_economic
  T2_economic --conflicts--> T0_ethical

Evaluating thoughts...
  T0_ethical: 8.5/10
  T1_practical: 7.8/10
  T2_economic: 7.2/10
  T3_legal: 8.0/10

Graph Metrics:
  PageRank: T0=0.31, T3=0.28, T1=0.22, T2=0.19
  Edges: 3 connections formed

Synthesizing solution...
[Final integrated answer combining top perspectives]
```

### Advanced GoT Output:
```
============================================================
ROUND 1: Initial Proposals
============================================================

OPTIMIST: This opportunity could transform our business...
CRITIC: We must consider significant risks including...
PRAGMATIST: Realistic implementation requires...
INNOVATOR: What if we approached this completely differently...

============================================================
ROUND 2: Refinement and Consensus
============================================================

OPTIMIST: Building on pragmatist's point, we can...
CRITIC: While I appreciate innovator's creativity, we still face...
PRAGMATIST: Addressing critic's concerns, here's how...
INNOVATOR: Synthesizing these perspectives...

[Graph evolution visualization]

============================================================
FINAL SYNTHESIS
============================================================
[Comprehensive answer integrating all rounds]
```

---

## 🚦 When to Use Each Implementation

### Use Basic GoT When:
- ✅ Need quick multi-perspective analysis
- ✅ Problem is moderately complex
- ✅ Single round sufficient
- ✅ Want to understand GoT fundamentals
- ✅ Resource-constrained environment

### Use Advanced GoT When:
- ✅ Problem requires deep exploration
- ✅ Consensus needs to emerge
- ✅ Initial perspectives may be incomplete
- ✅ Critique and refinement important
- ✅ Quality more important than speed

---

## 📈 Next Steps

After mastering the basics:

1. **Compare with ToT**: Run same problem with Tree of Thoughts to see difference
2. **Experiment with Rounds**: Try 1, 3, 5 rounds and compare quality
3. **Custom Perspectives**: Define perspectives specific to your domain
4. **Edge Weighting**: Implement weighted voting for connections
5. **Visualization**: Export to Graphviz for professional visualizations
6. **Integration**: Use GoT for your real-world decision-making problems

---

**Happy Graph Building! 🕸️**

For questions or issues, refer to the full [README.md](./README.md).
