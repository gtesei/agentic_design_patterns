# Graph of Thoughts (GoT)

## Overview

The **Graph of Thoughts (GoT) Pattern** is an advanced reasoning framework that extends Tree of Thoughts by enabling arbitrary connections between thoughts, creating a directed acyclic graph (DAG) or even cyclic graph structure. Unlike tree-based approaches where thoughts can only reference their parent, GoT allows thoughts to reference, build upon, critique, or merge with any other thought in the graph—enabling non-linear reasoning, collaborative problem-solving, and iterative refinement.

GoT transforms LLMs from hierarchical reasoners into networked thinkers that can form complex relationships between ideas, synthesize multiple perspectives, and iteratively refine solutions through consensus and critique—much like how human teams collaborate on complex problems.

## Why Use This Pattern?

Traditional reasoning approaches have significant limitations:

- **Chain-of-Thought (CoT)**: Single linear path, no exploration of alternatives
- **Tree of Thoughts (ToT)**: Hierarchical structure limits connections to parent-child relationships
- **ReAct**: Sequential tool use without thought interconnection
- **Multi-Agent**: Agents communicate but thoughts aren't explicitly connected in a graph

Graph of Thoughts solves these by:
- **Non-hierarchical connections**: Any thought can reference or build on any other thought
- **Thought merging**: Combine insights from multiple reasoning paths
- **Cyclic refinement**: Thoughts can be iteratively improved through feedback loops
- **Multi-perspective synthesis**: Integrate diverse viewpoints into cohesive solutions
- **Collaborative reasoning**: Multiple agents or perspectives contribute to a shared graph
- **Flexible topology**: Not constrained to tree structure—allows DAGs or controlled cycles

### Example: Ethical Dilemma Analysis

```
Without GoT (Linear or Tree):
Question: "Should autonomous vehicles prioritize passenger safety or pedestrian safety?"

Linear: Passenger safety is more important → Answer provided
→ Misses nuanced perspectives and ethical considerations

Tree:
Root → Passenger safety (branch 1)
     → Pedestrian safety (branch 2)
     → Utilitarian view (branch 3)
→ Branches don't interact or synthesize

With GoT (Graph Structure):
Node 1 (Passenger): "Passengers trust the vehicle, implicit contract"
Node 2 (Pedestrian): "Pedestrians are vulnerable, didn't choose the risk"
Node 3 (Utilitarian): "Minimize total harm across all parties"
Node 4 (Legal): "Manufacturer liability creates economic pressure"

Edge 1→3: Node 1 supports utilitarian IF passenger count > pedestrian count
Edge 2→3: Node 2 supports utilitarian IF pedestrian count > passenger count
Edge 1→4: Passenger safety affects legal/insurance frameworks
Edge 2→4: Pedestrian protection influences regulation

Node 5 (Synthesis): Merges 1, 2, 3, 4
"Solution requires contextual decision-making:
- Default: Minimize total harm (utilitarian)
- Legal framework: Protect vulnerable road users
- Technical: Implement graduated response based on scenario
- Ethical: Transparency about decision algorithm"

Score: 9.2/10 (comprehensive, multi-perspective)
```

## How It Works

The Graph of Thoughts process consists of five key phases:

```
┌─────────────────────────────────────────────────────┐
│                  Initial Problem                     │
│                  (Seed Nodes)                        │
└────────────────┬────────────────────────────────────┘
                 ↓
         ┌───────────────┐
         │  1. GENERATE  │  Create initial thought nodes
         └───────┬───────┘
                 ↓
         ┌───────────────┐
         │  2. CONNECT   │  Form edges between related thoughts
         └───────┬───────┘
                 ↓
         ┌───────────────┐
         │  3. EVALUATE  │  Score nodes and edges
         └───────┬───────┘
                 ↓
         ┌───────────────┐
         │  4. AGGREGATE │  Synthesize insights across graph
         └───────┬───────┘
                 ↓
         ┌───────────────┐
         │  5. ITERATE   │  Refine through additional rounds
         └───────┬───────┘
                 ↓
         ┌───────────────┐
         │ Final Solution │
         └───────────────┘
```

### Phase 1: Generate Thought Nodes

Create diverse initial thoughts from multiple perspectives:
- **Diversity**: Generate thoughts representing different viewpoints, angles, or approaches
- **Independence**: Initial thoughts don't necessarily build on each other
- **Completeness**: Cover the problem space comprehensively

```python
def generate_initial_thoughts(
    problem: str,
    num_perspectives: int = 4
) -> list[ThoughtNode]:
    """Generate diverse initial thoughts"""
    prompt = f"""Given problem: {problem}
    Generate {num_perspectives} different perspectives or approaches.
    Each should represent a distinct viewpoint or angle."""
    thoughts = llm.generate(prompt)
    return [ThoughtNode(content=t, id=i) for i, t in enumerate(thoughts)]
```

### Phase 2: Connect Thoughts with Edges

Form relationships between thoughts:
- **Support**: One thought reinforces another
- **Critique**: One thought challenges another
- **Build-upon**: One thought extends another's idea
- **Merge**: Two thoughts can be combined
- **Prerequisite**: One thought depends on another

```python
def connect_thoughts(
    nodes: list[ThoughtNode],
    graph: nx.DiGraph
) -> nx.DiGraph:
    """Create edges between related thoughts"""
    for node_a in nodes:
        for node_b in nodes:
            if node_a.id != node_b.id:
                relationship = evaluate_relationship(node_a, node_b)
                if relationship.strength > threshold:
                    graph.add_edge(
                        node_a.id,
                        node_b.id,
                        type=relationship.type,
                        weight=relationship.strength
                    )
    return graph
```

### Phase 3: Evaluate Nodes and Edges

Score individual thoughts and their connections:
- **Node scores**: Quality, relevance, originality of each thought
- **Edge weights**: Strength and validity of connections
- **Centrality metrics**: Identify key thoughts in the graph

```python
def evaluate_graph(graph: nx.DiGraph) -> dict[str, float]:
    """Score nodes and edges in the graph"""
    scores = {}

    # Score individual nodes
    for node_id in graph.nodes:
        node = graph.nodes[node_id]['thought']
        scores[node_id] = evaluate_thought_quality(node)

    # Calculate graph centrality
    centrality = nx.pagerank(graph)

    # Combine scores
    final_scores = {
        nid: 0.7 * scores[nid] + 0.3 * centrality[nid]
        for nid in graph.nodes
    }
    return final_scores
```

### Phase 4: Aggregate Insights

Synthesize understanding across the graph:
- **Path aggregation**: Trace reasoning paths through connected thoughts
- **Cluster synthesis**: Group related thoughts and summarize
- **Consensus building**: Identify common themes and agreements
- **Conflict resolution**: Address contradictions and disagreements

```python
def aggregate_insights(
    graph: nx.DiGraph,
    scores: dict[str, float]
) -> str:
    """Synthesize insights from the thought graph"""
    # Find highly-scored, well-connected nodes
    key_nodes = get_top_nodes(graph, scores, k=5)

    # Trace relationships
    subgraph = extract_subgraph(graph, key_nodes)

    # Generate synthesis
    synthesis_prompt = f"""
    Key thoughts and their relationships:
    {format_subgraph(subgraph)}

    Synthesize these perspectives into a coherent answer.
    """
    return llm.generate(synthesis_prompt)
```

### Phase 5: Iterate and Refine

Optionally perform multiple rounds of refinement:
- **Critique cycle**: Generate critiques of existing thoughts as new nodes
- **Refinement**: Improve low-scoring nodes based on connections
- **Expansion**: Add new perspectives based on identified gaps
- **Convergence**: Continue until consensus emerges or max rounds reached

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Multi-perspective analysis**: Ethical dilemmas, policy decisions, strategic planning
- **Collaborative problem-solving**: Team brainstorming, research synthesis, design thinking
- **Debate and argumentation**: Analyzing multiple viewpoints, building logical arguments
- **Complex system analysis**: Understanding interconnected factors and dependencies
- **Iterative refinement**: Problems requiring consensus building or critique cycles
- **Knowledge synthesis**: Research literature review, combining multiple sources
- **Creative ideation**: Generating and combining creative concepts
- **Conflict resolution**: Analyzing disagreements and finding common ground

### ❌ When NOT to Use

- **Simple queries**: Questions with straightforward answers
- **Strict hierarchies**: Problems naturally suited to tree structures
- **Time-critical tasks**: GoT has higher overhead than simpler patterns
- **Well-defined sequences**: When steps must follow a specific order
- **Single perspective**: When only one viewpoint is needed
- **Limited resources**: Graph exploration can be computationally expensive

## Rule of Thumb

**Use Graph of Thoughts when:**
1. Problem requires **synthesizing multiple perspectives** or viewpoints
2. Thoughts need to **reference and build on each other** non-hierarchically
3. Solution benefits from **iterative refinement** and critique
4. Need to **identify consensus** or resolve conflicts
5. Exploring **complex interdependencies** between ideas
6. **Collaborative reasoning** from multiple agents or personas

**Don't use Graph of Thoughts when:**
1. Problem has clear hierarchical structure (use ToT instead)
2. Linear reasoning suffices (use CoT or ReAct)
3. Speed is critical over solution quality
4. Resources are constrained (simpler patterns more efficient)
5. Single authoritative answer is needed without debate

## Core Components

### 1. Thought Nodes

Individual units of reasoning in the graph:

```python
@dataclass
class ThoughtNode:
    id: str                    # Unique identifier
    content: str              # The thought text
    perspective: str          # Viewpoint/angle it represents
    score: float = 0.0       # Quality score
    metadata: dict = None    # Additional context

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
```

**Node properties:**
- **Independence**: Each thought is self-contained
- **Perspective**: Represents a specific viewpoint or angle
- **Scorable**: Can be evaluated on quality metrics
- **Connectible**: Can form relationships with other nodes

### 2. Thought Edges (Relationships)

Connections between thoughts representing relationships:

```python
@dataclass
class ThoughtEdge:
    source_id: str           # Source thought
    target_id: str           # Target thought
    relationship_type: str   # support, critique, build-on, merge
    weight: float           # Strength of relationship (0-1)
    description: str        # Explanation of relationship
```

**Edge types:**
- **Support**: Source reinforces target
- **Critique**: Source challenges target
- **Build-upon**: Source extends target's idea
- **Prerequisite**: Target depends on source
- **Merge**: Source and target should be combined
- **Conflict**: Source contradicts target

### 3. Graph Structure

Network representation of thoughts and relationships:

```python
import networkx as nx

class ThoughtGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph
        self.nodes: dict[str, ThoughtNode] = {}
        self.round: int = 0

    def add_thought(self, thought: ThoughtNode):
        """Add a thought node to the graph"""
        self.nodes[thought.id] = thought
        self.graph.add_node(thought.id, thought=thought)

    def add_relationship(self, edge: ThoughtEdge):
        """Add an edge between thoughts"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            type=edge.relationship_type,
            weight=edge.weight,
            description=edge.description
        )

    def get_influential_thoughts(self, k: int = 5) -> list[ThoughtNode]:
        """Find most influential nodes using PageRank"""
        scores = nx.pagerank(self.graph)
        top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.nodes[nid] for nid in top_ids]

    def get_connected_subgraph(self, node_ids: list[str]) -> nx.DiGraph:
        """Extract subgraph containing specified nodes and their connections"""
        return self.graph.subgraph(node_ids)
```

### 4. Aggregation Mechanisms

Methods to synthesize insights from the graph:

**Centrality-based:**
```python
def centrality_aggregation(graph: ThoughtGraph) -> str:
    """Aggregate using most central nodes"""
    # PageRank for influence
    pagerank = nx.pagerank(graph.graph)

    # Betweenness for bridging concepts
    betweenness = nx.betweenness_centrality(graph.graph)

    # Combine metrics
    combined = {
        nid: 0.6 * pagerank[nid] + 0.4 * betweenness[nid]
        for nid in graph.graph.nodes
    }

    top_nodes = sorted(combined, key=combined.get, reverse=True)[:5]
    return synthesize_nodes(graph, top_nodes)
```

**Clustering-based:**
```python
def cluster_aggregation(graph: ThoughtGraph) -> str:
    """Aggregate by identifying thought clusters"""
    # Convert to undirected for clustering
    undirected = graph.graph.to_undirected()

    # Find communities
    communities = nx.community.louvain_communities(undirected)

    # Synthesize each cluster
    cluster_summaries = []
    for community in communities:
        summary = summarize_cluster(graph, community)
        cluster_summaries.append(summary)

    # Combine cluster summaries
    return combine_summaries(cluster_summaries)
```

**Consensus-based:**
```python
def consensus_aggregation(graph: ThoughtGraph) -> str:
    """Find common ground across perspectives"""
    # Identify agree/support edges
    agreements = [
        (u, v) for u, v, d in graph.graph.edges(data=True)
        if d['type'] in ['support', 'merge']
    ]

    # Find thoughts with broad support
    support_counts = {}
    for _, target in agreements:
        support_counts[target] = support_counts.get(target, 0) + 1

    # High-consensus thoughts
    consensus_threshold = len(graph.nodes) * 0.4
    consensus_nodes = [
        nid for nid, count in support_counts.items()
        if count >= consensus_threshold
    ]

    return synthesize_consensus(graph, consensus_nodes)
```

### 5. Iterative Refinement

Multi-round improvement of the thought graph:

```python
def iterative_refinement(
    problem: str,
    max_rounds: int = 3,
    convergence_threshold: float = 0.95
) -> ThoughtGraph:
    """Refine graph over multiple rounds"""
    graph = ThoughtGraph()

    # Round 1: Initial generation
    initial_thoughts = generate_initial_thoughts(problem)
    for thought in initial_thoughts:
        graph.add_thought(thought)
    connect_thoughts(graph)

    # Subsequent rounds: Critique and refine
    for round_num in range(1, max_rounds):
        graph.round = round_num

        # Evaluate current state
        scores = evaluate_graph(graph)

        # Check convergence
        if check_convergence(scores, convergence_threshold):
            break

        # Generate critiques and refinements
        new_thoughts = generate_critiques(graph, scores)
        for thought in new_thoughts:
            graph.add_thought(thought)

        # Update connections
        connect_thoughts(graph)

    return graph
```

## Implementation Approaches

### Approach 1: Basic DAG with Single Round

Simplest implementation using directed acyclic graph:

```python
def graph_of_thoughts_basic(problem: str, num_perspectives: int = 4) -> str:
    """Basic GoT with single-round DAG construction"""

    # Initialize graph
    G = nx.DiGraph()

    # Step 1: Generate initial thoughts from different perspectives
    print("Generating perspectives...")
    perspectives = [
        "ethical", "practical", "economic", "legal"
    ][:num_perspectives]

    thoughts = {}
    for i, perspective in enumerate(perspectives):
        prompt = f"""
        Problem: {problem}

        Provide your analysis from a {perspective} perspective.
        Be specific and concise (2-3 sentences).
        """
        thought = llm.generate(prompt)
        thought_id = f"T{i}_{perspective}"

        thoughts[thought_id] = {
            'content': thought,
            'perspective': perspective,
            'score': 0.0
        }
        G.add_node(thought_id, **thoughts[thought_id])
        print(f"  {thought_id}: {thought[:80]}...")

    # Step 2: Connect related thoughts
    print("\nForming connections...")
    for tid_a in thoughts:
        for tid_b in thoughts:
            if tid_a != tid_b:
                relationship = evaluate_connection(
                    thoughts[tid_a]['content'],
                    thoughts[tid_b]['content']
                )
                if relationship['strength'] > 0.3:
                    G.add_edge(
                        tid_a, tid_b,
                        type=relationship['type'],
                        weight=relationship['strength']
                    )
                    print(f"  {tid_a} --{relationship['type']}--> {tid_b}")

    # Step 3: Evaluate nodes
    print("\nEvaluating thoughts...")
    for tid in thoughts:
        score = evaluate_thought(thoughts[tid]['content'], problem)
        G.nodes[tid]['score'] = score
        print(f"  {tid}: {score:.2f}/10")

    # Step 4: Aggregate insights
    print("\nSynthesizing solution...")
    # Use PageRank to find influential thoughts
    if len(G.edges) > 0:
        pagerank = nx.pagerank(G)
    else:
        pagerank = {tid: 1.0/len(thoughts) for tid in thoughts}

    # Combine top thoughts
    top_thoughts = sorted(
        pagerank.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    synthesis_prompt = f"""
    Problem: {problem}

    Key perspectives:
    {chr(10).join([f"- {thoughts[tid]['content']}" for tid, _ in top_thoughts])}

    Synthesize these perspectives into a comprehensive answer.
    """
    final_answer = llm.generate(synthesis_prompt)

    return final_answer
```

### Approach 2: Multi-Round Consensus Building

Advanced implementation with iterative refinement:

```python
def graph_of_thoughts_consensus(
    problem: str,
    num_agents: int = 4,
    max_rounds: int = 3
) -> tuple[str, ThoughtGraph]:
    """GoT with multiple rounds of consensus building"""

    graph = ThoughtGraph()
    agent_personas = ["optimist", "critic", "pragmatist", "innovator"][:num_agents]

    # Round 1: Initial proposals
    print(f"\n{'='*60}")
    print(f"ROUND 1: Initial Proposals")
    print(f"{'='*60}")

    for agent in agent_personas:
        thought = generate_perspective(problem, agent, [])
        node = ThoughtNode(
            id=f"R1_{agent}",
            content=thought,
            perspective=agent
        )
        graph.add_thought(node)
        print(f"\n{agent.upper()}: {thought}")

    # Form initial connections
    connect_all_thoughts(graph)

    # Subsequent rounds: Critique and refine
    for round_num in range(2, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}: Refinement and Consensus")
        print(f"{'='*60}")

        # Each agent reviews others' thoughts
        for agent in agent_personas:
            # Get all thoughts from previous rounds
            context = get_recent_thoughts(graph, round_num - 1)

            # Generate response (support, critique, or refinement)
            thought = generate_perspective(problem, agent, context)
            node = ThoughtNode(
                id=f"R{round_num}_{agent}",
                content=thought,
                perspective=agent
            )
            graph.add_thought(node)
            print(f"\n{agent.upper()}: {thought}")

            # Connect to relevant previous thoughts
            connect_to_previous(graph, node, context)

    # Final aggregation
    print(f"\n{'='*60}")
    print("FINAL SYNTHESIS")
    print(f"{'='*60}")

    final_answer = synthesize_graph(graph, problem)

    return final_answer, graph
```

### Approach 3: Weighted Edge with Voting

Implements voting mechanisms and edge weights:

```python
def graph_of_thoughts_voting(
    problem: str,
    num_perspectives: int = 5
) -> tuple[str, dict]:
    """GoT with voting mechanism for edge validation"""

    G = nx.DiGraph()

    # Generate diverse thoughts
    thoughts = generate_diverse_thoughts(problem, num_perspectives)
    for tid, thought in thoughts.items():
        G.add_node(tid, **thought)

    # Propose connections (any thought can propose connection to any other)
    proposed_edges = []
    for tid_a in thoughts:
        for tid_b in thoughts:
            if tid_a != tid_b:
                edge_proposal = propose_connection(
                    thoughts[tid_a],
                    thoughts[tid_b],
                    problem
                )
                if edge_proposal['valid']:
                    proposed_edges.append({
                        'source': tid_a,
                        'target': tid_b,
                        'type': edge_proposal['type'],
                        'weight': edge_proposal['weight'],
                        'votes': 1  # Self-vote
                    })

    # Voting phase: Each thought votes on proposed connections
    print("\nVoting on proposed connections...")
    for edge in proposed_edges:
        votes = 0
        for tid in thoughts:
            if tid not in [edge['source'], edge['target']]:
                vote = vote_on_connection(
                    thoughts[edge['source']],
                    thoughts[edge['target']],
                    edge['type'],
                    problem
                )
                if vote:
                    votes += 1
        edge['votes'] = votes

        # Add edge if it has sufficient support
        support_threshold = len(thoughts) * 0.3
        if votes >= support_threshold:
            G.add_edge(
                edge['source'],
                edge['target'],
                type=edge['type'],
                weight=edge['weight'],
                votes=votes
            )
            print(f"  ✓ {edge['source']} --{edge['type']}--> {edge['target']} ({votes} votes)")

    # Score thoughts based on graph structure
    scores = {}
    if len(G.edges) > 0:
        pagerank = nx.pagerank(G, weight='weight')
        in_degree = dict(G.in_degree(weight='votes'))

        for tid in thoughts:
            scores[tid] = {
                'influence': pagerank.get(tid, 0),
                'support': in_degree.get(tid, 0),
                'combined': 0.6 * pagerank.get(tid, 0) + 0.4 * (in_degree.get(tid, 0) / max(in_degree.values(), default=1))
            }

    # Synthesize using highest-scored thoughts
    top_thoughts = sorted(
        scores.items(),
        key=lambda x: x[1]['combined'],
        reverse=True
    )[:3]

    final_answer = synthesize_with_graph_context(
        [thoughts[tid] for tid, _ in top_thoughts],
        G,
        problem
    )

    return final_answer, {'graph': G, 'scores': scores, 'thoughts': thoughts}
```

## Key Benefits

### 🧠 Non-Linear Reasoning

- **Flexible connections**: Thoughts can reference any other thought, not just parent
- **Complex relationships**: Support, critique, merge, build-upon relationships
- **Holistic view**: See interconnections and dependencies across all ideas
- **Emergent insights**: New understanding emerges from thought interactions

### 🤝 Collaborative Problem-Solving

- **Multi-perspective**: Integrate diverse viewpoints naturally
- **Consensus building**: Identify common ground through graph structure
- **Conflict resolution**: Make disagreements explicit and addressable
- **Team reasoning**: Simulate collaborative thinking process

### 🔄 Iterative Refinement

- **Critique loops**: Thoughts can critique and improve each other
- **Convergence**: Multiple rounds lead to refined solutions
- **Adaptive**: Graph evolves based on evaluation feedback
- **Quality improvement**: Each round increases solution quality

### 📊 Rich Graph Analytics

- **Centrality metrics**: Identify key ideas (PageRank, betweenness)
- **Community detection**: Find clusters of related thoughts
- **Path analysis**: Trace reasoning chains through the graph
- **Influence tracking**: See which thoughts affect others most

### 🎯 Superior for Complex Problems

- **Multi-dimensional**: Handles problems with many interdependent factors
- **Synthesis**: Naturally combines multiple perspectives
- **Debate simulation**: Models argumentation and counterarguments
- **Research integration**: Combines insights from multiple sources

## Trade-offs

### ⚠️ Higher Complexity

**Issue**: More complex to implement and understand than tree structures

**Impact**:
- Harder to debug and visualize
- More cognitive overhead for developers
- Graph algorithms add complexity

**Mitigation**:
- Start with simple DAG before adding cycles
- Use graph visualization tools (networkx, graphviz)
- Implement comprehensive logging
- Clear documentation of edge types

### ⚠️ Computational Cost

**Issue**: Evaluating connections between N thoughts requires O(N²) comparisons

**Impact**:
- Significantly more LLM calls than ToT
- Higher token usage
- Longer latency

**Mitigation**:
- Limit number of initial thoughts (4-6)
- Use heuristics to pre-filter potential connections
- Parallelize connection evaluation
- Cache relationship evaluations
- Use cheaper models for relationship evaluation

### ⚠️ Cycle Management

**Issue**: Cyclic connections can create infinite loops or reasoning circles

**Impact**:
- Aggregation algorithms may not terminate
- Unclear how to synthesize cyclic reasoning

**Mitigation**:
- Use DAG instead of general graph (most common)
- If cycles needed, implement cycle detection
- Set maximum path length for aggregation
- Track visited nodes during traversal

### ⚠️ Aggregation Challenges

**Issue**: Synthesizing insights from complex graph is non-trivial

**Impact**:
- Multiple valid aggregation strategies
- Risk of losing important minority perspectives
- Difficulty in conflict resolution

**Mitigation**:
- Use established graph algorithms (PageRank, centrality)
- Implement multiple aggregation strategies
- Include human-in-the-loop for critical decisions
- Preserve dissenting views in final output

## Best Practices

### 1. Perspective Generation

```python
# ❌ BAD: Homogeneous thoughts
prompt = "Give me ideas about X"
# All thoughts may be similar

# ✅ GOOD: Explicit diverse perspectives
perspectives = ["ethical", "practical", "economic", "technical"]
for p in perspectives:
    prompt = f"Analyze problem X from a {p} perspective"
```

### 2. Edge Type Definition

```python
# Define clear edge types
class EdgeType(Enum):
    SUPPORT = "supports"      # A reinforces B
    CRITIQUE = "critiques"    # A challenges B
    BUILD_ON = "builds-on"    # A extends B
    PREREQ = "requires"       # A depends on B
    MERGE = "merges-with"     # A and B should combine
    CONFLICT = "conflicts"    # A contradicts B

# Be explicit about relationships
edge = ThoughtEdge(
    source_id="T1",
    target_id="T2",
    relationship_type=EdgeType.SUPPORT,
    weight=0.8,
    description="Economic analysis supports ethical conclusion"
)
```

### 3. Graph Pruning

```python
def prune_weak_connections(graph: nx.DiGraph, threshold: float = 0.4):
    """Remove low-weight edges to simplify graph"""
    edges_to_remove = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d['weight'] < threshold
    ]
    graph.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated = [n for n in graph.nodes if graph.degree(n) == 0]
    graph.remove_nodes_from(isolated)
```

### 4. Visualization

```python
def visualize_graph(graph: ThoughtGraph, output_path: str = None):
    """Create visual representation of thought graph"""

    # ASCII visualization for console
    print("\nThought Graph Structure:")
    print("="*60)

    for node_id in graph.graph.nodes:
        node = graph.nodes[node_id]
        print(f"\n[{node_id}] {node.perspective.upper()}")
        print(f"  {node.content[:100]}...")

        # Show outgoing edges
        edges = graph.graph.out_edges(node_id, data=True)
        if edges:
            print("  Connections:")
            for _, target, data in edges:
                print(f"    → {target} ({data['type']}, weight={data['weight']:.2f})")

    # Optional: matplotlib visualization
    if output_path:
        pos = nx.spring_layout(graph.graph)
        nx.draw(graph.graph, pos, with_labels=True, node_color='lightblue')
        plt.savefig(output_path)
```

### 5. Consensus Detection

```python
def detect_consensus(graph: ThoughtGraph, threshold: float = 0.7) -> list[str]:
    """Find thoughts with broad support"""

    consensus_nodes = []

    for node_id in graph.graph.nodes:
        # Count supporting connections
        support_edges = [
            (u, v, d) for u, v, d in graph.graph.edges(data=True)
            if v == node_id and d['type'] in ['support', 'merge']
        ]

        # Calculate consensus score
        support_ratio = len(support_edges) / max(len(graph.nodes) - 1, 1)
        avg_weight = sum(d['weight'] for _, _, d in support_edges) / max(len(support_edges), 1)

        consensus_score = support_ratio * avg_weight

        if consensus_score >= threshold:
            consensus_nodes.append(node_id)
            print(f"  Consensus found: {node_id} (score={consensus_score:.2f})")

    return consensus_nodes
```

## Performance Metrics

Track these metrics to optimize GoT performance:

### Effectiveness Metrics
- **Solution quality**: Comprehensiveness and coherence of final answer
- **Perspective diversity**: How well different viewpoints are represented
- **Consensus level**: Degree of agreement among thoughts
- **Synthesis quality**: How well perspectives are integrated
- **Conflict resolution**: Success in addressing contradictions

### Graph Metrics
- **Node count**: Number of thoughts generated
- **Edge count**: Number of connections formed
- **Edge density**: Ratio of actual to possible edges
- **Average path length**: Typical distance between thoughts
- **Clustering coefficient**: How well thoughts cluster
- **Centralization**: How dominated by key thoughts

### Efficiency Metrics
- **LLM calls**: Total generation + evaluation calls
- **Rounds to convergence**: How many iterations needed
- **Token usage**: Total input + output tokens
- **Time to solution**: Latency from start to answer
- **Cost per problem**: Dollar cost of solving

### Quality Metrics
- **Inter-thought coherence**: How well thoughts relate
- **Edge validity**: Accuracy of proposed connections
- **Aggregation quality**: How well synthesis captures graph
- **Minority view preservation**: Are dissenting views included

## Example Scenarios

### Scenario 1: Product Design Decision

```
Problem: "Should our new smart thermostat prioritize energy efficiency or user comfort?"

Initial Thoughts (Round 1):
┌────────────────────────────────────────────────────────┐
│ T1_Environmental: "Prioritize efficiency - climate     │
│ crisis demands we reduce energy consumption even if    │
│ it means minor comfort trade-offs"                     │
│ Score: 7.5/10                                          │
└────────────────────────────────────────────────────────┘
                    ↓ supports
┌────────────────────────────────────────────────────────┐
│ T2_Economic: "Energy savings translate to lower bills │
│ which is our main value proposition. Efficiency wins" │
│ Score: 8.0/10                                          │
└────────────────────────────────────────────────────────┘
                    ↑ critiques
┌────────────────────────────────────────────────────────┐
│ T3_UserExperience: "Users won't accept discomfort.    │
│ Product will fail if home feels cold. Comfort first"  │
│ Score: 8.5/10                                          │
└────────────────────────────────────────────────────────┘
                    ↓ builds-on
┌────────────────────────────────────────────────────────┐
│ T4_Technical: "AI can optimize for both by learning   │
│ user patterns and pre-heating/cooling efficiently"    │
│ Score: 9.0/10                                          │
└────────────────────────────────────────────────────────┘

Round 2: Refinements
┌────────────────────────────────────────────────────────┐
│ T5_Synthesis: "Implement smart scheduling: aggressive │
│ efficiency when home is empty, prioritize comfort     │
│ when occupied. Learn user preferences over time"      │
│ Score: 9.5/10                                          │
│                                                        │
│ Connections:                                           │
│ ← T1 (supports): Efficiency during empty periods      │
│ ← T3 (supports): Comfort when users present           │
│ ← T4 (builds-on): Uses AI learning approach           │
└────────────────────────────────────────────────────────┘

Graph Analysis:
- Node T5 has highest PageRank (0.35)
- T4 and T5 form consensus cluster
- T3 acts as bridge between efficiency and comfort camps

Final Solution:
"Implement adaptive scheduling that optimizes for energy efficiency during
unoccupied periods while ensuring comfort during occupied times. Use machine
learning to predict occupancy patterns and user preferences. Provide users
with transparency (energy savings dashboard) and control (adjust efficiency/
comfort balance). This approach satisfies both environmental responsibility
(T1), economic value (T2), user satisfaction (T3), and technical feasibility (T4)."
```

### Scenario 2: Ethical Dilemma - Data Privacy

```
Problem: "Should social media platforms scan private messages to detect illegal content?"

Initial Graph:
        [T1_Safety]
            ↓ supports
        [T2_Legal] ←─ conflicts ─→ [T3_Privacy]
            ↑ requires                    ↑
            │                             │
        [T4_Technical] ─── builds-on ────┘

T1_Safety (Law Enforcement Perspective):
"Scanning is essential for child protection and preventing terrorism.
Moral imperative to use available tools to stop crime."
Score: 7.0/10, Connections: 2

T2_Legal (Regulatory Perspective):
"Legal frameworks in many jurisdictions mandate platforms to detect
illegal content. Failure to comply brings liability."
Score: 7.5/10, Connections: 3

T3_Privacy (Civil Liberties Perspective):
"Private communications are fundamental rights. Mass surveillance
chills free speech and enables authoritarian abuse."
Score: 8.5/10, Connections: 3

T4_Technical (Security Expert Perspective):
"End-to-end encryption is mathematically incompatible with content
scanning. Backdoors undermine security for everyone."
Score: 9.0/10, Connections: 3

Round 2: Refinements
T5_Alternative (Policy Perspective):
"Instead of mass scanning, use: (1) user reporting tools,
(2) metadata analysis, (3) warrant-based targeted access."
Connections:
  ← T3 (supports): Preserves privacy
  ← T1 (partially supports): Still enables some safety measures
  ← T4 (compatible with): Maintains encryption
  → T2 (critiques): May not satisfy all legal requirements

T6_Transparency (Governance Perspective):
"Any scanning must be: transparent about what's scanned, subject
to oversight, limited in scope, with strong error appeal process."
Connections:
  ← T3 (mitigates): Adds safeguards to privacy concerns
  ← T2 (supports): Provides legal accountability
  ← T1 (enables): Allows safety measures with controls

Consensus Analysis:
- High-consensus nodes: T4 (technical constraints), T3 (privacy value)
- Bridge nodes: T6 (connects opposing camps)
- Conflict edges: T2 ↔ T3 (legal vs. privacy)

Final Synthesis:
"Recommendation: No mass scanning of private encrypted messages.
Instead, implement:
1. Strong user reporting tools (addresses T1 safety)
2. Metadata analysis where legally permitted (addresses T2)
3. Maintain end-to-end encryption (addresses T3 privacy, T4 technical)
4. Warrant-based targeted access with judicial oversight (balances T1, T3)
5. Transparency reports on requests and appeals (addresses T6 governance)

This approach respects technical reality (T4), prioritizes privacy (T3),
provides some safety mechanisms (T1), and establishes legal frameworks (T2)
with proper oversight (T6). Acknowledges this won't satisfy all T2 legal
requirements - may need advocacy for law changes."
```

### Scenario 3: Strategic Planning - Market Entry

```
Problem: "Should we enter the European market now or wait 2 years?"

Round 1: Initial Analysis
[T1_Market] (Market Analysis):
"European demand is growing 30% annually. First-mover advantage is significant."
Score: 7.5/10

[T2_Finance] (Financial Analysis):
"Entry costs are €2M. Break-even in 18 months if projections hold."
Score: 7.0/10

[T3_Operations] (Operational Readiness):
"Our team lacks EU regulatory expertise. Supply chain not established."
Score: 8.0/10

[T4_Competition] (Competitive Analysis):
"Two competitors planning entry in 12 months. Window is closing."
Score: 8.5/10

Graph Connections:
T1 → T4 (supports): Market growth attracts competition
T2 → T3 (conflicts): Finance says go, Operations says not ready
T4 → T1 (builds-on): Competition validates market opportunity

Round 2: Deep Dives
[T5_Risk] (Risk Analysis):
"Delayed entry means: (1) lost revenue €3M over 2 years,
(2) competitor lock-in, (3) higher future entry costs.
Early entry risks: (1) regulatory mistakes, (2) operational failures."
Connections:
  ← T1, T4 (analyzes implications)
  ← T2, T3 (quantifies trade-offs)

[T6_Hybrid] (Alternative Strategy):
"Establish legal entity and begin regulatory approval NOW.
Soft launch in 12 months, full operations in 18 months.
Hire EU expert as consultant immediately."
Connections:
  ← T3 (addresses): Builds operational readiness
  ← T2 (compatible): Phases investment
  ← T4 (responds to): Moves before competitors
  ← T1 (captures): Gets into growing market

Round 3: Validation
[T7_Validation] (Scenario Analysis):
"Ran 3 scenarios:
- Early entry (now): 60% success, high risk/reward
- Delayed entry (2yr): 35% success, competitors entrenched
- Phased entry (6mo): 75% success, balanced approach"
Connections:
  ← T6 (validates): Phased approach has best odds
  ← T5 (quantifies): Risk/reward analysis confirms

PageRank Analysis:
1. T6_Hybrid: 0.28 (highest influence)
2. T5_Risk: 0.21
3. T4_Competition: 0.18
4. T7_Validation: 0.15

Consensus: Nodes T5, T6, T7 form strong cluster with mutual support.

Final Decision:
"Recommend Phased European Entry:

Phase 1 (Immediate):
- Hire EU regulatory consultant (addresses T3)
- Begin legal entity setup
- Initial market research and partner outreach

Phase 2 (Months 6-12):
- Complete regulatory approvals
- Soft launch with limited customers
- Build operational capabilities

Phase 3 (Months 12-18):
- Full market entry
- Scale operations
- Competitive positioning

Rationale:
- Captures market opportunity (T1) before competitors (T4)
- Manages financial risk through phasing (T2)
- Builds operational readiness systematically (T3)
- Validated by scenario analysis (T7) as highest probability approach
- Balances first-mover advantage against execution risk (T5)"
```

## Advanced Patterns

### 1. Thought Merging

Combine complementary thoughts into unified nodes:

```python
def merge_compatible_thoughts(
    graph: ThoughtGraph,
    similarity_threshold: float = 0.8
) -> ThoughtGraph:
    """Merge similar thoughts to reduce graph complexity"""

    # Find merge candidates
    merge_pairs = []
    for nid_a in graph.nodes:
        for nid_b in graph.nodes:
            if nid_a < nid_b:  # Avoid duplicates
                similarity = compute_similarity(
                    graph.nodes[nid_a].content,
                    graph.nodes[nid_b].content
                )
                if similarity > similarity_threshold:
                    merge_pairs.append((nid_a, nid_b, similarity))

    # Merge thoughts
    for nid_a, nid_b, sim in merge_pairs:
        merged_content = llm.generate(f"""
        Merge these complementary thoughts into one:

        Thought A: {graph.nodes[nid_a].content}
        Thought B: {graph.nodes[nid_b].content}

        Create a unified thought that captures both perspectives.
        """)

        merged_id = f"M_{nid_a}_{nid_b}"
        merged_node = ThoughtNode(
            id=merged_id,
            content=merged_content,
            perspective="merged"
        )

        # Add merged node and reconnect edges
        graph.add_thought(merged_node)

        # Transfer edges
        for pred in graph.graph.predecessors(nid_a):
            edge_data = graph.graph[pred][nid_a]
            graph.graph.add_edge(pred, merged_id, **edge_data)
        for pred in graph.graph.predecessors(nid_b):
            edge_data = graph.graph[pred][nid_b]
            graph.graph.add_edge(pred, merged_id, **edge_data)

        # Remove original nodes
        graph.graph.remove_node(nid_a)
        graph.graph.remove_node(nid_b)
        del graph.nodes[nid_a]
        del graph.nodes[nid_b]

    return graph
```

### 2. Subgraph Extraction

Focus on most relevant portions of large graphs:

```python
def extract_relevant_subgraph(
    graph: ThoughtGraph,
    query: str,
    k: int = 5
) -> nx.DiGraph:
    """Extract most relevant subgraph for a query"""

    # Score nodes by relevance to query
    relevance_scores = {}
    for nid, node in graph.nodes.items():
        relevance_scores[nid] = compute_relevance(node.content, query)

    # Get top-k most relevant nodes
    top_nodes = sorted(
        relevance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]
    top_node_ids = [nid for nid, _ in top_nodes]

    # Expand to include immediate neighbors
    expanded_nodes = set(top_node_ids)
    for nid in top_node_ids:
        expanded_nodes.update(graph.graph.predecessors(nid))
        expanded_nodes.update(graph.graph.successors(nid))

    # Extract subgraph
    subgraph = graph.graph.subgraph(expanded_nodes).copy()

    return subgraph
```

### 3. Debate Simulation

Model argumentation through graph structure:

```python
def simulate_debate(
    topic: str,
    positions: list[str],
    rounds: int = 3
) -> ThoughtGraph:
    """Simulate multi-round debate with arguments and counterarguments"""

    graph = ThoughtGraph()

    # Round 1: Opening statements
    for pos in positions:
        opening = generate_opening_statement(topic, pos)
        node = ThoughtNode(
            id=f"R1_{pos}",
            content=opening,
            perspective=pos
        )
        graph.add_thought(node)

    # Subsequent rounds: Arguments and counterarguments
    for round_num in range(2, rounds + 1):
        for pos in positions:
            # Review opposing arguments
            opposing = [
                graph.nodes[nid]
                for nid in graph.graph.nodes
                if graph.nodes[nid].perspective != pos
            ]

            # Generate counterarguments
            counter = generate_counterargument(topic, pos, opposing)
            node = ThoughtNode(
                id=f"R{round_num}_{pos}",
                content=counter,
                perspective=pos
            )
            graph.add_thought(node)

            # Link as critique to opposing arguments
            for opp_node in opposing:
                edge = ThoughtEdge(
                    source_id=node.id,
                    target_id=opp_node.id,
                    relationship_type="critique",
                    weight=0.8,
                    description=f"{pos} critiques {opp_node.perspective}"
                )
                graph.add_relationship(edge)

    return graph
```

### 4. Confidence Propagation

Propagate confidence scores through graph:

```python
def propagate_confidence(graph: ThoughtGraph) -> dict[str, float]:
    """Propagate confidence through graph edges"""

    confidence = {}

    # Initialize with base scores
    for nid, node in graph.nodes.items():
        confidence[nid] = node.score / 10.0  # Normalize to [0, 1]

    # Iteratively update based on connections
    for _ in range(10):  # Fixed iterations
        new_confidence = confidence.copy()

        for nid in graph.graph.nodes:
            # Gather confidence from supporters
            supporting_edges = [
                (u, v, d) for u, v, d in graph.graph.edges(data=True)
                if v == nid and d['type'] == 'support'
            ]

            if supporting_edges:
                support_conf = sum(
                    confidence[u] * d['weight']
                    for u, _, d in supporting_edges
                )
                # Update: 70% own, 30% from supporters
                new_confidence[nid] = 0.7 * confidence[nid] + 0.3 * support_conf

            # Decrease confidence from critiques
            critique_edges = [
                (u, v, d) for u, v, d in graph.graph.edges(data=True)
                if v == nid and d['type'] == 'critique'
            ]

            if critique_edges:
                critique_strength = sum(
                    confidence[u] * d['weight']
                    for u, _, d in critique_edges
                )
                new_confidence[nid] *= (1 - 0.2 * critique_strength)

        confidence = new_confidence

    return confidence
```

### 5. Graph Compression

Reduce graph complexity while preserving key insights:

```python
def compress_graph(
    graph: ThoughtGraph,
    target_nodes: int = 10
) -> ThoughtGraph:
    """Compress graph to target number of nodes"""

    compressed = ThoughtGraph()

    # Use hierarchical clustering
    # Convert graph to feature vectors
    embeddings = {
        nid: get_embedding(node.content)
        for nid, node in graph.nodes.items()
    }

    # Cluster nodes
    clusters = hierarchical_clustering(
        embeddings,
        n_clusters=target_nodes
    )

    # Create summary node for each cluster
    for cluster_id, node_ids in clusters.items():
        cluster_thoughts = [
            graph.nodes[nid].content
            for nid in node_ids
        ]

        summary = llm.generate(f"""
        Summarize these related thoughts into one:
        {chr(10).join(f"- {t}" for t in cluster_thoughts)}
        """)

        summary_node = ThoughtNode(
            id=f"C{cluster_id}",
            content=summary,
            perspective="synthesized"
        )
        compressed.add_thought(summary_node)

    # Reconstruct edges between cluster representatives
    for c1 in clusters:
        for c2 in clusters:
            if c1 != c2:
                # Check if original nodes had connections
                has_connection = any(
                    graph.graph.has_edge(n1, n2)
                    for n1 in clusters[c1]
                    for n2 in clusters[c2]
                )

                if has_connection:
                    # Create edge between cluster summaries
                    compressed.graph.add_edge(f"C{c1}", f"C{c2}")

    return compressed
```

## Comparison with Related Patterns

| Pattern | Structure | Connections | Refinement | Best For |
|---------|-----------|-------------|------------|----------|
| **Graph of Thoughts** | DAG or graph | Any-to-any | Iterative cycles | Multi-perspective synthesis |
| **Tree of Thoughts** | Tree | Parent-child only | Backtracking | Hierarchical exploration |
| **Chain of Thought** | Linear | Sequential | None | Step-by-step reasoning |
| **ReAct** | Linear | Tool-mediated | Observation-based | External tool use |
| **Multi-Agent** | Network | Message-passing | Agent interaction | Distributed problem-solving |
| **Self-Consistency** | Parallel samples | None | Voting | Uncertainty reduction |

### When to Choose GoT

- **vs ToT**: When thoughts need to reference non-parent thoughts, or when you need synthesis across branches
- **vs CoT**: When problem requires multiple perspectives that interact, not linear reasoning
- **vs ReAct**: When reasoning about internal ideas, not external tool use
- **vs Multi-Agent**: When you want explicit graph of thought relationships, not just agent communication
- **vs Self-Consistency**: When you need thoughts to build on each other, not independent samples

## Common Pitfalls

### 1. Over-Connected Graphs

**Problem**: Evaluating all possible edges creates O(N²) connections

**Symptoms**: Slow performance, cluttered graphs, high costs

**Solution**:
```python
# Use heuristics to pre-filter connections
def should_evaluate_connection(node_a: ThoughtNode, node_b: ThoughtNode) -> bool:
    # Only evaluate if perspectives are different
    if node_a.perspective == node_b.perspective:
        return False

    # Only evaluate if content has sufficient overlap
    similarity = quick_similarity(node_a.content, node_b.content)
    if similarity < 0.2 or similarity > 0.9:
        return False  # Too different or too similar

    return True
```

### 2. Weak Aggregation

**Problem**: Synthesis doesn't effectively capture graph insights

**Symptoms**: Final answer misses key perspectives, feels disconnected

**Solution**:
- Use multiple aggregation methods and compare
- Include graph structure in synthesis prompt
- Preserve minority views explicitly
- Validate synthesis covers all major clusters

### 3. Unbalanced Perspectives

**Problem**: Some perspectives dominate the graph

**Symptoms**: High centrality for certain viewpoints, others marginalized

**Solution**:
```python
def balance_perspectives(graph: ThoughtGraph) -> dict[str, float]:
    """Ensure all perspectives are fairly represented"""

    # Count nodes per perspective
    perspective_counts = {}
    for node in graph.nodes.values():
        perspective_counts[node.perspective] = \
            perspective_counts.get(node.perspective, 0) + 1

    # Apply balancing weights
    balance_weights = {}
    max_count = max(perspective_counts.values())
    for perspective, count in perspective_counts.items():
        balance_weights[perspective] = max_count / count

    # Apply to node scores
    for nid, node in graph.nodes.items():
        node.score *= balance_weights[node.perspective]

    return balance_weights
```

### 4. Ignoring Conflicts

**Problem**: Contradictions in graph are not addressed

**Symptoms**: Final answer contains logical contradictions

**Solution**:
- Explicitly identify conflict edges
- Include conflict resolution in synthesis
- Acknowledge unresolved disagreements
- Present multiple valid perspectives if needed

### 5. Poor Visualization

**Problem**: Graph is hard to understand and interpret

**Symptoms**: Can't debug reasoning, unclear why certain paths chosen

**Solution**:
- Implement clear console output with ASCII art
- Use colors for different edge types
- Show graph metrics (centrality, clusters)
- Provide path tracing functionality
- Export to graph visualization tools (Graphviz, Gephi)

## Conclusion

The Graph of Thoughts pattern represents the cutting edge of LLM reasoning, enabling sophisticated multi-perspective analysis and collaborative problem-solving that goes beyond hierarchical tree structures. By allowing arbitrary connections between thoughts, GoT enables synthesis, debate, and iterative refinement that mirrors how human teams approach complex problems.

**Use Graph of Thoughts when:**
- Problem requires synthesizing multiple perspectives or viewpoints
- Thoughts need to reference and build on each other non-hierarchically
- Solution benefits from iterative refinement and critique
- Need to identify consensus or resolve conflicts
- Exploring complex interdependencies between ideas
- Simulating collaborative or adversarial reasoning

**Implementation checklist:**
- ✅ Define clear perspectives or viewpoints to explore
- ✅ Implement thought generation with diversity
- ✅ Define explicit edge types (support, critique, build-on, etc.)
- ✅ Use graph algorithms for evaluation (PageRank, centrality)
- ✅ Implement multiple aggregation strategies
- ✅ Prune weak connections to manage complexity
- ✅ Visualize graph structure clearly
- ✅ Set iteration limits for refinement rounds
- ✅ Preserve minority perspectives in synthesis
- ✅ Validate final answer captures key insights

**Key Takeaways:**
- 🕸️ GoT uses graph structure, not limited to trees
- 🔗 Any thought can connect to any other thought
- 🤝 Ideal for multi-perspective synthesis and consensus
- 🔄 Supports iterative refinement through cycles
- 📊 Uses graph algorithms (PageRank, clustering) for analysis
- ⚖️ Trade-off: Rich synthesis vs. higher complexity and cost
- 🎯 Superior for problems requiring diverse viewpoint integration
- 🛠️ NetworkX provides excellent graph support

**Performance Guidelines:**
- **Simple problems**: 3-4 perspectives, single round, DAG structure
- **Moderate problems**: 4-6 perspectives, 2 rounds, weighted edges
- **Complex problems**: 6-8 perspectives, 3 rounds, full graph analytics
- **Cost management**: Use heuristics to pre-filter connections, limit iterations

---

*Graph of Thoughts transforms LLMs from hierarchical thinkers into networked reasoners capable of synthesizing diverse perspectives, building consensus, and iteratively refining solutions—enabling AI to tackle the most complex, multi-faceted problems that require collaborative intelligence.*
