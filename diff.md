# Pattern Differences

Several patterns in this catalog overlap visibly in their demos and can blur together on first read. This document **disambiguates the commonly-conflated pairs**: what they share, the actual axis of difference, and how to choose between them in practice.

Where a useful framing exists: **one pattern is the *mechanism*, the other is the *shape* built on top of it** — or — **they look similar but operate on different *axes*** (context vs goal, persistence vs window, tree vs graph).

---

## Index

- [ReAct vs Tool Use](#react-vs-tool-use) — *loop shape* vs *single-turn primitive*
- [Subagents vs Parallelization](#subagents-vs-parallelization) — *isolated contexts* vs *one-context concurrency*
- [Tree of Thoughts vs Graph of Thoughts](#tree-of-thoughts-vs-graph-of-thoughts) — *branches don't merge* vs *branches recombine*
- [Deep Research vs ReAct](#deep-research-vs-react) — *typed research pipeline* vs *generic act-observe loop*
- [Goal Management vs Resource Optimization](#goal-management-vs-resource-optimization) — *what to do* vs *how much capacity you have*
- [Memory Management vs Context Management](#memory-management-vs-context-management) — *across sessions* vs *within this turn*

---

## ReAct vs Tool Use

[ReAct →](./foundational_design_patterns/8_react/)  ·  [Tool Use →](./foundational_design_patterns/5_tool_use/)

**One-liner.** Tool Use is the *primitive*; ReAct is the *loop pattern* built on top of it.

**What they share.** Both involve the model calling out to external functions and consuming results.

**Axis of difference.**

| Aspect          | Tool Use                                              | ReAct                                                                  |
|---|---|---|
| Granularity     | A single turn: emit a call → execute → return result | A multi-turn loop: Reason → Act → Observe → repeat                     |
| What it defines | The *mechanism* by which a model invokes a function   | The *shape* of an agent that keeps invoking functions until done       |
| Stop condition  | n/a — one call per turn                                | Agent decides "no more actions needed" (or hits an iteration cap)      |
| Typical demo    | "Model calls `get_weather(location)` and replies"      | "Model writes a thought, calls a search, reads the result, writes another thought, calls another tool…" |
| Maps onto       | OpenAI function calling, Anthropic tool use, MCP tools | Any agent loop that alternates assistant text + tool calls             |

**How to choose.** If your task is "model needs to call one function and answer," it's Tool Use. If your task is "model needs to think, act, observe, and keep going until it solves the problem," it's ReAct. **Modern agent loops are almost always implicit ReAct loops driven by tool calling** — the two are intertwined, not competing.

**Common confusion.** "Doesn't every framework with `tools=[...]` already do ReAct?" Yes, *implicitly*. Many frameworks bundle them. Tool Use is the lower-level capability; ReAct is the conventional way it gets used at scale.

---

## Subagents vs Parallelization

[Subagents →](./orchestration/subagents/)  ·  [Parallelization →](./foundational_design_patterns/3_parallelization/)

**One-liner.** Parallelization fans work out **within one context**; subagents fan work out **across isolated contexts**.

**What they share.** Both produce N results from N concurrent operations and synthesize them.

**Axis of difference: who owns the context window.**

| Aspect              | Parallelization                                     | Subagents                                                    |
|---|---|---|
| Context             | Single — all N operations run in the same agent's prompt scope | One per worker — each subagent has its own isolated context |
| Isolation level     | Message-array isolation at best                      | Process / container / session isolation (much stronger)     |
| Result interface    | All raw outputs flow back into the parent prompt    | Only a **structured summary** crosses back; raw work stays in the worker |
| Context cost        | Sum of all N outputs hits the parent context        | Workers absorb their own reads/thoughts; parent stays lean |
| Latency overhead    | Minimal — concurrent function calls                  | Per-worker spawn cost (~100ms+ for separate processes)     |
| Failure blast radius | One bad output pollutes the parent transcript       | One worker failing doesn't corrupt the parent's reasoning  |

**How to choose.**

- Need to call 5 search APIs concurrently and synthesize? **Parallelization.**
- Need 5 workers to each summarize a different 50-file directory and return a paragraph each? **Subagents** — you'd blow the parent's context if you tried this with parallelization.

**Common confusion.** "Subagents are just parallelization with extra steps." False. The structural commitment subagents make — **summary-not-transcript** — is what makes them scale. Parallelization with 5 outputs of 20k tokens each is unworkable; subagents with 5 outputs of 200 tokens each fits.

---

## Tree of Thoughts vs Graph of Thoughts

[Tree of Thoughts →](./reasoning/tree_of_thoughts/)  ·  [Graph of Thoughts →](./reasoning/graph_of_thoughts/)

**One-liner.** Both explore many partial solutions and score them. **ToT keeps the explorations as a tree; GoT lets them merge into a graph.**

**What they share.** Candidate generation + state scoring + pruning of bad branches. Both come from a research lineage that treats reasoning as classical search over thought states.

**Axis of difference: topology of the reasoning state.**

| Aspect                  | Tree of Thoughts                                | Graph of Thoughts                                              |
|---|---|---|
| State topology          | Rooted tree — each node has one parent          | DAG / graph — nodes can have multiple parents (merges allowed) |
| Operations              | Expand, score, prune, backtrack (classical search) | Above + **aggregate** (combine multiple partial results) and **refine** (improve via merge) |
| Algorithmic analogue    | BFS / DFS / beam search                          | Evolutionary / genetic operators + search                      |
| Best when               | Solution space decomposes cleanly into mutually-exclusive paths | Subproblems share structure or partial solutions can be productively recombined |
| Example                 | "Game of 24" — each branch is a distinct arithmetic path | "Summarize many documents" — sub-summaries can be merged pairwise |

**How to choose.** If branches in your problem are *genuinely independent* — try one, fail, try another — **ToT** is enough and simpler. If partial solutions *combine into better partial solutions* — sub-summaries that aggregate, sub-proofs that compose — you want **GoT**'s merge nodes.

**Common confusion.** "ToT branches must eventually merge anyway when the agent picks one." No — ToT's *winning path* is selected from the tree, but other branches are *discarded*. GoT explicitly **recombines** branches rather than picking one.

---

## Deep Research vs ReAct

[Deep Research →](./reasoning/deep_research/)  ·  [ReAct →](./foundational_design_patterns/8_react/)

**One-liner.** ReAct is a generic act-observe loop the agent steers freely; Deep Research is a **typed pipeline with prescribed phases and citation semantics**.

**What they share.** Both are multi-round loops where the agent gathers information and then produces output.

**Axis of difference: prescribed structure vs free-form.**

| Aspect              | ReAct                                          | Deep Research                                                          |
|---|---|---|
| Loop shape          | Reason → Act → Observe → repeat, agent decides | Plan → Search → Reflect → Follow-up → Synthesize (typed phases)         |
| Termination         | Agent decides "done"                            | Coverage / depth / claim-confidence criteria                            |
| Output              | Whatever the task asks for                      | **Cited synthesis** — claims linked to source documents                 |
| Gap-handling        | Implicit ("I should look harder")               | Explicit gap-analysis step between rounds                              |
| Memory of sources   | Implicit in transcript                          | Explicit evidence store with stable handles for citation               |
| Substrate           | Agent-level pattern                             | Often *implemented on top of* a ReAct loop                              |

**How to choose.** "Build a customer-support agent that calls our tools" — **ReAct.** "Build a competitive-analysis agent that produces a cited report on a market" — **Deep Research** (its phase structure and citation tracking are load-bearing).

**Common confusion.** "Deep Research is just ReAct with web search." False — the *loop's structure* differs. ReAct lets the model choose what to do each turn; Deep Research's controller decides which **phase** to run (plan vs search vs reflect vs synthesize) and enforces citation tracking between rounds.

---

## Goal Management vs Resource Optimization

[Goal Management →](./orchestration/goal_management/)  ·  [Resource Optimization →](./observability/resource_optimization/)

**One-liner.** Goal Management tracks **what** the agent is doing. Resource Optimization tracks **how much capacity** it has to do anything.

**What they share.** Both surface in the same agent run and both inform decisions — but on totally different axes.

**Axis of difference: goal-space vs resource-space.**

| Aspect              | Goal Management                                  | Resource Optimization                                            |
|---|---|---|
| Domain              | Goals, tasks, todos, milestones, "what's next?"  | Tokens, cost, context window, latency, API quota, "how much budget left?" |
| Question it answers | *What should the agent attempt now?*             | *Can the agent afford to attempt this now?*                       |
| Failure mode if missing | Agent thrashes between objectives or forgets what it was doing | Agent runs out of context mid-task, exhausts budget, or stalls   |
| Typical artifacts   | Plan steps, todo list, current focus, completion criteria | Token counts, cost meters, compaction triggers, cache hit rates  |
| Operates over       | The *workflow*                                    | The *runtime constraints* of the workflow                          |

**How to choose.** Different concern entirely — most production agents need **both**. They compose:

- Goal Manager: "the next thing to do is summarize all 50 docs."
- Resource Optimizer: "but the context window only has room for 12 of them — paginate, or send each to a subagent."

**Common confusion.** "Aren't todos a form of resource tracking?" No — a todo records *what to do*, not *how much capacity remains*. Conversely, a token counter says nothing about *which* goal you're pursuing. The two are orthogonal and you typically want both wired together.

---

## Memory Management vs Context Management

[Memory Management →](./memory/memory_management/)  ·  [Context Management →](./memory/context_management/)

**One-liner.** Memory Management is **across sessions** (persistent). Context Management is **within this turn** (the prompt window).

**What they share.** Both decide what information the agent has access to. They're in the same category directory for a reason — but they answer different questions.

**Axis of difference: time horizon and persistence.**

| Aspect              | Memory Management                                  | Context Management                                            |
|---|---|---|
| Time horizon        | Long-lived — persists across sessions               | Short-lived — this conversation, this turn                    |
| Storage             | Vector store, episodic memory, session forks, sidecar files | The prompt itself: system prompt + history + injected context |
| Question it answers | *What does the agent remember next week?*           | *What does the agent see in this turn?*                       |
| Typical mechanisms  | Embeddings + retrieval, durable JSONL session repos, memory-as-tool | Compaction, project-context loading (`AGENTS.md`/`CLAUDE.md` walk), system-prompt assembly |
| Failure mode        | Agent forgets you discussed X two weeks ago         | Agent's context overflows mid-conversation                    |
| Granularity         | Records / chunks / episodes                          | Tokens                                                         |

**How they relate.** Memory **feeds** context. A retrieved memory enters context when relevant; a compacted conversation is summarized *into* a memory record. They flow into each other:

```
   ┌─────────────────────────┐
   │     Memory (durable)    │  ← long-term, across sessions
   └────┬────────────────▲───┘
        │ retrieve         │ persist
        ▼                  │
   ┌─────────────────────────┐
   │    Context (this turn)  │  ← short-term, within prompt window
   └─────────────────────────┘
```

**How to choose.**

- Need agent to recall the user's preferences from last week? **Memory.**
- Need agent to fit a 200k-token conversation into a 100k window without losing thread? **Context.**
- Building a long-running personal assistant? **Both** — memory for facts the agent should never forget; context engineering for the working set this turn.

**Common confusion.** "Compaction is memory." No — compaction shrinks context to fit a window. If you also persist the compacted summary across sessions, *that part* is memory. Compaction is context engineering; the durable summary is memory.

---

## When in doubt

If you're squinting at two patterns and can't tell which to use, ask:

- **"Is one the mechanism and the other the loop?"** → ReAct vs Tool Use; Deep Research vs ReAct.
- **"Are they on different axes?"** → Goal vs Resource; Memory vs Context.
- **"Is one a stricter / topology-different version of the other?"** → ToT vs GoT (tree vs graph); Subagents vs Parallelization (isolated vs shared context).

In all three framings, the answer is rarely "use one *instead of* the other" — production agents usually compose them.
