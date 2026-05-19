# Pi Across the Agentic Design Patterns — Summary

Roll-up of the 28 per-pattern `pi.md` analyses in this repo. Each per-pattern doc answers *"does Pi implement this pattern, and how?"*. This file aggregates those verdicts into a heat map plus a compact per-pattern dossier.

Source `pi.md` files accessed: **2026-05-17**. Rebuild this summary if any per-pattern doc is updated.

> **Pi** here refers to the Pi coding-agent framework (`packages/coding-agent`, `packages/agent`, `packages/ai`) — the underlying SDK, not the CLI surface alone.

---

## 1. Scoring rubric

| Score | Label | Meaning |
|:---:|---|---|
| **5** | Core | First-class runtime abstraction with dedicated APIs |
| **4** | Strong | Substantial implementation that meaningfully matches the pattern |
| **3** | Substantial | Implements the *spirit* via a related mechanism, not the textbook construction |
| **2** | Partial | Pi has primitives in the neighborhood but doesn't implement the pattern itself |
| **1** | Minimal | Only a small or tangential piece exists |
| **0** | Not implemented | Verdict says Pi does not implement the pattern meaningfully |

---

## 2. Coverage at a glance

28 patterns analyzed.

| Band | Count | Patterns |
|:---:|:---:|---|
| **5 — Core**          | 6 | Parallelization · Tool Use · ReAct · HITL · Skills · Context Management |
| **4 — Strong**        | 5 | Structured Outputs · Error Recovery · Guardrails · Subagents · Resource Optimization |
| **3 — Substantial**   | 5 | Planning · Multi-Agent Collaboration · Exploration & Discovery · Evaluation & Monitoring · Memory Management |
| **2 — Partial**       | 4 | Routing · Reflection · Goal Management · Agent Communication |
| **1 — Minimal**       | 5 | Prompt Chaining · Computer Use · Tree of Thoughts · Deep Research · Prioritization |
| **0 — Not implemented** | 3 | Graph of Thoughts · MCP · Adaptive Learning |

**Center of gravity.** Pi is strongest on the runtime mechanics that a real coding agent uses every turn — the **ReAct loop, parallel tool execution, HITL approvals, skills, and context engineering / compaction**. It is weakest on **search-based reasoning** (ToT/GoT/Deep Research) and **explicit pattern catalog features that conflict with Pi's "core stays small, extensions own opinions" stance** (MCP, plan-mode, todos, multi-agent topologies).

---

## 3. Heat map (sorted by score)

| # | Pattern | Category | Score | Heat |
|:---:|---|---|:---:|---|
| 3  | [Parallelization](./foundational_design_patterns/3_parallelization/pi.md)               | Foundational   | 5 | 🟩🟩🟩🟩🟩 |
| 5  | [Tool Use](./foundational_design_patterns/5_tool_use/pi.md)                             | Foundational   | 5 | 🟩🟩🟩🟩🟩 |
| 8  | [ReAct](./foundational_design_patterns/8_react/pi.md)                                   | Foundational   | 5 | 🟩🟩🟩🟩🟩 |
| 10 | [Human-in-the-Loop](./foundational_design_patterns/10_hitl/pi.md)                       | Foundational   | 5 | 🟩🟩🟩🟩🟩 |
| 21 | [Skills](./orchestration/skills/pi.md)                                                  | Orchestration  | 5 | 🟩🟩🟩🟩🟩 |
| 28 | [Context Management](./memory/context_management/pi.md)                                 | Memory         | 5 | 🟩🟩🟩🟩🟩 |
| 11 | [Structured Outputs](./foundational_design_patterns/11_structured_outputs/pi.md)        | Foundational   | 4 | 🟩🟩🟩🟩⬜ |
| 17 | [Error Recovery](./reliability/error_recovery/pi.md)                                    | Reliability    | 4 | 🟩🟩🟩🟩⬜ |
| 18 | [Guardrails](./reliability/guardrails/pi.md)                                            | Reliability    | 4 | 🟩🟩🟩🟩⬜ |
| 20 | [Subagents](./orchestration/subagents/pi.md)                                            | Orchestration  | 4 | 🟩🟩🟩🟩⬜ |
| 26 | [Resource Optimization](./observability/resource_optimization/pi.md)                    | Observability  | 4 | 🟩🟩🟩🟩⬜ |
| 6  | [Planning](./foundational_design_patterns/6_planning/pi.md)                             | Foundational   | 3 | 🟩🟩🟩⬜⬜ |
| 7  | [Multi-Agent Collaboration](./foundational_design_patterns/7_multi_agent_collaboration/pi.md) | Foundational   | 3 | 🟩🟩🟩⬜⬜ |
| 15 | [Exploration & Discovery](./reasoning/exploration_discovery/pi.md)                      | Reasoning      | 3 | 🟩🟩🟩⬜⬜ |
| 25 | [Evaluation & Monitoring](./observability/evaluation_monitoring/pi.md)                  | Observability  | 3 | 🟩🟩🟩⬜⬜ |
| 27 | [Memory Management](./memory/memory_management/pi.md)                                   | Memory         | 3 | 🟩🟩🟩⬜⬜ |
| 2  | [Routing](./foundational_design_patterns/2_routing/pi.md)                               | Foundational   | 2 | 🟩🟩⬜⬜⬜ |
| 4  | [Reflection](./foundational_design_patterns/4_reflection/pi.md)                         | Foundational   | 2 | 🟩🟩⬜⬜⬜ |
| 19 | [Goal Management](./orchestration/goal_management/pi.md)                                | Orchestration  | 2 | 🟩🟩⬜⬜⬜ |
| 22 | [Agent Communication](./orchestration/agent_communication/pi.md)                        | Orchestration  | 2 | 🟩🟩⬜⬜⬜ |
| 1  | [Prompt Chaining](./foundational_design_patterns/1_prompt_chain/pi.md)                  | Foundational   | 1 | 🟩⬜⬜⬜⬜ |
| 12 | [Computer Use](./foundational_design_patterns/12_computer_use/pi.md)                    | Foundational   | 1 | 🟩⬜⬜⬜⬜ |
| 13 | [Tree of Thoughts](./reasoning/tree_of_thoughts/pi.md)                                  | Reasoning      | 1 | 🟩⬜⬜⬜⬜ |
| 16 | [Deep Research](./reasoning/deep_research/pi.md)                                        | Reasoning      | 1 | 🟩⬜⬜⬜⬜ |
| 24 | [Prioritization](./orchestration/prioritization/pi.md)                                  | Orchestration  | 1 | 🟩⬜⬜⬜⬜ |
| 14 | [Graph of Thoughts](./reasoning/graph_of_thoughts/pi.md)                                | Reasoning      | 0 | ⬜⬜⬜⬜⬜ |
| 23 | [MCP](./orchestration/mcp/pi.md)                                                        | Orchestration  | 0 | ⬜⬜⬜⬜⬜ |
| 29 | [Adaptive Learning](./learning/adaptive_learning/pi.md)                                 | Learning       | 0 | ⬜⬜⬜⬜⬜ |

---

## 4. Per-pattern dossiers

Compact summary of every per-pattern `pi.md`: the one-line verdict, the key Pi construct that gets closest, the most representative code citation, and the biggest gap. Follow the pattern link to the full analysis for the verbatim code excerpts.

### Foundational design patterns

#### 1. Prompt Chaining — score **1 / 5** (Minimal)
[full pi.md →](./foundational_design_patterns/1_prompt_chain/pi.md)
- **Verdict.** Pi does not implement prompt chaining as a first-class runtime abstraction; the closest mechanisms are slash-command template expansion, system-prompt composition, and example workflow prompts.
- **Closest construct.** Deterministic prompt-template expansion in `expandPromptTemplate`.
- **Citation.** `packages/coding-agent/src/core/agent-session.ts:973-1008`
- **Main gap.** No typed stage outputs, per-stage retries, or chain observability in core.

#### 2. Routing — score **2 / 5** (Partial)
[full pi.md →](./foundational_design_patterns/2_routing/pi.md)
- **Verdict.** Pi implements rule-based deterministic input dispatch (slash-commands, prompt templates, skills, extension input hooks) but no LLM-driven semantic classifier router.
- **Closest construct.** Name-based slash-command dispatch in `AgentSession.prompt()`.
- **Citation.** `packages/coding-agent/src/core/agent-session.ts:973-1008`
- **Main gap.** No semantic / classifier-based prompt routing, only explicit name dispatch.

#### 3. Parallelization — score **5 / 5** (Core)
[full pi.md →](./foundational_design_patterns/3_parallelization/pi.md)
- **Verdict.** Pi parallelizes tool execution by default in the core agent loop with a named `ToolExecutionMode` policy and per-tool sequential override, plus parallel subagent fan-out in the subagent extension.
- **Closest construct.** `executeToolCallsParallel` with `Promise.all` in the agent loop.
- **Citation.** `packages/agent/src/agent-loop.ts:447-505`
- **Main gap.** Tool *preparation* is still serialized; one per-tool sequential mode poisons the whole batch.

#### 4. Reflection — score **2 / 5** (Partial)
[full pi.md →](./foundational_design_patterns/4_reflection/pi.md)
- **Verdict.** Pi has no built-in self-critique loop in the core runtime; reflection exists only as an `implement-and-review` subagent chain workflow in an example extension.
- **Closest construct.** Subagent `chain` mode with worker → reviewer → worker template.
- **Citation.** `packages/coding-agent/examples/extensions/subagent/prompts/implement-and-review.md:4-10`
- **Main gap.** No runtime reflection primitive; handoff is plain text and costs extra subprocess turns.

#### 5. Tool Use — score **5 / 5** (Core)
[full pi.md →](./foundational_design_patterns/5_tool_use/pi.md)
- **Verdict.** Tool use is Pi's core competency with a TypeBox-typed `Tool` contract, parallel-by-default execution, pre/post hooks, multimodal results, streaming updates, and extension-registered tools indistinguishable from core ones.
- **Closest construct.** Tool lifecycle (prepare → validate → beforeToolCall → execute → afterToolCall) in `agent-loop.ts`.
- **Citation.** `packages/agent/src/agent-loop.ts:568-684`
- **Main gap.** No built-in MCP, no automatic tool retry, no mid-execution tool-result streaming to the LLM.

#### 6. Planning — score **3 / 5** (Substantial)
[full pi.md →](./foundational_design_patterns/6_planning/pi.md)
- **Verdict.** Planning is explicitly kept out of core but shipped as two reference extensions: `plan-mode` (tool-restricted read-only mode with regex-parsed plan extraction) and subagent chain workflows with a dedicated `planner` role.
- **Closest construct.** `plan-mode` extension with read-only tool allowlist and `[DONE:n]` step parsing.
- **Citation.** `packages/coding-agent/examples/extensions/plan-mode/index.ts:158-205`
- **Main gap.** No typed plan schema, no replan-on-failure, plan format is brittle regex over assistant text.

#### 7. Multi-Agent Collaboration — score **3 / 5** (Substantial)
[full pi.md →](./foundational_design_patterns/7_multi_agent_collaboration/pi.md)
- **Verdict.** Multi-agent topologies (orchestrator–worker single / parallel / chain, handoff) are implemented as opt-in extensions with process-level isolation, never as core framework abstractions.
- **Closest construct.** `subagent` extension spawning isolated `pi --mode json --no-session` subprocesses.
- **Citation.** `packages/coding-agent/examples/extensions/subagent/index.ts:265-310`
- **Main gap.** No peer-to-peer / mesh, no shared transcript, no A2A protocol, no typed handoff schemas.

#### 8. ReAct (Reason + Act) — score **5 / 5** (Core)
[full pi.md →](./foundational_design_patterns/8_react/pi.md)
- **Verdict.** Pi's core agent loop *is* a ReAct loop with typed separation of thinking / text / tool-call content, plus Pi-specific extensions for mid-run steering and follow-up message injection.
- **Closest construct.** `runLoop` alternating `streamAssistantResponse` and `executeToolCalls` in `agent-loop.ts`.
- **Citation.** `packages/agent/src/agent-loop.ts:170-220`
- **Main gap.** No built-in iteration cap or tool-failure retry policy; structure is in message types, not tagged prompt text.

#### 10. Human-in-the-Loop — score **5 / 5** (Core)
[full pi.md →](./foundational_design_patterns/10_hitl/pi.md)
- **Verdict.** Pi treats HITL as first-class with a `beforeToolCall` block primitive, session-event hooks, typed UI APIs (confirm / select / input / editor with timeout), mid-run steering / followUp methods, and seven reference extensions.
- **Closest construct.** `beforeToolCall` returning `{ block, reason }` plus `AgentSession.steer()` / `followUp()`.
- **Citation.** `packages/agent/src/types.ts:262-281`
- **Main gap.** No async approval queues, multi-reviewer flows, or built-in audit log of decisions.

#### 11. Structured Outputs — score **4 / 5** (Strong)
[full pi.md →](./foundational_design_patterns/11_structured_outputs/pi.md)
- **Verdict.** Structured *tool inputs* (TypeBox schemas end-to-end) and transport-level `--mode json` event stream are first-class; structured *final answers* are an extension pattern (`structured_output` tool with `terminate: true`), not a `generateObject` shorthand.
- **Closest construct.** TypeBox `Tool<TParameters extends TSchema>` with `Static<TParameters>` typed `execute`.
- **Citation.** `packages/agent/src/types.ts:361-385`
- **Main gap.** No `generateObject` / `withStructuredOutput` shorthand, no constrained-grammar decoding, no JSON-mode passthrough at the agent layer.

#### 12. Computer Use — score **1 / 5** (Minimal)
[full pi.md →](./foundational_design_patterns/12_computer_use/pi.md)
- **Verdict.** Computer use in the screenshot+click+observe sense is not implemented; Pi only ships passive multimodal input (CLI image attach, `read` tool inline images, clipboard paste) and shell access via `bash`.
- **Closest construct.** `read` tool returning inline `ImageContent` for image files (passive vision only).
- **Citation.** `packages/coding-agent/src/core/tools/read.ts:245-276`
- **Main gap.** No browser / desktop control, no screenshot capture, no click / type primitives, no GUI action vocabulary.

### Reasoning patterns

#### 13. Tree of Thoughts — score **1 / 5** (Minimal)
[full pi.md →](./reasoning/tree_of_thoughts/pi.md)
- **Verdict.** Pi has tree-shaped *session* storage with branching and branch summarization, but no ToT reasoning algorithm: no candidate generation, scoring, or branch pruning at the model level.
- **Closest construct.** Append-only session tree with `id` / `parentId` in `session-manager.ts`.
- **Citation.** `packages/coding-agent/src/core/session-manager.ts:696-704`
- **Main gap.** Session tree is for human navigation, not automatic branch expansion / scoring / pruning.

#### 14. Graph of Thoughts — score **0 / 5** (Not implemented)
[full pi.md →](./reasoning/graph_of_thoughts/pi.md)
- **Verdict.** Pi does not implement Graph of Thoughts; its session structure is explicitly a parent / child tree with no shared children, merge nodes, or arbitrary graph edges.
- **Closest construct.** Rooted `SessionTreeNode` structure (tree only, no graph).
- **Citation.** `packages/coding-agent/src/core/session-manager.ts:1108-1145`
- **Main gap.** No graph-shaped reasoning state, no recombination / merge nodes.

#### 15. Exploration & Discovery — score **3 / 5** (Substantial)
[full pi.md →](./reasoning/exploration_discovery/pi.md)
- **Verdict.** Exploration is supported via three distributed mechanisms: system-prompt exploration guidelines, a read-only tool bundle (`createReadOnlyToolDefinitions`), and a dedicated `scout` subagent with `scout-and-plan` workflow.
- **Closest construct.** Read-only tool bundle plus `scout` subagent role definition.
- **Citation.** `packages/coding-agent/examples/extensions/subagent/agents/scout.md:2-21`
- **Main gap.** No typed evidence graph or durable retrieval index; results passed as plain text.

#### 16. Deep Research — score **1 / 5** (Minimal)
[full pi.md →](./reasoning/deep_research/pi.md)
- **Verdict.** Deep research is not implemented as a first-class pattern; Pi only ships adjacent infrastructure (deep-research model IDs in registry, a prompt mention of `brave-search` via bash skill, chained recon examples).
- **Closest construct.** Model registry entries for `o3-deep-research` / `o4-mini-deep-research` and external skills.
- **Citation.** `packages/ai/src/models.generated.ts:2496-2512`
- **Main gap.** No built-in web search tool, citation tracking, or multi-round search–refine–synthesize loop.

### Reliability patterns

#### 17. Error Recovery — score **4 / 5** (Strong)
[full pi.md →](./reliability/error_recovery/pi.md)
- **Verdict.** Pi has substantial error recovery: configurable exponential-backoff retries for transient provider errors, automatic compact-and-retry on context overflow, and tool-failure isolation into structured error results.
- **Closest construct.** `_isRetryableError` / `_handleRetryableError` plus overflow-triggered auto-compaction.
- **Citation.** `packages/coding-agent/src/core/agent-session.ts:2410-2505`
- **Main gap.** Retryability is regex over provider error strings; overflow recovery is one-shot and compaction is lossy.

#### 18. Guardrails — score **4 / 5** (Strong)
[full pi.md →](./reliability/guardrails/pi.md)
- **Verdict.** Guardrails are meaningfully implemented as a policy layer: `tool_call` blocking hook plus reference extensions for command approval, protected paths, and an OS-level sandboxed bash wrapper.
- **Closest construct.** `tool_call` extension event returning `{ block, reason }` with sandbox tool replacement.
- **Citation.** `packages/coding-agent/examples/extensions/sandbox/index.ts:214-279`
- **Main gap.** No centralized always-on safety subsystem; strength depends on which extensions the operator enables.

### Orchestration patterns

#### 19. Goal Management — score **2 / 5** (Partial)
[full pi.md →](./orchestration/goal_management/pi.md)
- **Verdict.** Pi explicitly excludes built-in plan mode and todos from core; goal management exists only via opt-in `todo` and `plan-mode` reference extensions with extension-persisted state.
- **Closest construct.** `todo` tool reconstructing state from session entries plus `plan-mode` phase switching.
- **Citation.** `packages/coding-agent/examples/extensions/plan-mode/index.ts:38-97`
- **Main gap.** No standardized goal-management primitive in core; behavior varies per installation.

#### 20. Subagents — score **4 / 5** (Strong)
[full pi.md →](./orchestration/subagents/pi.md)
- **Verdict.** Subagents are fully implemented as an opt-in reference extension with true process-level context isolation, three orchestration modes (single / parallel / chain), and markdown-defined agent roles with frontmatter.
- **Closest construct.** Subagent extension spawning `pi --mode json -p --no-session` subprocesses.
- **Citation.** `packages/coding-agent/examples/extensions/subagent/index.ts:265-310`
- **Main gap.** Extension-only (not core); hardcoded concurrency caps (4 / 8); chain communication is plain text via `{previous}`.

#### 21. Skills — score **5 / 5** (Core)
[full pi.md →](./orchestration/skills/pi.md)
- **Verdict.** Skills are built-in at both framework and app layers, spec-conformant to agentskills.io, with progressive disclosure via compact metadata index and on-demand body loading, plus `/skill:name` explicit invocation.
- **Closest construct.** `SKILL.md` discovery with `<available_skills>` index in system prompt and `_expandSkillCommand`.
- **Citation.** `packages/coding-agent/src/core/skills.ts`
- **Main gap.** No registry / marketplace, no skill versioning or composition, two duplicated implementations (framework + app).

#### 22. Agent Communication — score **2 / 5** (Partial)
[full pi.md →](./orchestration/agent_communication/pi.md)
- **Verdict.** Agent communication is pragmatic and convention-based: subprocess JSON-event capture, `{previous}` text chaining, session handoff, and an extension event bus, but no typed A2A protocol.
- **Closest construct.** Subagent subprocess JSON event stream plus `pi.events` inter-extension bus.
- **Citation.** `packages/coding-agent/examples/extensions/subagent/index.ts:304-338`
- **Main gap.** No versioned typed protocol; communication is mostly text-based and extension-level.

#### 24. Prioritization — score **1 / 5** (Minimal)
[full pi.md →](./orchestration/prioritization/pi.md)
- **Verdict.** Pi has narrow ordering mechanisms (queue-drain modes, steering-before-followup, resource precedence ranks) but no general-purpose goal / task prioritization system.
- **Closest construct.** `PendingMessageQueue` with `QueueMode` and steering / followUp queues.
- **Citation.** `packages/agent/src/agent.ts:118-144`
- **Main gap.** No importance scoring or strategic prioritization over competing objectives.

#### 23. Model Context Protocol (MCP) — score **0 / 5** (Not implemented)
[full pi.md →](./orchestration/mcp/pi.md)
- **Verdict.** MCP is explicitly and intentionally not implemented; the docs state MCP belongs in extensions, and the closest in-tree mechanism is `resources_discover` for local resource discovery only.
- **Closest construct.** `resources_discover` extension event for local skill / prompt / theme paths (not MCP).
- **Citation.** `packages/coding-agent/README.md:470-472`
- **Main gap.** No MCP client, server, or transport; stock Pi cannot talk to MCP servers.

### Observability patterns

#### 25. Evaluation & Monitoring — score **3 / 5** (Substantial)
[full pi.md →](./observability/evaluation_monitoring/pi.md)
- **Verdict.** Monitoring is meaningfully implemented (structured lifecycle events, streaming assistant deltas, `--mode json` machine output, live cost / token / context footer), but built-in evaluation against references is absent.
- **Closest construct.** `AgentSessionEvent` stream covering turns, tools, compaction, retries.
- **Citation.** `packages/coding-agent/src/core/agent-session.ts:120-140`
- **Main gap.** No built-in evaluator that scores outputs against references or runs benchmarks.

#### 26. Resource Optimization — score **4 / 5** (Strong)
[full pi.md →](./observability/resource_optimization/pi.md)
- **Verdict.** Resource optimization is one of Pi's stronger areas: proactive compaction with reserve headroom, structured checkpoint summaries, provider-side prompt caching and session affinity, and explicit per-call cost accounting.
- **Closest construct.** `shouldCompact` proactive trigger plus `sessionId`-based prompt cache routing.
- **Citation.** `packages/coding-agent/src/core/compaction/compaction.ts:121-124`
- **Main gap.** Compaction is lossy; token estimation is `chars / 4` heuristic; no background pre-compaction.

### Memory patterns

#### 27. Memory Management — score **3 / 5** (Substantial)
[full pi.md →](./memory/memory_management/pi.md)
- **Verdict.** Pi implements branchable session persistence as a framework primitive (`SessionRepo` interface, JSONL append-only storage, `uuidv7` entries, fork by `entryId`) but no semantic / vector memory or recall-as-tool.
- **Closest construct.** `JsonlSessionRepo` with `fork(metadata, { entryId, position })` and `CustomEntry` sidecar.
- **Citation.** `packages/agent/src/harness/session/jsonl-repo.ts`
- **Main gap.** No vector store / embedding retrieval, no memory-as-tool exposed to LLM, CWD-scoped only.

#### 28. Context Management — score **5 / 5** (Core)
[full pi.md →](./memory/context_management/pi.md)
- **Verdict.** Context engineering is comprehensive: cwd-to-root `AGENTS.md` / `CLAUDE.md` project-context walk, explicit system-prompt assembly, `transformContext` framework hook, plus ~845 lines of compaction with branch summarization and split-turn handling.
- **Closest construct.** `loadProjectContextFiles` walk plus dual-layer compaction with `shouldCompact` + LLM summarization.
- **Citation.** `packages/coding-agent/src/core/compaction/compaction.ts:454-485`
- **Main gap.** Compaction is sync-blocking; `chars / 4` heuristic for tokens; settings duplicated across framework and app layers.

### Learning patterns

#### 29. Adaptive Learning — score **0 / 5** (Not implemented)
[full pi.md →](./learning/adaptive_learning/pi.md)
- **Verdict.** Adaptive learning is not meaningfully implemented; the word *adaptive* in Pi refers to provider-side adaptive thinking (per-call reasoning effort), not learning from experience over time.
- **Closest construct.** Extension `appendEntry` could persist learning state, but no in-repo feedback loop uses it.
- **Citation.** `packages/ai/src/providers/anthropic.ts:939-948`
- **Main gap.** No feedback collection, prompt / policy adaptation, learned skill acquisition, or trajectory retrieval.

---

## 5. Reading the heat map

A few observations the per-pattern docs converge on:

- **What Pi optimizes for.** The 5-score patterns (parallel tool exec, tool use, ReAct, HITL, skills, context management) describe a coding agent that runs many tool calls per turn, lets a human interrupt, and aggressively manages the prompt window. This matches Pi's product surface.
- **The "core stays small" trade.** Patterns that Pi *could* implement but consciously pushes into reference extensions (planning, multi-agent, todos / goal management, guardrails) all score 3–4. The implementations exist and are good; they are just opt-in. Operators who run stock Pi will not see them.
- **What Pi is not for.** Search-based reasoning (ToT, GoT), web-scale research, computer use, and adaptive learning all score 0–1. These are out of scope, not unfinished work — several `pi.md` docs explicitly say so.
- **MCP gap is notable.** Score **0** on MCP is the one place where the gap is likely to surprise a reader, because MCP is increasingly assumed in modern agent stacks. The docs are explicit that MCP integration is left to extensions.
