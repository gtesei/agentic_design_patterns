# Pi — Multi-Agent Collaboration (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Partial — implemented as extensions, explicitly kept out of core.** Pi has no multi-agent abstraction in the framework. The framework's `Agent` class represents *one* agent loop; multi-agent topologies are built **above** the framework, in extensions.

Two reference implementations ship in the repo:

1. **`subagent/` extension** — full orchestrator-worker pattern with process-level isolation (each subagent runs in a separate `pi` subprocess). Three modes: single, parallel (up to 4 concurrent, queue cap 8), chain (sequential with `{previous}` substitution). Discovered from `~/.pi/agent/agents/*.md` files. The closest analog to multi-agent collaboration in Pi; covered in depth at `orchestration/subagents/pi.md`.
2. **`handoff.ts` extension** — context-transfer-to-new-session pattern. Instead of two agents running concurrently, the user invokes `/handoff <goal>`, an LLM generates a focused prompt summarizing the current session (respecting compaction), and the user starts a new session from that prompt. This is sequential, single-agent-at-a-time collaboration — a *handoff* in the OpenAI-Agents-SDK sense.

Pi's documented stance:

> It intentionally does not include built-in MCP, **sub-agents**, permission popups, plan mode, to-dos, or background bash. You can build or install those workflows as extensions or packages.
> *(`packages/coding-agent/docs/usage.md:275`)*

> **No sub-agents.** There's many ways to do this. Spawn pi instances via tmux, or build your own with extensions, or install a package that does it your way.
> *(`packages/coding-agent/README.md:474`)*

Both subagents and handoff sit as opt-in workflows, not framework primitives.

## Where it lives

| Concern | File:line |
|---|---|
| Explicit "no multi-agent in core" stance | `packages/coding-agent/docs/usage.md:275`, `packages/coding-agent/README.md:474` |
| Subagent extension entry point | `packages/coding-agent/examples/extensions/subagent/index.ts:431-442` |
| Agent discovery (user vs project, with security prompts) | `packages/coding-agent/examples/extensions/subagent/agents.ts:97-115` |
| Subprocess spawn (true context isolation) | `packages/coding-agent/examples/extensions/subagent/index.ts:265-310` |
| Sample subagent definitions | `packages/coding-agent/examples/extensions/subagent/agents/{scout,planner,reviewer,worker}.md` |
| Workflow templates | `packages/coding-agent/examples/extensions/subagent/prompts/{implement,scout-and-plan,implement-and-review}.md` |
| Chain mode execution | `packages/coding-agent/examples/extensions/subagent/index.ts:501-549` |
| Handoff extension | `packages/coding-agent/examples/extensions/handoff.ts` |
| Handoff prompt-generation logic | `packages/coding-agent/examples/extensions/handoff.ts:31-46, 80-103` |

## Topologies Pi supports (and how)

| Topology | Pi mechanism | Layer |
|---|---|---|
| **Orchestrator-worker** with isolated context | `subagent` extension `single` mode | extension |
| **Orchestrator-N-workers** in parallel | `subagent` extension `parallel` mode (cap 4 concurrent, queue 8) | extension |
| **Sequential chain** (worker A → worker B → worker C) | `subagent` extension `chain` mode with `{previous}` substitution | extension |
| **Specialist roles** (scout, planner, reviewer, worker) | Markdown agent definitions with per-agent system prompt + tool allowlist + model | extension |
| **Handoff** (transfer context to a new focused session) | `handoff.ts` extension via `/handoff <goal>` slash command | extension |
| **Critic/reviewer loop** (worker → reviewer → worker) | Subagent `chain` mode + `implement-and-review.md` template | extension |
| **Hierarchical** (orchestrator → subagent → sub-subagent) | Works mechanically; no first-class support | by-construction |
| **Peer-to-peer / mesh** (agents talking freely) | ❌ not implemented |
| **Group chat** (multiple agents sharing one transcript) | ❌ not implemented |
| **A2A protocol** (remote agent-to-agent across processes/orgs) | ❌ not implemented |

## Key code excerpts

### "No sub-agents" — design stance verbatim

```md
<!-- packages/coding-agent/README.md:474 -->
**No sub-agents.** There's many ways to do this. Spawn pi instances via tmux,
or build your own with extensions, or install a package that does it your way.
```

```md
<!-- packages/coding-agent/docs/usage.md:275 -->
It intentionally does not include built-in MCP, sub-agents, permission popups,
plan mode, to-dos, or background bash. You can build or install those workflows
as extensions or packages.
```

**Why relevant:** Multi-agent collaboration is a workflow concern, not a framework concern, by maintainer choice. Building it externally also means each topology gets the implementation it deserves rather than being forced into one shape.

### Subagent topology: a single tool exposes single/parallel/chain modes

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts:431-442 (excerpted)
pi.registerTool({
    name: "subagent",
    label: "Subagent",
    description: [
        "Delegate tasks to specialized subagents with isolated context.",
        "Modes: single (agent + task), parallel (tasks array), chain (sequential with {previous} placeholder).",
        'Default agent scope is "user" (from ~/.pi/agent/agents).',
        'To enable project-local agents in .pi/agents, set agentScope: "both" (or "project").',
    ].join(" "),
    parameters: SubagentParams,
    // ...
});
```

**Why relevant:** All three topologies (single, parallel, chain) are surfaced through one LLM-facing tool. The orchestrator-LLM chooses the mode per call. There is no separate "swarm" or "supervisor" abstraction — the orchestrator's *prompt* determines the topology.

### Agent discovery — user vs project scope, with name-keyed dedup

```ts
// packages/coding-agent/examples/extensions/subagent/agents.ts:97-115 (excerpted)
export function discoverAgents(cwd: string, scope: AgentScope): AgentDiscoveryResult {
    const userDir = path.join(getAgentDir(), "agents");
    const projectAgentsDir = findNearestProjectAgentsDir(cwd);

    const userAgents = scope === "project" ? [] : loadAgentsFromDir(userDir, "user");
    const projectAgents = scope === "user" || !projectAgentsDir ? [] : loadAgentsFromDir(projectAgentsDir, "project");

    const agentMap = new Map<string, AgentConfig>();

    if (scope === "both") {
        for (const agent of userAgents) agentMap.set(agent.name, agent);
        for (const agent of projectAgents) agentMap.set(agent.name, agent);
    }
    // ...
    return { agents: Array.from(agentMap.values()), projectAgentsDir };
}
```

**Why relevant:** Agents are loaded from disk paths (`~/.pi/agent/agents/` for user, `<repo>/.pi/agents/` for project). Default is `user` only. The `Map<string, AgentConfig>` deduplicates by name; when `scope === "both"`, project agents win (loaded last). Loading project agents requires `agentScope: "both"` or `"project"` plus an interactive confirmation prompt (security boundary against repo-controlled agents).

### Subprocess spawn — true context isolation via separate `pi` process

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts:265-310 (excerpted)
const args: string[] = ["--mode", "json", "-p", "--no-session"];
if (agent.model) args.push("--model", agent.model);
if (agent.tools && agent.tools.length > 0) args.push("--tools", agent.tools.join(","));
// ...
if (agent.systemPrompt.trim()) {
    const tmp = await writePromptToTempFile(agent.name, agent.systemPrompt);
    // ...
    args.push("--append-system-prompt", tmpPromptPath);
}

args.push(`Task: ${task}`);

const proc = spawn(invocation.command, invocation.args, {
    cwd: cwd ?? defaultCwd,
    shell: false,
    stdio: ["ignore", "pipe", "pipe"],
});
```

**Why relevant:** Each subagent invocation is a separate OS process running the `pi` binary in **JSON mode** (`--mode json`), **non-interactive** (`-p`), and crucially **`--no-session`** — the subagent's run is not persisted to disk as a session file. The orchestrator passes none of its own context; the child boots fresh with only the system prompt (loaded from `.md` file via `--append-system-prompt`) and the user-provided task string.

The strongest form of context isolation available without containerization. Stronger than in-process "agent-as-tool" patterns that share message arrays. The price: subprocess boot overhead (~100ms+) and a one-shot result interface.

### Sample subagent definition (worker.md)

```markdown
<!-- packages/coding-agent/examples/extensions/subagent/agents/worker.md (excerpted) -->
---
name: worker
description: General-purpose subagent with full capabilities, isolated context
model: claude-sonnet-4-5
---

You are a worker agent with full capabilities. You operate in an isolated context window to handle delegated tasks without polluting the main conversation.

Work autonomously to complete the assigned task. Use all available tools as needed.

Output format when finished:

## Completed
What was done.

## Files Changed
- `path/to/file.ts` - what changed
```

**Why relevant:** A subagent definition is just a markdown file. Frontmatter declares `name`, `description`, optional `tools` allowlist, optional `model`. The body becomes the subagent's system prompt. The "Output format when finished" instruction is the *contract* — it tells the subagent to produce a structured summary the orchestrator can paste back as a tool result.

### Chain mode — sequential handoff with explicit output passing

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts:501-549 (excerpted)
for (let i = 0; i < params.chain.length; i++) {
    const step = params.chain[i];
    const taskWithContext = step.task.replace(/\{previous\}/g, previousOutput);
    // ...
    const result = await runSingleAgent(
        ctx.cwd, agents, step.agent, taskWithContext, step.cwd,
        i + 1, signal, chainUpdate, makeDetails("chain"),
    );
    results.push(result);
    previousOutput = getFinalOutput(result.messages);
}
```

**Why relevant:** No shared message history — only the previous subagent's final text is substituted into the next subagent's task via `{previous}`. Each step boots a fresh `pi` process with no memory of the prior step except the explicit placeholder. The "structured summary as the interface" pattern from Anthropic's multi-agent research system.

### Handoff — single-agent context transfer, not concurrent collaboration

```ts
// packages/coding-agent/examples/extensions/handoff.ts:31-46 (SYSTEM_PROMPT excerpt)
const SYSTEM_PROMPT = `You are a context transfer assistant. Given a conversation history and the user's goal for a new thread, generate a focused prompt that:

1. Summarizes relevant context from the conversation (decisions made, approaches taken, key findings)
2. Lists any relevant files that were discussed or modified
3. Clearly states the next task based on the user's goal
4. Is self-contained - the new thread should be able to proceed without the old conversation

Format your response as a prompt the user can send to start the new thread. Be concise but include all necessary context.`;
```

```ts
// packages/coding-agent/examples/extensions/handoff.ts:80-103 (excerpted)
pi.registerCommand("handoff", {
    description: "Transfer context to a new focused session",
    handler: async (args, ctx) => {
        // ... gather conversation context from current branch (respects compaction!)
        const messages = getHandoffMessages(ctx.sessionManager.getBranch());
        // ...
        // Generate the handoff prompt with loader UI; the result is a draft for the user
    },
});
```

**Why relevant:** Handoff is a different shape from subagents. Subagents are *concurrent or chained delegation*; handoff is *sequential context transfer between sessions*. The user invokes `/handoff <goal>`, an LLM reads the current session's history (respecting compaction!) and writes a focused starter prompt for the next session, which appears as a draft in the editor for review/editing. This is the **OpenAI-Agents-SDK-style handoff** pattern: agent A finishes, agent B starts with a curated context, only one agent active at a time.

Also the *honest alternative* to compaction: where compaction is lossy and keeps you in one session, handoff is goal-directed and explicitly starts a new session — the user reviews the synthesized prompt, so it's a more controllable form of "what should the next thread know?"

### Workflow templates — multi-agent recipes the orchestrator follows

The subagent extension ships three workflow prompts under `examples/extensions/subagent/prompts/`:

- `implement.md` → `scout → planner → worker` (recon, plan, execute)
- `scout-and-plan.md` → `scout → planner` (analysis without implementation)
- `implement-and-review.md` → `worker → reviewer → worker` (implement, critique, revise — reflection-as-chain; see `4_reflection/pi.md`)

These are not code — they are markdown prompts loaded into the orchestrator's context that instruct it to invoke the `subagent` tool in chain mode with the right `agent` + `task` per step. **Patterns layered on top of the topology primitive (the chain mode).**

## Tradeoffs and limitations

- **All cross-agent topologies live in extensions.** Stock `pi` is single-agent.
- **Process isolation is non-negotiable.** Subagents are always separate `pi` processes — heavy (cold-start latency), strong (true OS-level isolation). No in-process "agent-as-tool" lightweight alternative.
- **No shared state between agents.** Chain mode passes plain text via `{previous}`. No shared scratchpad, no shared memory store, no typed handoff schema. Quality depends on upstream agents producing parseable output.
- **No mesh / peer-to-peer.** All flows are tree-shaped (one orchestrator, leaf workers) or linear (chain). Agents don't talk to each other freely.
- **No group chat / shared transcript.** Each subagent has its own ephemeral context (`--no-session`); their transcripts don't merge.
- **No A2A protocol.** Cross-organization or cross-process agent interop out of scope.
- **Orchestrator is the LLM** that drives the subagent tool. The "supervisor" pattern is implemented by *prompting* the orchestrator LLM via workflow templates, not as a programmatic class.
- **Subagent failure handling is the orchestrator's responsibility.** No built-in retry or fallback policy.
- **Handoff is one-way and session-bounded.** It's a "wrap up here and start there" pattern. Two sessions can't be active simultaneously and coordinating.

## "Not implemented" caveats

- ❌ Framework-level multi-agent class (`Swarm`, `Supervisor`, `GroupChat`)
- ❌ Peer-to-peer / mesh / network topologies
- ❌ Shared transcript or shared scratchpad between agents
- ❌ Typed handoff schemas between agents
- ❌ A2A protocol or remote agent-to-agent
- ❌ Programmatic orchestrator (orchestration is LLM-driven via prompts)
- ❌ Concurrent multi-session coordination
- ❌ Built-in retry / fallback across agents

What Pi does ship:

- ✅ Orchestrator-worker topology (`subagent` extension, single mode)
- ✅ Orchestrator-N-workers (`subagent` parallel mode, concurrency-capped)
- ✅ Sequential chain (`subagent` chain mode with `{previous}` substitution)
- ✅ Specialist roles via markdown agent definitions (system prompt, tool allowlist, model)
- ✅ Critic/reviewer pattern via `implement-and-review.md` workflow template
- ✅ Handoff (`/handoff` slash command, LLM-generated focused next-thread prompt that respects compaction)
- ✅ Process-level context isolation for subagents
- ✅ Project-vs-user agent discovery with security boundary on project-local agents
- ✅ Reference workflows showing how to compose the primitives
