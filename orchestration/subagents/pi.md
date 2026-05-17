# Pi — Subagents (Orchestrator-Worker)

**Repository:** https://github.com/earendil-works/repo/pi
**Accessed on:** 2026-05-17

## Summary

**Yes, fully — but as an *opt-in extension*, not core.** Pi implements the orchestrator-worker subagent pattern with **process-level context isolation**: each subagent runs in a separate `pi` subprocess, with its own conversation history, its own model selection, and its own tool allowlist. The orchestrator receives the subagent's structured output over stdout (JSON event stream) and feeds the final result back to the lead agent as a tool result. This is closer to Anthropic's "fresh-context subagent" pattern than to in-process "agent-as-tool" wrappers that share message history.

The implementation ships as `examples/extensions/subagent/` — not in the core `tools/` directory. The core coding-agent tools are `bash`, `edit`, `find`, `grep`, `ls`, `read`, `write` only. Users must symlink the extension into `~/.pi/agent/extensions/subagent/` to activate it.

The extension supports three orchestration modes — single, parallel (up to 4 concurrent), and chain (sequential with `{previous}` placeholder for inter-step context passing). Subagents are discovered from on-disk `.md` files with YAML frontmatter, mirroring the Claude Code subagent convention.

## Where it lives

| Concern | File |
|---|---|
| Tool registration, mode dispatch, subprocess spawn | `packages/coding-agent/examples/extensions/subagent/index.ts` (987 lines) |
| Agent discovery from `~/.pi/agent/agents/*.md` and `.pi/agents/*.md` | `packages/coding-agent/examples/extensions/subagent/agents.ts` |
| Sample subagent definitions (frontmatter + body) | `packages/coding-agent/examples/extensions/subagent/agents/{scout,planner,reviewer,worker}.md` |
| Sample workflow prompts (multi-step chains) | `packages/coding-agent/examples/extensions/subagent/prompts/{implement,scout-and-plan,implement-and-review}.md` |
| Extension README documenting security model, modes, usage | `packages/coding-agent/examples/extensions/subagent/README.md` |

Pi's **core** subagent surface is empty. Verified by listing `packages/coding-agent/src/core/tools/`: only filesystem and shell tools are present. There is no `task`, `delegate`, `subagent`, `spawn`, or `worker` tool in core.

## Key code excerpts

### Subprocess spawn — true context isolation

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts
import { spawn } from "node:child_process";

// ... inside runSingleAgent():
const exitCode = await new Promise<number>((resolve) => {
  const invocation = getPiInvocation(args);
  const proc = spawn(invocation.command, invocation.args, {
    cwd: cwd ?? defaultCwd,
    shell: false,
    stdio: ["ignore", "pipe", "pipe"],
  });
  // ... read JSON events from proc.stdout line by line:
  const processLine = (line: string) => {
    let event: any;
    try { event = JSON.parse(line); } catch { return; }
    if (event.type === "message_end" && event.message) {
      currentResult.messages.push(event.message as Message);
      // ... track usage, model, stopReason
    }
  };
});
```

**Why relevant:** Each subagent invocation is a *separate OS process* running the `pi` binary. The orchestrator does not pass any of its own context to the child; the child boots fresh with only the system prompt (loaded from the agent's `.md` file) and the user-provided task string. This is the strongest form of context isolation — stronger than in-process "agent-as-tool" patterns that share Python objects or share a message array.

The price: subprocess spawn overhead per subagent (~100ms+ cold start in addition to LLM latency), and the orchestrator only sees what the child writes to stdout as JSON events.

### Subagent definition — markdown file with frontmatter

```markdown
<!-- packages/coding-agent/examples/extensions/subagent/agents/worker.md -->
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

## Notes (if any)
Anything the main agent should know.
```

**Why relevant:** A subagent definition is just a markdown file. Frontmatter declares `name`, `description`, optional `tools` allowlist, optional `model`. The body becomes the subagent's system prompt. This mirrors the Claude Code subagent convention closely. The "Output format when finished" instruction is the *contract* — it tells the subagent to produce a structured summary that the orchestrator can paste back as the tool result.

### Agent discovery — disk-scoped, user vs project

```ts
// packages/coding-agent/examples/extensions/subagent/agents.ts
export type AgentScope = "user" | "project" | "both";

export interface AgentConfig {
  name: string;
  description: string;
  tools?: string[];
  model?: string;
  systemPrompt: string;
  source: "user" | "project";
  filePath: string;
}

function loadAgentsFromDir(dir: string, source: "user" | "project"): AgentConfig[] {
  // reads *.md files, parses frontmatter, builds AgentConfig
  // ignores files without name + description in frontmatter
}
```

**Why relevant:** Agents are discovered from `~/.pi/agent/agents/` (user scope) or `<repo>/.pi/agents/` (project scope). Default is `user` only. Loading `project` agents requires explicit opt-in via the `agentScope` parameter, with an interactive confirmation prompt for repo-controlled agents — a security boundary preventing untrusted repos from injecting arbitrary system prompts.

### Tool registration — three modes (single / parallel / chain)

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts (excerpted)
const SubagentParams = Type.Object({
  agent: Type.Optional(Type.String({ description: "Name of the agent to invoke (for single mode)" })),
  task: Type.Optional(Type.String({ description: "Task to delegate (for single mode)" })),
  tasks: Type.Optional(Type.Array(TaskItem, { description: "Array of {agent, task} for parallel execution" })),
  chain: Type.Optional(Type.Array(ChainItem, { description: "Array of {agent, task} for sequential execution" })),
  agentScope: Type.Optional(AgentScopeSchema),
  confirmProjectAgents: Type.Optional(Type.Boolean({ default: true })),
  cwd: Type.Optional(Type.String({ description: "Working directory for the agent process (single mode)" })),
});

export default function (pi: ExtensionAPI) {
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
    async execute(_toolCallId, params, signal, onUpdate, ctx) { /* ... */ },
  });
}
```

**Why relevant:** A single tool exposed to the LLM handles all three orchestration shapes via parameter union. The three modes correspond directly to the canonical orchestrator-worker variants:

- **single**: orchestrator delegates one task to one subagent → receives one summary back.
- **parallel**: orchestrator delegates N independent tasks → N subagents run concurrently (capped at `MAX_CONCURRENCY = 4`) → orchestrator gets N summaries.
- **chain**: orchestrator sequences subagents, with `{previous}` substituted into the next task's prompt. Equivalent to LangGraph supervisor-style sequential handoff but with isolated contexts at each hop.

### Chain mode — sequential handoff with explicit output passing

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts (excerpted)
if (params.chain && params.chain.length > 0) {
  const results: SingleResult[] = [];
  let previousOutput = "";

  for (let i = 0; i < params.chain.length; i++) {
    const step = params.chain[i];
    const taskWithContext = step.task.replace(/\{previous\}/g, previousOutput);
    // ...
    const result = await runSingleAgent(
      ctx.cwd, agents, step.agent, taskWithContext, step.cwd,
      i + 1, signal, chainUpdate, makeDetails("chain"),
    );
    results.push(result);
    // ...
    previousOutput = /* extract final assistant text from result.messages */;
  }
}
```

**Why relevant:** This is the orchestrator's view of inter-subagent communication. There is **no shared message history** — only the previous subagent's final text is substituted into the next subagent's task. Each step boots a fresh `pi` process with no memory of the prior step except the explicit `{previous}` slot. This is the "structured summary as the interface" pattern that Anthropic's multi-agent research system advocates.

### Sample workflow — chained agents

```markdown
<!-- packages/coding-agent/examples/extensions/subagent/prompts/implement-and-review.md -->
[implementation reads as worker -> reviewer -> worker chain;
 the prompt template tells the orchestrator how to invoke the subagent tool with chain mode]
```

The repo ships three workflow templates: `implement.md` (scout → planner → worker), `scout-and-plan.md` (scout → planner), `implement-and-review.md` (worker → reviewer → worker).

## Tradeoffs and limitations

- **Extension, not core.** Users must opt in by symlinking the extension into `~/.pi/agent/extensions/`. The core distribution of `pi` does not include subagent capability out of the box. Defensible (keeps the binary minimal and the security surface smaller) but means subagent capability is not universally available across Pi installs.
- **Process isolation costs latency.** Each subagent invocation is a full `pi` subprocess boot. Compared to in-process agent-as-tool patterns, this is heavier (~100ms+ cold start) but gives true OS-level isolation: a misbehaving subagent cannot crash or corrupt the orchestrator process.
- **Result interface is one-shot.** The orchestrator receives the subagent's final assistant message text as the tool result. There is no streaming of intermediate subagent thoughts back into the orchestrator's prompt (though the subagent's own UI does stream its progress). This is by design — it enforces the "summary, not raw transcript" contract.
- **Parallel cap of 4.** `MAX_CONCURRENCY = 4` and `MAX_PARALLEL_TASKS = 8` are hardcoded. Fine for typical orchestrator-worker use but not configurable per-invocation.
- **No automatic re-planning.** Chain mode is rigid: if step 2 fails, the orchestrator (the LLM driving the `subagent` tool) decides what to do next; there is no built-in retry/replan within the tool itself.
- **Security model requires care.** Project-local agents (`.pi/agents/*.md`) can inject arbitrary system prompts. Default is `user` scope only; opting into `project` requires `agentScope: "both"` and prompts the user. This is good — but if `confirmProjectAgents` is set to `false` by an extension wrapper, that boundary disappears.
- **No multi-orchestrator topology.** This is single-orchestrator-N-workers. Pi has no built-in mesh of orchestrators delegating to each other (which would map to A2A territory anyway).

## "Not implemented" caveats

- ❌ Subagent tool in core (it's an example extension)
- ❌ In-process subagent spawning (process boundary is mandatory)
- ❌ Configurable parallelism cap
- ❌ Built-in subagent failure recovery / retry policy
- ❌ Streaming partial summaries from subagent to orchestrator's context
- ❌ Hierarchical subagent trees (subagent spawning sub-subagents) — would work in principle (each is just a `pi` process) but no first-class support
- ❌ Cross-process A2A protocol (this is local subprocess subagents, not remote agent-to-agent)

What Pi does ship that the orchestrator-worker pattern asks for:

- ✅ True context isolation per subagent (separate process)
- ✅ Three orchestration shapes (single / parallel / chain) in one tool
- ✅ Per-subagent system prompt + model + tool allowlist via on-disk `.md` files
- ✅ Discovery convention (user-scope vs project-scope)
- ✅ Structured-summary contract (the agent definition documents the expected output format)
- ✅ Security boundary against repo-controlled agents (interactive confirmation)
- ✅ Usage tracking (tokens, cost, turns) per subagent surfaced to the orchestrator
