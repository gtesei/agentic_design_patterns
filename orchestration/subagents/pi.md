# Pi — Subagents (Orchestrator-Worker) (revised)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates findings from `pi.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, accessed 2026-05-15), and `pi_codex.md` (Codex agent read, accessed 2026-05-17). Every excerpt below was re-verified against a fresh `git clone` of the repo on 2026-05-17.

## Summary

**Yes, fully — but Pi's maintainers explicitly chose to keep it out of core.** The orchestrator-worker pattern is implemented as an **opt-in extension example** at `packages/coding-agent/examples/extensions/subagent/`. The decision is documented verbatim in both the README and `docs/usage.md`:

> **No sub-agents.** There's many ways to do this. Spawn pi instances via tmux, or build your own with extensions, or install a package that does it your way.
> *(packages/coding-agent/README.md:474)*

> It intentionally does not include built-in MCP, sub-agents, permission popups, plan mode, to-dos, or background bash. You can build or install those workflows as extensions or packages.
> *(packages/coding-agent/docs/usage.md:275)*

The extension delivers true **process-level context isolation**: each subagent runs in a separate `pi` subprocess (`spawn`), with its own model, tool allowlist, system prompt loaded from a `.md` agent definition file, and — critically — `--no-session` so the subagent transcript is ephemeral and never pollutes the orchestrator's session log. The orchestrator parses the subagent's JSON event stream from stdout and feeds the final assistant message back as the tool result. Three orchestration modes: single, parallel (concurrency-limited), and chain (sequential with `{previous}` placeholder).

This pattern is closer to Anthropic's "fresh-context subagent" architecture than to in-process agent-as-tool wrappers that share message arrays — process boundaries provide OS-level isolation, not just message-array isolation.

## Where it lives

| Concern | File |
|---|---|
| Extension entry point: tool registration, mode dispatch, subprocess spawn | `packages/coding-agent/examples/extensions/subagent/index.ts` (987 lines) |
| Agent discovery from `~/.pi/agent/agents/*.md` and `<repo>/.pi/agents/*.md` | `packages/coding-agent/examples/extensions/subagent/agents.ts` |
| Sample subagent definitions (frontmatter + body) | `packages/coding-agent/examples/extensions/subagent/agents/{scout,planner,reviewer,worker}.md` |
| Sample workflow prompts (multi-step chains) | `packages/coding-agent/examples/extensions/subagent/prompts/{implement,scout-and-plan,implement-and-review}.md` |
| Extension README — security model, modes, install instructions | `packages/coding-agent/examples/extensions/subagent/README.md` |
| Maintainer's "no sub-agents in core" statement | `packages/coding-agent/README.md:474`, `packages/coding-agent/docs/usage.md:275` |

Pi's **core** subagent surface is empty by design: `packages/coding-agent/src/core/tools/` contains only `bash`, `edit`, `find`, `grep`, `ls`, `read`, `write` (verified). No `task`, `delegate`, `subagent`, `spawn`, or `worker` tool. No equivalent in `packages/agent/`.

## Key code excerpts

### Explicit non-core stance — from the docs themselves

```md
<!-- packages/coding-agent/README.md (excerpt) -->
**No sub-agents.** There's many ways to do this. Spawn pi instances via tmux,
or build your own with extensions, or install a package that does it your way.
```

```md
<!-- packages/coding-agent/docs/usage.md (excerpt) -->
It intentionally does not include built-in MCP, sub-agents, permission popups,
plan mode, to-dos, or background bash. You can build or install those workflows
as extensions or packages, or use external tools such as containers and tmux.
```

**Why relevant:** This is a deliberate architectural stance, not a missing feature. Pi's core stays small and the multi-agent topology is a composition concern. The strongest claim a reader should make: *"Pi supports subagents as an extension pattern and provides a complete reference implementation, but subagents are not a built-in core abstraction on the same level as sessions, skills, or compaction."*

### Tool registration — one tool, three modes

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

**Why relevant:** A single LLM-facing tool handles all three orchestration shapes via parameter union. The validation `modeCount === 1` enforces exactly one mode per invocation. The three modes correspond directly to canonical orchestrator-worker variants:

- **single**: one task → one subagent → one summary back.
- **parallel**: N independent tasks → up to `MAX_CONCURRENCY = 4` subagents concurrent (with `MAX_PARALLEL_TASKS = 8` queue cap).
- **chain**: orchestrator sequences subagents, with `{previous}` substituted into the next task's prompt. Equivalent to sequential handoff with isolated contexts at each hop.

### Subprocess spawn — true context isolation via separate `pi` process

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts
import { spawn } from "node:child_process";

// ... inside runSingleAgent():
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

const exitCode = await new Promise<number>((resolve) => {
  const invocation = getPiInvocation(args);
  const proc = spawn(invocation.command, invocation.args, {
    cwd: cwd ?? defaultCwd,
    shell: false,
    stdio: ["ignore", "pipe", "pipe"],
  });
  // ... read JSON events line-by-line from proc.stdout
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

**Why relevant:** Each subagent invocation is a separate OS process running the `pi` binary in **JSON mode** (`--mode json`), **non-interactive** (`-p`), and crucially **`--no-session`** — the subagent's run is not persisted to disk as a session file. The orchestrator passes none of its own context to the child; the child boots fresh with only the system prompt (loaded from the agent's `.md` file via `--append-system-prompt`) and the user-provided task string as the prompt.

This is the strongest form of context isolation available without containerization. Stronger than in-process "agent-as-tool" patterns that share Python objects or message arrays. The price: subprocess spawn overhead (~100ms+ cold start) and a one-shot result interface — the orchestrator only sees what the child writes to stdout as JSON events.

### Subagent definition — markdown file with frontmatter (Claude-Code-like)

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

**Why relevant:** A subagent definition is just a markdown file. Frontmatter declares `name`, `description`, optional `tools` allowlist, optional `model`. The body becomes the subagent's system prompt. This mirrors the Claude Code subagent convention closely. The "Output format when finished" instruction is the *contract* — it tells the subagent to produce a structured summary the orchestrator can paste back as a tool result.

### Agent discovery — disk-scoped, user vs project trust boundary

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
  // reads *.md files, parses YAML frontmatter, builds AgentConfig
  // ignores files without name + description in frontmatter
}
```

**Why relevant:** Agents are discovered from `~/.pi/agent/agents/` (user scope) or `<repo>/.pi/agents/` (project scope). Default is `user` only. Loading `project` agents requires explicit opt-in via the `agentScope` parameter, with an interactive confirmation prompt for repo-controlled agents:

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts (excerpted)
if ((agentScope === "project" || agentScope === "both") && confirmProjectAgents && ctx.hasUI) {
  // ...
  const ok = await ctx.ui.confirm(
    "Run project-local agents?",
    `Agents: ${names}\nSource: ${dir}\n\nProject agents are repo-controlled. Only continue for trusted repositories.`,
  );
  if (!ok) return { content: [{ type: "text", text: "Canceled: project-local agents not approved." }], /* ... */ };
}
```

**Why relevant:** A security boundary. Cloning an untrusted repo with a `.pi/agents/malicious.md` does not give that repo the ability to silently spawn a subagent with arbitrary system prompt and tool access — the user is prompted. `confirmProjectAgents: true` is the default; turning it off is an explicit choice.

### Parallel mode — concurrency cap via helper

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts (excerpted)
const MAX_PARALLEL_TASKS = 8;
const MAX_CONCURRENCY = 4;

if (params.tasks.length > MAX_PARALLEL_TASKS) {
  return {
    content: [{ type: "text", text: `Too many parallel tasks (${params.tasks.length}). Max is ${MAX_PARALLEL_TASKS}.` }],
    // ...
  };
}

const results = await mapWithConcurrencyLimit(params.tasks, MAX_CONCURRENCY, async (t, index) => {
  const result = await runSingleAgent(
    ctx.cwd, agents, t.agent, t.task, t.cwd,
    undefined, signal, /* ... */,
  );
  return result;
});
```

**Why relevant:** Two limits in play: `MAX_PARALLEL_TASKS = 8` is the maximum queue size accepted by the tool; `MAX_CONCURRENCY = 4` is the maximum simultaneously-running subprocesses. The helper `mapWithConcurrencyLimit` queues tasks beyond the concurrency limit and starts new ones as others finish. Both are hardcoded constants — not configurable per invocation. (Correction to my earlier `pi.md`: I conflated these two numbers.)

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

**Why relevant:** Chain mode is the orchestrator's view of inter-subagent communication. There is **no shared message history** — only the previous subagent's final text is substituted into the next subagent's task. Each step boots a fresh `pi` process with no memory of the prior step except the explicit `{previous}` slot. This is the "structured summary as the interface" pattern that Anthropic's multi-agent research system advocates.

### Sample workflow templates

The repo ships three workflow prompts:

- `prompts/implement.md` — scout → planner → worker (recon, then plan, then execute)
- `prompts/scout-and-plan.md` — scout → planner (no implementation)
- `prompts/implement-and-review.md` — worker → reviewer → worker (implement, review, fix)

These are not code — they are human-readable prompt templates that walk the orchestrator LLM through invoking the `subagent` tool in chain mode with the right `agent` + `task` per step. They are loaded into the orchestrator's prompt via the standard Pi `prompts/` discovery (separate from `agents/`).

## Tradeoffs and limitations

- **Extension, not core.** Users must opt in by symlinking the extension into `~/.pi/agent/extensions/`. The core distribution of `pi` does not include subagent capability out of the box. Defensible (keeps the binary minimal and the security surface smaller) but means subagent capability is not universally available across Pi installs.
- **Subagent runs are ephemeral by default.** `--no-session` means the subagent's full transcript is not persisted as a session file. The orchestrator captures the final result and usage stats; intermediate tool calls are visible in the orchestrator's UI as they stream but are not durable.
- **Process isolation costs latency.** Each subagent invocation is a full `pi` subprocess boot (~100ms+ cold start, plus model latency). Heavier than in-process agent-as-tool patterns, but gives true OS-level isolation — a misbehaving subagent cannot crash or corrupt the orchestrator process.
- **Result interface is one-shot.** Orchestrator sees the subagent's final assistant message text as the tool result. No streaming of intermediate subagent thoughts back into the orchestrator's prompt (the subagent's own UI does stream its progress). This is by design — it enforces the "summary, not raw transcript" contract.
- **Hardcoded concurrency limits.** `MAX_PARALLEL_TASKS = 8`, `MAX_CONCURRENCY = 4`. Not configurable per invocation; would require editing the extension.
- **No automatic re-planning.** Chain mode is rigid: if step 2 fails, the orchestrator (the LLM driving the `subagent` tool) decides what to do next; there is no built-in retry/replan within the tool itself.
- **Chain communication is plain text, not structured.** `{previous}` is a string substitution into the next task's prompt. No structured shared workspace, no typed handoff schema. Quality depends on the upstream subagent producing parseable output.
- **Security model requires care.** Project-local agents (`.pi/agents/*.md`) can inject arbitrary system prompts. Default is `user` scope only; opting into `project` requires `agentScope: "both"` and prompts the user. Good defaults — but if an extension wrapper sets `confirmProjectAgents: false`, that boundary disappears.
- **No multi-orchestrator topology.** Single-orchestrator-N-workers. No built-in mesh of orchestrators delegating to each other (which would map to A2A territory).

## "Not implemented" caveats

- ❌ Subagent tool in core (it's an example extension)
- ❌ In-process subagent spawning (process boundary is mandatory)
- ❌ Configurable parallelism cap
- ❌ Built-in subagent failure recovery / retry policy
- ❌ Streaming partial summaries from subagent into orchestrator's prompt
- ❌ Structured workspace memory shared across chain steps
- ❌ Hierarchical subagent trees (subagent spawning sub-subagents) as a first-class feature — would work in principle (each is just a `pi` process) but no scaffolding
- ❌ Cross-process A2A protocol — this is local subprocess subagents, not remote agent-to-agent

What Pi does ship that the orchestrator-worker pattern asks for:

- ✅ True context isolation per subagent (separate process)
- ✅ Three orchestration shapes (single / parallel / chain) in one tool
- ✅ Per-subagent system prompt + model + tool allowlist via on-disk `.md` files
- ✅ Discovery convention (user-scope vs project-scope)
- ✅ Structured-summary contract (agent definition documents the expected output format)
- ✅ Security boundary against repo-controlled agents (interactive confirmation, opt-in default)
- ✅ Usage tracking (tokens, cost, turns) per subagent surfaced to the orchestrator
- ✅ Ephemeral subagent runs by default (`--no-session`)
- ✅ Complete reference implementation with sample agents (scout/planner/reviewer/worker) and workflow templates (implement / scout-and-plan / implement-and-review)
