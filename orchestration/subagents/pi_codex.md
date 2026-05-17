# Pi implementation notes: subagents

Accessed on: 2026-05-17

Verdict: Pi does not appear to implement subagents as a first-class core runtime primitive in `packages/agent` or `packages/coding-agent/src/core`. The concrete implementation I found is an example extension at `packages/coding-agent/examples/extensions/subagent/`. That still counts as a real implementation pattern in Pi, but it is extension-driven rather than built into the base agent loop.

## Relevant Pi code

- `packages/coding-agent/examples/extensions/subagent/index.ts`
- `packages/coding-agent/examples/extensions/subagent/agents.ts`
- `packages/coding-agent/examples/extensions/subagent/agents/*.md`
- `packages/coding-agent/examples/extensions/subagent/README.md`

## 1. Pi exposes subagents as a tool, not a core scheduler

The example registers a normal Pi tool named `subagent`:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts`

```ts
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
		...
	});
}
```

Why this matters: in Pi, subagents are implemented as an extensible orchestration tool. They are not hardcoded into the core `AgentSession` or `agent-loop` abstractions.

## 2. Isolation comes from spawning separate `pi` processes

Each delegated task runs as a separate CLI process in JSON mode:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts`

```ts
const args: string[] = ["--mode", "json", "-p", "--no-session"];
if (agent.model) args.push("--model", agent.model);
if (agent.tools && agent.tools.length > 0) args.push("--tools", agent.tools.join(","));
...
if (agent.systemPrompt.trim()) {
	const tmp = await writePromptToTempFile(agent.name, agent.systemPrompt);
	...
	args.push("--append-system-prompt", tmpPromptPath);
}
...
const proc = spawn(invocation.command, invocation.args, {
	cwd: cwd ?? defaultCwd,
	shell: false,
	stdio: ["ignore", "pipe", "pipe"],
});
```

Why this matters: the key subagent mechanism is process isolation. Each subagent gets its own context window, model selection, tool allowlist, and appended system prompt.

The README states the same design directly:

Source: `packages/coding-agent/examples/extensions/subagent/README.md`

```md
- **Isolated context**: Each subagent runs in a separate `pi` process
```

## 3. Agent definitions are markdown prompts with optional tool/model constraints

Subagents are discovered from markdown files rather than from special compiled agent objects:

Source: `packages/coding-agent/examples/extensions/subagent/agents.ts`

```ts
const { frontmatter, body } = parseFrontmatter<Record<string, string>>(content);
...
agents.push({
	name: frontmatter.name,
	description: frontmatter.description,
	tools: tools && tools.length > 0 ? tools : undefined,
	model: frontmatter.model,
	systemPrompt: body,
	source,
	filePath,
});
```

A sample agent file looks like this:

Source: `packages/coding-agent/examples/extensions/subagent/agents/worker.md`

```md
---
name: worker
description: General-purpose subagent with full capabilities, isolated context
model: claude-sonnet-4-5
---

You are a worker agent with full capabilities. You operate in an isolated context window to handle delegated tasks without polluting the main conversation.
```

Why this matters: Pi's subagent pattern is prompt-defined and resource-driven. New subagents are mostly data files plus extension code, not framework subclasses.

## 4. The extension supports single, parallel, and chained delegation

Parallel work is implemented explicitly in the extension with a concurrency cap:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts`

```ts
const MAX_PARALLEL_TASKS = 8;
const MAX_CONCURRENCY = 4;
...
const results = await mapWithConcurrencyLimit(params.tasks, MAX_CONCURRENCY, async (t, index) => {
	const result = await runSingleAgent(
		ctx.cwd,
		agents,
		t.agent,
		t.task,
		t.cwd,
		undefined,
		signal,
		...
	);
	...
	return result;
});
```

Chained orchestration passes the previous agent's final output into the next task through a text placeholder:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts`

```ts
for (let i = 0; i < params.chain.length; i++) {
	const step = params.chain[i];
	const taskWithContext = step.task.replace(/\{previous\}/g, previousOutput);
	...
	previousOutput = getFinalOutput(result.messages);
}
```

Why this matters: Pi's subagent orchestration semantics are explicit and inspectable. "Parallel" and "chain" are not hidden magic; they are implemented in ordinary extension code.

## 5. Project-local agents are treated as a trust boundary

The example defaults to user-scoped agents and prompts before running project-local ones:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts`

```ts
if ((agentScope === "project" || agentScope === "both") && confirmProjectAgents && ctx.hasUI) {
	...
	const ok = await ctx.ui.confirm(
		"Run project-local agents?",
		`Agents: ${names}\nSource: ${dir}\n\nProject agents are repo-controlled. Only continue for trusted repositories.`,
	);
	if (!ok) {
		return {
			content: [{ type: "text", text: "Canceled: project-local agents not approved." }],
			...
		};
	}
}
```

Why this matters: Pi treats subagents as executable delegation prompts. The implementation acknowledges that repo-local agent definitions are powerful enough to need an approval step.

## Architectural tradeoffs and limitations

- This is not a first-class core feature. The implementation lives in an example extension, so applications must opt in and maintain the orchestration code themselves.
- Isolation is strong because each subagent is a separate `pi` process, but state sharing is weak. The chain mode mostly passes plain text (`{previous}`), not a structured shared workspace memory object.
- The example launches subagents with `--no-session`, so delegated runs are ephemeral by default.
- Parallelism is intentionally capped (`MAX_PARALLEL_TASKS = 8`, `MAX_CONCURRENCY = 4`), which is pragmatic but not a general scheduler.
- The pattern is flexible because subagents are just markdown-defined prompts, but that also means quality and safety depend heavily on prompt authoring and tool restrictions.

If you need to document Pi conservatively, the strongest claim is: Pi supports subagents as an extension pattern and provides a complete reference implementation, but subagents are not currently a built-in core abstraction on the same level as sessions, skills, or compaction.
