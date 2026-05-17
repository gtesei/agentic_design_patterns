# Pi: Reflection

Accessed on: 2026-05-17

## Verdict

Pi does not appear to implement reflection as a first-class core pattern. I did not find a built-in "generate answer, critique it, revise it" stage in the main agent loop.

The closest implementation is an example-level workflow built on the `subagent` extension, where one agent produces work, a second agent reviews it, and the first agent applies feedback.

## What the core loop does not do

The main loop alternates between assistant responses and tool execution, then ends the turn:

Source: `packages/agent/src/agent-loop.ts:192-218`

```ts
const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
newMessages.push(message);

if (message.stopReason === "error" || message.stopReason === "aborted") {
	await emit({ type: "turn_end", message, toolResults: [] });
	await emit({ type: "agent_end", messages: newMessages });
	return;
}

const toolCalls = message.content.filter((c) => c.type === "toolCall");

const toolResults: ToolResultMessage[] = [];
hasMoreToolCalls = false;
if (toolCalls.length > 0) {
	const executedToolBatch = await executeToolCalls(currentContext, message, config, signal, emit);
	toolResults.push(...executedToolBatch.messages);
	hasMoreToolCalls = !executedToolBatch.terminate;

	for (const result of toolResults) {
		currentContext.messages.push(result);
		newMessages.push(result);
	}
}

await emit({ type: "turn_end", message, toolResults });
```

Why this matters: there is no explicit reflection checkpoint here. The loop does not insert a built-in critique pass after an answer.

## Where reflection shows up

The clearest reflection-like workflow is the `implement-and-review` prompt template:

Source: `packages/coding-agent/examples/extensions/subagent/prompts/implement-and-review.md:4-10`

```md
Use the subagent tool with the chain parameter to execute this workflow:

1. First, use the "worker" agent to implement: $@
2. Then, use the "reviewer" agent to review the implementation from the previous step (use {previous} placeholder)
3. Finally, use the "worker" agent to apply the feedback from the review (use {previous} placeholder)

Execute this as a chain, passing output between steps via {previous}.
```

Why this matters: this is a reflection pattern in practice, but it is authored as a workflow prompt, not built into the agent runtime.

The reviewer role is also explicitly defined:

Source: `packages/coding-agent/examples/extensions/subagent/agents/reviewer.md:8-35`

```md
You are a senior code reviewer. Analyze code for quality, security, and maintainability.

Bash is for read-only commands only: `git diff`, `git log`, `git show`. Do NOT modify files or run builds.
Assume tool permissions are not perfectly enforceable; keep all bash usage strictly read-only.

Strategy:
1. Run `git diff` to see recent changes (if applicable)
2. Read the modified files
3. Check for bugs, security issues, code smells

Output format:

## Files Reviewed
- `path/to/file.ts` (lines X-Y)

## Critical (must fix)
- `file.ts:42` - Issue description
...
```

Why this matters: Pi models reflection here as role-specialized critique, not as a generic self-reflection primitive.

The chain executor feeds each step's output into the next step through `{previous}`:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts:501-549`

```ts
if (params.chain && params.chain.length > 0) {
	const results: SingleResult[] = [];
	let previousOutput = "";

	for (let i = 0; i < params.chain.length; i++) {
		const step = params.chain[i];
		const taskWithContext = step.task.replace(/\{previous\}/g, previousOutput);
		...
		const result = await runSingleAgent(
			ctx.cwd,
			agents,
			step.agent,
			taskWithContext,
			step.cwd,
			i + 1,
			signal,
			chainUpdate,
			makeDetails("chain"),
		);
		results.push(result);
		...
		previousOutput = getFinalOutput(result.messages);
	}
```

Why this matters: the critique-revision loop is implemented by chaining subprocess agents with text handoff.

## Tradeoffs and limitations

- Reflection exists, but mostly as an example/workflow layer instead of a guaranteed runtime behavior.
- Handoff between steps is plain text, so critique is not strongly typed or structurally validated.
- Because reflection is extension-driven, it costs extra turns and subprocess overhead when used.
- If the pattern requires a built-in self-critique loop in the main runtime, Pi does not meaningfully implement that in core.
