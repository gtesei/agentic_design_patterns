# Pi: Parallelization

Accessed on: 2026-05-17

## Verdict

Parallelization is meaningfully implemented in Pi. The clearest implementations are:

- parallel tool execution in the core agent loop
- per-tool overrides to force sequential execution when needed
- an example `subagent` extension that runs multiple delegated tasks concurrently with a concurrency cap

## Core parallel tool execution

Pi models tool execution mode explicitly:

Source: `packages/agent/src/types.ts:29-36`

```ts
export type ToolExecutionMode = "sequential" | "parallel";
```

Why this matters: parallelization is not accidental. It is a named policy in the core type system.

The default agent behavior is parallel:

Source: `packages/agent/src/agent.ts:209-215`

```ts
this.steeringQueue = new PendingMessageQueue(options.steeringMode ?? "one-at-a-time");
this.followUpQueue = new PendingMessageQueue(options.followUpMode ?? "one-at-a-time");
this.sessionId = options.sessionId;
this.thinkingBudgets = options.thinkingBudgets;
this.transport = options.transport ?? "auto";
this.maxRetryDelayMs = options.maxRetryDelayMs;
this.toolExecution = options.toolExecution ?? "parallel";
```

Why this matters: callers have to opt out of concurrency. Pi assumes concurrent tool execution is normal.

The loop chooses sequential or parallel execution per turn:

Source: `packages/agent/src/agent-loop.ts:380-387`

```ts
const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
const hasSequentialToolCall = toolCalls.some(
	(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
);
if (config.toolExecution === "sequential" || hasSequentialToolCall) {
	return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
}
return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
```

Why this matters: Pi supports both a global policy and a per-tool safety override.

The parallel path launches prepared tool executions and waits with `Promise.all(...)`:

Source: `packages/agent/src/agent-loop.ts:447-505`

```ts
async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			...
			continue;
		}

		finalizedCalls.push(async () => {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return finalized;
		});
	}

	const orderedFinalizedCalls = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
```

Why this matters: this is the concrete concurrency mechanism for parallel tool use.

## Parallel delegated agents

Pi's `subagent` example also supports parallel delegated work:

Source: `packages/coding-agent/examples/extensions/subagent/index.ts:556-618`

```ts
if (params.tasks && params.tasks.length > 0) {
	if (params.tasks.length > MAX_PARALLEL_TASKS)
		return {
			content: [
				{
					type: "text",
					text: `Too many parallel tasks (${params.tasks.length}). Max is ${MAX_PARALLEL_TASKS}.`,
				},
			],
			details: makeDetails("parallel")([]),
		};

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
			(partial) => {
				if (partial.details?.results[0]) {
					allResults[index] = partial.details.results[0];
					emitParallelUpdate();
				}
			},
			makeDetails("parallel"),
		);
		allResults[index] = result;
		emitParallelUpdate();
		return result;
	});
```

Why this matters: Pi demonstrates parallelization at a second layer too, not just within a single assistant turn.

## Tradeoffs and limitations

- Tool preparation still happens in a controlled loop before the async executions are awaited, so Pi is not "fully parallel everywhere."
- Per-tool sequential overrides are necessary because some tools are unsafe to run concurrently.
- The subagent example bounds parallel fan-out with `MAX_PARALLEL_TASKS` and `MAX_CONCURRENCY`, which is safer but means it is not an unbounded swarm architecture.
