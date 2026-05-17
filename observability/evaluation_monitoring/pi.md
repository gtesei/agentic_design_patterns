# Pi — Evaluation & Monitoring

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Monitoring is meaningfully implemented. Evaluation is much weaker.** Pi exposes structured events, streaming updates, token/cost/context telemetry, and machine-readable output modes. What I did **not** find is a built-in evaluator that scores outputs against references or runs automated quality benchmarks as part of the normal runtime.

So the honest characterization is: **strong monitoring, weak built-in evaluation**.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Structured session/runtime events | ✅ `packages/coding-agent/src/core/agent-session.ts` |
| Core agent/tool lifecycle events | ✅ `packages/agent/src/types.ts` |
| Streaming assistant event model | ✅ `packages/ai/src/types.ts` |
| Machine-readable event output | ✅ `packages/coding-agent/src/modes/print-mode.ts` |
| Live token/cost/context display | ✅ `packages/coding-agent/src/modes/interactive/components/footer.ts` |
| Built-in output evaluator / benchmark harness in runtime | ❌ not meaningfully implemented |

## Key code excerpts

Source: `packages/coding-agent/src/core/agent-session.ts:120-140`

```ts
export type AgentSessionEvent =
	| AgentEvent
	| { type: "queue_update"; steering: readonly string[]; followUp: readonly string[] }
	| { type: "compaction_start"; reason: "manual" | "threshold" | "overflow" }
	| {
			type: "compaction_end";
			reason: "manual" | "threshold" | "overflow";
			result: CompactionResult | undefined;
			aborted: boolean;
			willRetry: boolean;
			errorMessage?: string;
	  }
	| { type: "auto_retry_start"; attempt: number; maxAttempts: number; delayMs: number; errorMessage: string }
	| { type: "auto_retry_end"; success: boolean; attempt: number; finalError?: string };
```

Why this matters: Pi surfaces a rich lifecycle for queueing, compaction, and retries instead of treating those as invisible internals.

Source: `packages/agent/src/types.ts:405-418`

```ts
| { type: "agent_start" }
| { type: "agent_end"; messages: AgentMessage[] }
| { type: "turn_start" }
| { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
| { type: "message_start"; message: AgentMessage }
| { type: "message_update"; message: AgentMessage; assistantMessageEvent: AssistantMessageEvent }
| { type: "message_end"; message: AgentMessage }
| { type: "tool_execution_start"; toolCallId: string; toolName: string; args: any }
| { type: "tool_execution_update"; toolCallId: string; toolName: string; args: any; partialResult: any }
| { type: "tool_execution_end"; toolCallId: string; toolName: string; result: any; isError: boolean };
```

Why this matters: the monitoring model extends all the way down to turn-level and tool-level execution.

Source: `packages/ai/src/types.ts:347-359`

```ts
export type AssistantMessageEvent =
	| { type: "start"; partial: AssistantMessage }
	| { type: "text_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
	| { type: "thinking_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "thinking_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
	| { type: "done"; reason: Extract<StopReason, "stop" | "length" | "toolUse">; message: AssistantMessage }
```

Why this matters: Pi exposes fine-grained streaming events, which makes deep monitoring possible for UIs and external consumers.

Source: `packages/coding-agent/src/modes/print-mode.ts:103-115`

```ts
unsubscribe = session.subscribe((event) => {
	if (mode === "json") {
		writeRawStdout(`${JSON.stringify(event)}\n`);
	}
});

if (mode === "json") {
	const header = session.sessionManager.getHeader();
	if (header) {
		writeRawStdout(`${JSON.stringify(header)}\n`);
	}
}
```

Why this matters: monitoring is not just for the terminal UI. Pi can emit machine-readable event streams for downstream tooling.

Source: `packages/coding-agent/src/modes/interactive/components/footer.ts:29-32,68-89,118-139`

```ts
/**
 * Footer component that shows pwd, token stats, and context usage.
 * Computes token/context stats from session...
 */
...
for (const entry of this.session.sessionManager.getEntries()) {
	if (entry.type === "message" && entry.message.role === "assistant") {
		totalInput += entry.message.usage.input;
		totalOutput += entry.message.usage.output;
		totalCacheRead += entry.message.usage.cacheRead;
		totalCacheWrite += entry.message.usage.cacheWrite;
		totalCost += entry.message.usage.cost.total;
	}
}
...
const costStr = `$${totalCost.toFixed(3)}${usingSubscription ? " (sub)" : ""}`;
```

Why this matters: Pi treats cost, cache, and context-window pressure as first-class runtime observability signals.

## Tradeoffs and limitations

- Pi is strong on operational observability.
- I did not find an equally strong built-in evaluation layer that measures answer quality against expected results or datasets.
- If a team wants real evaluation in Pi, they would likely need an external harness or an extension-built scoring loop.

## Final word

Pi meaningfully implements **monitoring**. It does **not** meaningfully implement built-in **evaluation** to the same degree.
