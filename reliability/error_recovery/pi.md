# Pi — Error Recovery

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Meaningfully implemented.** Pi has real recovery paths for several common failure modes:

- transient provider / transport failures trigger automatic retry with backoff
- context overflow triggers automatic compaction and one retry
- tool failures are converted into structured tool-result errors instead of crashing the loop

This is one of the clearer reliability patterns in Pi's core runtime.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Retry settings and defaults | ✅ `packages/coding-agent/src/core/settings-manager.ts` |
| Post-turn retry/compaction recovery flow | ✅ `packages/coding-agent/src/core/agent-session.ts` |
| Context overflow detection | ✅ `packages/ai/src/utils/overflow.ts` |
| Tool-call failure isolation | ✅ `packages/agent/src/agent-loop.ts` |

## Key code excerpts

Source: `packages/coding-agent/src/core/settings-manager.ts:25-30,721-726`

```ts
export interface RetrySettings {
	enabled?: boolean; // default: true
	maxRetries?: number; // default: 3
	baseDelayMs?: number; // default: 2000 (exponential backoff: 2s, 4s, 8s)
	provider?: ProviderRetrySettings;
}

getRetrySettings(): { enabled: boolean; maxRetries: number; baseDelayMs: number } {
	return {
		enabled: this.getRetryEnabled(),
		maxRetries: this.settings.retry?.maxRetries ?? 3,
		baseDelayMs: this.settings.retry?.baseDelayMs ?? 2000,
	};
}
```

Why this matters: retry is not a vague aspiration in Pi; it is configurable runtime behavior with explicit defaults.

Source: `packages/coding-agent/src/core/agent-session.ts:572-585`

```ts
// Check auto-retry and auto-compaction after agent completes
if (event.type === "agent_end" && this._lastAssistantMessage) {
	const msg = this._lastAssistantMessage;
	...
	if (this._isRetryableError(msg)) {
		const didRetry = await this._handleRetryableError(msg);
		if (didRetry) return;
	}

	this._resolveRetry();
	await this._checkCompaction(msg);
}
```

Why this matters: Pi has a deliberate recovery order. It tries transient-error recovery first, then context repair.

Source: `packages/ai/src/utils/overflow.ts:122-150`

```ts
export function isContextOverflow(message: AssistantMessage, contextWindow?: number): boolean {
	if (message.stopReason === "error" && message.errorMessage) {
		...
		if (!isNonOverflow && OVERFLOW_PATTERNS.some((p) => p.test(message.errorMessage!))) {
			return true;
		}
	}

	if (contextWindow && message.stopReason === "stop") {
		const inputTokens = message.usage.input + message.usage.cacheRead;
		if (inputTokens > contextWindow) {
			return true;
		}
	}
	...
}
```

Why this matters: overflow recovery is based on centralized detection logic, including both explicit provider errors and usage-based overflow cases.

Source: `packages/coding-agent/src/core/agent-session.ts:1759-1815`

```ts
 * Two cases:
 * 1. Overflow: LLM returned context overflow error, remove error message from agent state, compact, auto-retry
 * 2. Threshold: Context over threshold, compact, NO auto-retry (user continues manually)
...
if (sameModel && isContextOverflow(assistantMessage, contextWindow)) {
	...
	this._overflowRecoveryAttempted = true;
	...
	if (messages.length > 0 && messages[messages.length - 1].role === "assistant") {
		this.agent.state.messages = messages.slice(0, -1);
	}
	await this._runAutoCompaction("overflow", true);
	return;
}
```

Why this matters: Pi does real reactive recovery when the provider rejects a request for overflow, including stripping the failed assistant error from live context before retrying.

Source: `packages/coding-agent/src/core/agent-session.ts:2410-2505`

```ts
private _isRetryableError(message: AssistantMessage): boolean {
	...
	return /overloaded|provider.?returned.?error|rate.?limit|...|timeout|terminated|retry delay/i.test(err);
}

private async _handleRetryableError(message: AssistantMessage): Promise<boolean> {
	...
	this._retryAttempt++;
	...
	const delayMs = settings.baseDelayMs * 2 ** (this._retryAttempt - 1);
	...
	await sleep(delayMs, this._retryAbortController.signal);
	...
	setTimeout(() => {
		this.agent.continue().catch(() => {});
	}, 0);
	return true;
}
```

Why this matters: transient-failure recovery is automatic and uses exponential backoff, rather than forcing the user to manually re-run every time.

Source: `packages/agent/src/agent-loop.ts:552-600,652-676`

```ts
if (!tool) {
	return {
		kind: "immediate",
		result: createErrorToolResult(`Tool ${toolCall.name} not found`),
		isError: true,
	};
}
...
} catch (error) {
	return {
		kind: "immediate",
		result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
		isError: true,
	};
}
...
} catch (error) {
	result = createErrorToolResult(error instanceof Error ? error.message : String(error));
	isError = true;
}
```

Why this matters: tool failures become structured tool results that the agent loop can continue around; they do not automatically tear down the entire run.

## Tradeoffs and limitations

- Retryability is regex-based, so behavior still depends on provider error strings unless providers normalize errors well.
- Overflow recovery is intentionally conservative: Pi attempts one compact-and-retry cycle, not unlimited retries.
- Compaction-based recovery is lossy, so success depends on the quality of the generated summary.

## Final word

Pi has a substantial, implementation-level error-recovery story: retries, overflow compaction, and tool-failure containment are all built into the runtime.
