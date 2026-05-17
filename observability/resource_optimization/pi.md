# Pi — Resource Optimization

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Meaningfully implemented.** Pi has several concrete mechanisms for optimizing scarce runtime resources:

- proactive context compaction before the hard window limit
- token estimation and checkpoint summarization
- provider-side prompt caching and session affinity hints
- explicit usage and cost accounting

This is one of the stronger implementation areas in the Pi codebase.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Default compaction reserve/headroom | ✅ `packages/coding-agent/src/core/compaction/compaction.ts` |
| Context token estimation | ✅ `packages/coding-agent/src/core/compaction/compaction.ts` |
| Structured compaction summaries | ✅ `packages/coding-agent/src/core/compaction/compaction.ts` |
| Runtime compaction integration | ✅ `packages/coding-agent/src/core/agent-session.ts` |
| Prompt caching/session affinity | ✅ `packages/ai/src/types.ts`, `packages/ai/src/providers/openai-completions.ts` |
| Cost accounting | ✅ `packages/ai/src/models.ts` |

## Key code excerpts

Source: `packages/coding-agent/src/core/compaction/compaction.ts:121-124,219-222`

```ts
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};

export function shouldCompact(contextTokens: number, contextWindow: number, settings: CompactionSettings): boolean {
	if (!settings.enabled) return false;
	return contextTokens > contextWindow - settings.reserveTokens;
}
```

Why this matters: Pi intentionally reserves headroom instead of waiting until every request hard-fails on context overflow.

Source: `packages/coding-agent/src/core/compaction/compaction.ts:182-213`

```ts
/**
 * Estimate context tokens from messages, using the last assistant usage when available.
 * If there are messages after the last usage, estimate their tokens with estimateTokens.
 */
export function estimateContextTokens(messages: AgentMessage[]): ContextUsageEstimate {
	...
	const usageTokens = calculateContextTokens(usageInfo.usage);
	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		trailingTokens += estimateTokens(messages[i]);
	}
	...
	return {
		tokens: usageTokens + trailingTokens,
		usageTokens,
		trailingTokens,
		lastUsageIndex: usageInfo.index,
	};
}
```

Why this matters: Pi does not depend purely on the last provider response. It can estimate context size when later messages have been added.

Source: `packages/coding-agent/src/core/compaction/compaction.ts:454-485`

```ts
const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
...
## Constraints & Preferences
...
## Progress
...
## Next Steps
...
Keep each section concise. Preserve exact file paths, function names, and error messages.`;
```

Why this matters: Pi optimizes context through structured checkpointing, not just by truncating older messages.

Source: `packages/coding-agent/src/core/agent-session.ts:1759-1842`

```ts
 * Two cases:
 * 1. Overflow: LLM returned context overflow error, remove error message from agent state, compact, auto-retry
 * 2. Threshold: Context over threshold, compact, NO auto-retry
...
if (shouldCompact(contextTokens, contextWindow, settings)) {
	await this._runAutoCompaction("threshold", false);
}
```

Why this matters: resource optimization is integrated into the live session runtime, not left as manual cleanup.

Source: `packages/ai/src/types.ts:95-104,396-399`

```ts
cacheRetention?: CacheRetention;
/**
 * Optional session identifier for providers that support session-based caching.
 * Providers can use this to enable prompt caching, request routing, or other
 * session-aware features.
 */
sessionId?: string;
...
sendSessionAffinityHeaders?: boolean;
supportsLongCacheRetention?: boolean;
```

Why this matters: Pi has explicit provider-facing controls for caching and session locality.

Source: `packages/ai/src/providers/openai-completions.ts:470-474,508-517`

```ts
if (sessionId && compat.sendSessionAffinityHeaders) {
	headers.session_id = sessionId;
	headers["x-client-request-id"] = sessionId;
	headers["x-session-affinity"] = sessionId;
}
...
prompt_cache_key:
	(model.baseUrl.includes("api.openai.com") && cacheRetention !== "none") ||
	(cacheRetention === "long" && compat.supportsLongCacheRetention)
		? options?.sessionId
		: undefined,
prompt_cache_retention: cacheRetention === "long" && compat.supportsLongCacheRetention ? "24h" : undefined,
```

Why this matters: the caching abstractions are not decorative. The provider adapters translate them into concrete request parameters and headers.

Source: `packages/ai/src/models.ts:39-45`

```ts
export function calculateCost<TApi extends Api>(model: Model<TApi>, usage: Usage): Usage["cost"] {
	usage.cost.input = (model.cost.input / 1000000) * usage.input;
	usage.cost.output = (model.cost.output / 1000000) * usage.output;
	usage.cost.cacheRead = (model.cost.cacheRead / 1000000) * usage.cacheRead;
	usage.cost.cacheWrite = (model.cost.cacheWrite / 1000000) * usage.cacheWrite;
	usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite;
```

Why this matters: Pi makes spend measurable, which is a core part of runtime optimization in practice.

## Tradeoffs and limitations

- Compaction is lossy; optimized context is still reduced context.
- Prompt caching benefits depend on provider support and compatibility behavior.
- Pi manages token budget and cost well, but it cannot eliminate the underlying tradeoff between rich context and finite resources.

## Final word

Pi meaningfully implements resource optimization through **compaction, caching/session affinity, and explicit usage/cost accounting**.
