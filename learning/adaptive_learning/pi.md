# Pi — Adaptive Learning

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Not meaningfully implemented.** Pi does not appear to collect feedback, update prompts from outcomes, maintain a learned skill library, retrieve past successful trajectories automatically, or fine-tune itself from prior runs.

One important distinction: Pi does mention **adaptive thinking** in provider integrations, but that is not adaptive learning. It means the model decides how much reasoning effort to use during a call, not that Pi learns from experience over time.

## Where it lives (or doesn't)

| Concern | Status in Pi |
|---|---|
| Feedback collection / ratings | ❌ none found |
| Prompt or policy adaptation from past outcomes | ❌ none found |
| Learned skill acquisition | ❌ none found |
| In-context retrieval of prior successful runs | ❌ none found |
| Provider-side adaptive thinking | ✅ `packages/ai/src/providers/anthropic.ts` |
| Persistent session history | ✅ `packages/coding-agent/src/core/session-manager.ts`, but this is memory, not learning |
| Extension state persistence primitive | ✅ `packages/coding-agent/src/core/extensions/types.ts` |

## Key code excerpts

Source: `packages/ai/src/providers/anthropic.ts:183-195`

```ts
/**
 * Enable extended thinking.
 * For Opus 4.6 and Sonnet 4.6: uses adaptive thinking (model decides when/how much to think).
 * For older models: uses budget-based thinking with thinkingBudgetTokens.
 */
thinkingEnabled?: boolean;
...
/**
 * Effort level for adaptive thinking ...
 */
```

Source: `packages/ai/src/providers/anthropic.ts:939-948`

```ts
// Configure thinking mode: adaptive ...
if (model.reasoning) {
	if (options?.thinkingEnabled) {
		...
		if (supportsAdaptiveThinking(model.id)) {
			// Adaptive thinking: Claude decides when and how much to think.
			params.thinking = { type: "adaptive", display };
```

Why this matters: this is the main place Pi uses the word "adaptive," and it refers to provider-side reasoning allocation, not a learning loop.

Source: `packages/coding-agent/src/core/session-manager.ts:696-704`

```ts
 * Manages conversation sessions as append-only trees stored in JSONL files.
 *
 * Each session entry has an id and parentId forming a tree structure. The "leaf"
 * pointer tracks the current position. Appending creates a child of the current leaf.
 * Branching moves the leaf to an earlier entry, allowing new branches without
 * modifying history.
 *
 * Use buildSessionContext() to get the resolved message list for the LLM...
```

Why this matters: Pi does persist rich history, which is a prerequisite for many learning systems, but persistence alone is not learning.

Source: `packages/coding-agent/src/core/extensions/types.ts:1187-1193`

```ts
sendUserMessage(
	content: string | (TextContent | ImageContent)[],
	options?: { deliverAs?: "steer" | "followUp" },
): void;

/** Append a custom entry to the session for state persistence (not sent to LLM). */
appendEntry<T = unknown>(customType: string, data?: T): void;
```

Why this matters: Pi gives extension authors a place to persist custom state, which could support a future learning extension, but Pi itself does not use this as a feedback-learning mechanism.

Source: `packages/coding-agent/src/core/telemetry.ts:8-12`

```ts
export function isInstallTelemetryEnabled(
	settingsManager: SettingsManager,
	telemetryEnv: string | undefined = process.env.PI_TELEMETRY,
): boolean {
	return telemetryEnv !== undefined ? isTruthyEnvFlag(telemetryEnv) : settingsManager.getEnableInstallTelemetry();
}
```

Why this matters: telemetry exists in a narrow sense, but telemetry is not the same thing as adaptive learning.

## Tradeoffs and limitations

- Pi is easier to reason about because behavior is mostly explicit rather than self-modifying.
- The downside is that Pi does not improve itself from prior successes or failures without external machinery.
- A determined integrator could build adaptive learning on top of session persistence and extension hooks, but that capability is not present in the repository today.

## Final word

Pi does **not** meaningfully implement adaptive learning. The framework exposes a few ingredients that could support it, but no in-repo component closes the loop from experience to improved future behavior.
