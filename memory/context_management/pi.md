# Pi — Context Management

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17

## Summary

**Yes — strong implementation.** Pi treats context engineering as a first-class concern. The framework layer (`@earendil-works/pi-agent-core`) exposes an explicit `transformContext` hook fired before every LLM call, with documented use cases for pruning and external-context injection. The product layer (`coding-agent`) ships a full 845-line **compaction** module: token estimation, threshold-based triggering, LLM-driven summarization, structural file-operation tracking that survives compaction (a reversible-compression flavor), branch summarization for parallel paths, and persistence of the compaction summary as a typed session entry.

Token estimation is a `chars / 4` heuristic — fast and provider-agnostic, with no tiktoken dependency. The tradeoff is intentional: Pi prefers a conservative overestimate over byte-accurate counting.

## Where it lives

| Concern | File |
|---|---|
| `transformContext` / `shouldStopAfterTurn` / `prepareNextTurn` hooks | `packages/agent/src/types.ts` (`AgentLoopConfig`) |
| Compaction triggers, token estimation, cut-point detection, summarization | `packages/coding-agent/src/core/compaction/compaction.ts` (845 lines) |
| Branch-of-execution summarization | `packages/coding-agent/src/core/compaction/branch-summarization.ts` (355 lines) |
| File operation extraction (kept across compaction) | `packages/coding-agent/src/core/compaction/utils.ts` |
| Compaction summary persisted as a session entry | `packages/coding-agent/src/core/session-manager.ts` (`CompactionEntry`) |
| Summarization system prompt | `packages/coding-agent/src/core/compaction/utils.ts` (`SUMMARIZATION_SYSTEM_PROMPT`) |

## Key code excerpts

### Framework hook — `transformContext` runs before every LLM call

```ts
// packages/agent/src/types.ts (AgentLoopConfig)
/**
 * Optional transform applied to the context before `convertToLlm`.
 *
 * Use this for operations that work at the AgentMessage level:
 * - Context window management (pruning old messages)
 * - Injecting context from external sources
 *
 * Contract: must not throw or reject. Return the original messages or another
 * safe fallback value instead.
 *
 * @example
 * ```typescript
 * transformContext: async (messages) => {
 *   if (estimateTokens(messages) > MAX_TOKENS) {
 *     return pruneOldMessages(messages);
 *   }
 *   return messages;
 * }
 * ```
 */
transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
```

**Why relevant:** This is the pluggable seam. Any consumer of `@earendil-works/pi-agent-core` can implement its own context strategy — prune, summarize, reorder, inject retrieved docs — without forking the loop. The docstring explicitly calls out "pruning old messages" and "injecting context from external sources," which maps to the **reduce** and **retrieve** moves from Anthropic's context-engineering taxonomy.

### Graceful-stop hook — yield before the window blows

```ts
// packages/agent/src/types.ts (AgentLoopConfig)
/**
 * If it returns true, the loop emits `agent_end` and exits before polling steering or follow-up queues,
 * without starting another LLM call.
 *
 * Use this to request a graceful stop after the current turn, e.g. before context gets too full.
 */
shouldStopAfterTurn?: (context: ShouldStopAfterTurnContext) => boolean | Promise<boolean>;
```

**Why relevant:** Lets the consumer detect "next turn would overflow" and exit cleanly rather than waiting for the provider to error mid-turn. Pi puts the decision in the consumer's hands rather than enforcing a built-in policy at the framework layer — appropriate for a foundation package.

### Compaction trigger — pure-function threshold

```ts
// packages/coding-agent/src/core/compaction/compaction.ts
export function shouldCompact(
  contextTokens: number,
  contextWindow: number,
  settings: CompactionSettings,
): boolean {
  if (!settings.enabled) return false;
  return contextTokens > contextWindow - settings.reserveTokens;
}
```

**Why relevant:** Compaction fires when current usage exceeds `window - reserveTokens`. The reserve is the buffer kept free for the next assistant response. This is the simplest possible policy and exactly what you want — no thrashing near the edge, no premature compaction.

### Token estimation — chars / 4, deliberately conservative

```ts
// packages/coding-agent/src/core/compaction/compaction.ts
/**
 * Estimate token count for a message using chars/4 heuristic.
 * This is conservative (overestimates tokens).
 */
export function estimateTokens(message: AgentMessage): number {
  let chars = 0;
  switch (message.role) {
    case "user": {
      const content = (message as { content: string | Array<{ type: string; text?: string }> }).content;
      if (typeof content === "string") {
        chars = content.length;
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === "text" && block.text) {
            chars += block.text.length;
          }
        }
      }
      return Math.ceil(chars / 4);
    }
    case "assistant": {
      const assistant = message as AssistantMessage;
      for (const block of assistant.content) {
        if (block.type === "text") chars += block.text.length;
        else if (block.type === "thinking") chars += block.thinking.length;
        else if (block.type === "toolCall") chars += block.name.length + JSON.stringify(block.arguments).length;
      }
      return Math.ceil(chars / 4);
    }
    // ... toolResult / custom similar
  }
}
```

**Why relevant:** Pi does its own counting — no tiktoken, no provider call. The chars/4 heuristic is intentionally fuzzy and over-counts on average, which is fine because the goal is "don't blow the window" not "fill it to the byte." This avoids a heavy dependency and works uniformly across providers (Anthropic, OpenAI, Gemini have different tokenizers; the loss-of-precision is the cost of provider-neutrality).

### Compaction main entry — LLM-summarize, retain file structure

```ts
// packages/coding-agent/src/core/compaction/compaction.ts
export async function compact(
  preparation: CompactionPreparation,
  model: Model<any>,
  apiKey: string,
  headers?: Record<string, string>,
  customInstructions?: string,
  signal?: AbortSignal,
  thinkingLevel?: ThinkingLevel,
): Promise<CompactionResult> {
  // ...
  // Just generate history summary
  summary = await generateSummary(
    messagesToSummarize, model, settings.reserveTokens,
    apiKey, headers, signal, customInstructions, previousSummary, thinkingLevel,
  );

  // Compute file lists and append to summary
  const { readFiles, modifiedFiles } = computeFileLists(fileOps);
  summary += formatFileOperations(readFiles, modifiedFiles);

  return {
    summary,
    firstKeptEntryId,
    tokensBefore,
    details: { readFiles, modifiedFiles } as CompactionDetails,
  };
}
```

**Why relevant:** Two things that distinguish this from a naive "summarize and forget" loop:

1. **`previousSummary` is forwarded** so a second compaction can build on the first, not re-summarize from scratch.
2. **File operation lists are extracted from tool calls in the messages being summarized** and persisted in `details.readFiles` / `details.modifiedFiles`. This is **reversible compression**: even after lossy summarization, the agent still knows which files it touched. On the next turn, that structural info is appended to the summary and the agent can re-read any of those files by path. This is the "drop file contents, keep paths" move from Anthropic's context-engineering essay.

### Persisted compaction entry — typed record in the session log

```ts
// packages/coding-agent/src/core/session-manager.ts
export interface CompactionEntry<T = unknown> extends SessionEntryBase {
  type: "compaction";
  summary: string;
  firstKeptEntryId: string;
  tokensBefore: number;
  /** Extension-specific data (e.g., ArtifactIndex, version markers for structured compaction) */
  details?: T;
  /** True if generated by an extension, undefined/false if pi-generated */
  fromHook?: boolean;
}
```

**Why relevant:** Compaction is part of the session log — durable, replayable. On resume, `buildSessionContext` reads the compaction entries and reconstructs the compacted view, so a long session that was compacted three times yesterday looks the same today. The `fromHook` flag lets extensions hand-roll compaction and have it interoperate.

### Summarization prompt

```ts
// packages/coding-agent/src/core/compaction/utils.ts
export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;
```

**Why relevant:** The summarization model is sent a system prompt that explicitly tells it not to roleplay forward — a known failure mode when summarizing partial conversations. Small detail, real impact.

### Parallel summarization for split turns

When compaction lands in the middle of a turn (assistant message + pending tool results), Pi splits the summary into two: history-up-to-turn-boundary and turn-prefix. Both are generated in parallel:

```ts
// packages/coding-agent/src/core/compaction/compaction.ts
if (isSplitTurn && turnPrefixMessages.length > 0) {
  const [historyResult, turnPrefixResult] = await Promise.all([
    messagesToSummarize.length > 0
      ? generateSummary(messagesToSummarize, model, /* ... */)
      : Promise.resolve("No prior history."),
    generateTurnPrefixSummary(turnPrefixMessages, model, /* ... */),
  ]);
  summary = `${historyResult}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult}`;
}
```

**Why relevant:** Compaction in the middle of a multi-tool turn is the hard case. Pi handles it by generating two summaries and concatenating, rather than dropping the partial turn. Parallelism keeps latency down.

## Tradeoffs and limitations

- **Token estimation is heuristic, not exact.** Chars/4 overestimates for code-heavy content and underestimates for languages with longer tokens. Fine for window management; insufficient if you need precise cost forecasting per turn.
- **Compaction is sync-blocking per session.** When `shouldCompact` returns true, the next turn waits on `generateSummary` (one LLM call, possibly two for split turns). No background pre-compaction.
- **One summarization strategy per session.** The summarization prompt is fixed (with optional `customInstructions` to append). No hierarchical / topic-clustered summarization out of the box — but extensions can hand-roll compaction via the `fromHook` flag.
- **No semantic relevance scoring of older messages.** Compaction is "summarize everything before the cut point." There is no "keep these specific old messages because they're still relevant; drop the rest."
- **No `retrieve` move at framework level.** The `transformContext` hook *enables* retrieval-style injection but Pi ships no built-in retriever. That's a coding-agent app concern, not a framework primitive.
- **The framework-level `transformContext` hook has a strict no-throw contract.** If your implementation can fail, you must catch internally and return safe-fallback messages — this is documented but easy to violate.

## "Not implemented" caveats

- ❌ Exact token counting via provider-specific tokenizers
- ❌ Background / async compaction
- ❌ Per-message importance scoring or relevance retention
- ❌ Built-in retrieval injection (RAG-style "fetch top-k docs and prepend")
- ❌ Hierarchical / topic-clustered summarization out of the box

What Pi does ship that the pattern asks for:

- ✅ Compaction triggered on configurable window thresholds (`shouldCompact`)
- ✅ LLM-driven summarization with prior-summary chaining
- ✅ Reversible compression — file paths preserved after content summarized away
- ✅ Branch summarization for split conversation paths
- ✅ Mid-turn split-summary handling
- ✅ Persisted compaction entries that survive session resume
- ✅ Framework hook (`transformContext`) for custom strategies
- ✅ Graceful-stop hook (`shouldStopAfterTurn`) to exit before overflow
