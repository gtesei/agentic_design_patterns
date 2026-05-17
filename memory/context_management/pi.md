# Pi — Context Management (revised)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates findings from `pi.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, accessed 2026-05-15), and `pi_codex.md` (Codex agent read, accessed 2026-05-17). Every excerpt below was re-verified against a fresh `git clone` of the repo on 2026-05-17.

## Summary

**Yes — comprehensive implementation, broader than just compaction.** Pi treats context engineering as four cooperating concerns:

1. **Project-context loading.** Pi walks from the working directory up to root, collecting `AGENTS.md` / `CLAUDE.md` files at each ancestor, and layers them into the system prompt. Repo instructions are part of the live working context, not just chat history.
2. **System-prompt assembly.** Skills, project context files, runtime state (`cwd`, `date`), tool snippets, and a custom prompt are composed into the system prompt explicitly and rebuilt whenever resources change.
3. **Framework hook (`transformContext`).** Below the coding-agent, the generic agent loop exposes a pre-send transform that consumers can use to prune, reorder, summarize, or inject messages.
4. **Compaction (the canonical Anthropic-style pattern).** ~845 lines of token estimation, threshold detection, cut-point boundary-finding, LLM-driven summarization, branch summarization for tree navigation, and durable persistence of summaries as session entries.

Compaction logic exists at **two layers**: `packages/agent/src/harness/compaction/` (framework) and `packages/coding-agent/src/core/compaction/` (app). The constants `DEFAULT_COMPACTION_SETTINGS` are defined in both locations — duplicated rather than re-exported, which is a maintenance hazard worth flagging.

> Addition to my earlier `pi.md`: I missed the *project-context layering* pattern (`AGENTS.md`/`CLAUDE.md` walk) and the explicit slash-command/system-prompt assembly. Those are the strongest evidence that Pi treats context engineering as broader than "compact when full."

## Where it lives

| Concern | File |
|---|---|
| Project-context file discovery (cwd → root walk) | `packages/coding-agent/src/core/resource-loader.ts` (`loadProjectContextFiles`, `loadContextFileFromDir`) |
| System-prompt assembly | `packages/coding-agent/src/core/system-prompt.ts` |
| `transformContext` framework hook | `packages/agent/src/types.ts` (`AgentLoopConfig`), wired in `packages/agent/src/agent-loop.ts` |
| Lifecycle hooks (`shouldStopAfterTurn`, `prepareNextTurn`, steering) | `packages/agent/src/types.ts` |
| Compaction at framework level | `packages/agent/src/harness/compaction/` |
| Compaction at app level | `packages/coding-agent/src/core/compaction/` (845 lines) |
| Branch summarization | `packages/agent/src/harness/compaction/branch-summarization.ts`, `packages/coding-agent/src/core/compaction/branch-summarization.ts` |
| Compaction summary persisted as session entry | `packages/coding-agent/src/core/session-manager.ts` (`CompactionEntry`, `BranchSummaryEntry`) |
| Stable summary envelope | `packages/agent/src/harness/messages.ts` (`COMPACTION_SUMMARY_PREFIX`/`SUFFIX`) |

## Key code excerpts

### Project-context loading — walk from CWD to root

```ts
// packages/coding-agent/src/core/resource-loader.ts
function loadContextFileFromDir(dir: string): { path: string; content: string } | null {
  const candidates = ["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"];
  // ... returns first match
}

export function loadProjectContextFiles(options: {
  cwd: string;
  agentDir: string;
}): Array<{ path: string; content: string }> {
  // ...
  const globalContext = loadContextFileFromDir(resolvedAgentDir);
  // ...
  let currentDir = resolvedCwd;
  const root = resolve("/");

  while (true) {
    const contextFile = loadContextFileFromDir(currentDir);
    if (contextFile && !seenPaths.has(contextFile.path)) {
      ancestorContextFiles.unshift(contextFile);
      seenPaths.add(contextFile.path);
    }
    if (currentDir === root) break;
    const parentDir = resolve(currentDir, "..");
    // ...
    currentDir = parentDir;
  }

  contextFiles.push(...ancestorContextFiles);
  return contextFiles;
}
```

**Why relevant:** This is the **project-context layering** pattern: at every ancestor of the working directory, Pi looks for `AGENTS.md` (or `CLAUDE.md`) and includes its content in the system prompt. Both filename casings are supported. Ancestors are added in root-first order so deeper, more-specific instructions appear later in the prompt. This is a deliberate "context engineering" choice — repo-level instructions are not optional add-ons, they are part of the live context the model sees on every turn.

### System-prompt assembly — concrete composition

```ts
// packages/coding-agent/src/core/system-prompt.ts (excerpted)
if (contextFiles.length > 0) {
  prompt += "\n\n# Project Context\n\n";
  prompt += "Project-specific instructions and guidelines:\n\n";
  for (const { path: filePath, content } of contextFiles) {
    prompt += `## ${filePath}\n\n${content}\n\n`;
  }
}

if (hasRead && skills.length > 0) {
  prompt += formatSkillsForPrompt(skills);
}

prompt += `\nCurrent date: ${date}`;
prompt += `\nCurrent working directory: ${promptCwd}`;
```

**Why relevant:** Context composition is **explicit and inspectable** — readable Markdown with section headers, not a hidden middleware layer. Skills are included only when the `read` tool is available (without `read`, the skill files can't be loaded — see skills `pi2.md`). Runtime facts (date, cwd) are appended last.

This is refreshed whenever the session rebuilds its system prompt:

```ts
// packages/coding-agent/src/core/agent-session.ts (excerpted)
const loadedSkills = this._resourceLoader.getSkills().skills;
const loadedContextFiles = this._resourceLoader.getAgentsFiles().agentsFiles;

this._baseSystemPromptOptions = {
  cwd: this._cwd,
  skills: loadedSkills,
  contextFiles: loadedContextFiles,
  customPrompt: loaderSystemPrompt,
  appendSystemPrompt,
  selectedTools: validToolNames,
  toolSnippets,
  promptGuidelines,
};
return buildSystemPrompt(this._baseSystemPromptOptions);
```

**Why relevant:** Prompt assembly is part of normal session lifecycle, not a one-time bootstrap. If a user edits `AGENTS.md` mid-session and the resource loader picks it up, the next assembly reflects the change.

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

```ts
// packages/agent/src/agent-loop.ts (wiring)
let messages = context.messages;
if (config.transformContext) {
  messages = await config.transformContext(messages, signal);
}

const llmMessages = await config.convertToLlm(messages);
```

**Why relevant:** This is the framework-level seam for any context strategy beyond compaction — RAG injection, custom pruning, message reordering. The docstring explicitly calls out "pruning old messages" and "injecting context from external sources," matching the **reduce** and **retrieve** moves from Anthropic's context-engineering taxonomy.

### Compaction settings — explicit defaults

```ts
// packages/agent/src/harness/compaction/compaction.ts (also at coding-agent layer)
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
  enabled: true,
  reserveTokens: 16384,
  keepRecentTokens: 20000,
};
```

**Why relevant:** Pi reserves 16k tokens of headroom for the next assistant response, and tries to keep ~20k tokens of recent context as-is. Concrete numbers, easy to override. The settings live in *both* the framework `harness/compaction/` and the app `coding-agent/src/core/compaction/` — duplicated (verified — both files contain the same constant, neither re-exports from the other). This is a known papercut: changing the framework default doesn't auto-update the coding-agent default.

### Compaction trigger — two paths, threshold vs overflow

```ts
// packages/coding-agent/src/core/agent-session.ts (excerpted)
// Threshold path: proactive
if (shouldCompact(contextTokens, contextWindow, settings)) {
  await this._runAutoCompaction("threshold", false);
}

// Overflow path: reactive (after provider rejected the request)
await this._runAutoCompaction("overflow", true);
```

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

**Why relevant:** Pi compacts **proactively** when current usage exceeds `window - reserveTokens`, and also has a **reactive** path triggered when an actual provider call overflows (with `willRetry: true` so the failed call is retried after compaction). Two layers of safety, both implemented.

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
    case "user": { /* count text content */ }
    case "assistant": {
      const assistant = message as AssistantMessage;
      for (const block of assistant.content) {
        if (block.type === "text") chars += block.text.length;
        else if (block.type === "thinking") chars += block.thinking.length;
        else if (block.type === "toolCall") chars += block.name.length + JSON.stringify(block.arguments).length;
      }
      return Math.ceil(chars / 4);
    }
    // ...
  }
}
```

**Why relevant:** Pi does its own counting — no tiktoken, no provider call. The chars/4 heuristic is intentionally fuzzy and over-counts on average, which is fine for "don't blow the window" but insufficient if you need exact cost forecasting. Uniform across providers (Anthropic / OpenAI / Gemini have different tokenizers; the loss-of-precision is the cost of provider-neutrality).

### Cut-point detection — walk back from newest

```ts
// packages/coding-agent/src/core/compaction/compaction.ts (excerpted)
for (let i = endIndex - 1; i >= startIndex; i--) {
  const entry = entries[i];
  if (entry.type !== "message") continue;

  const messageTokens = estimateTokens(entry.message);
  accumulatedTokens += messageTokens;

  if (accumulatedTokens >= keepRecentTokens) {
    for (let c = 0; c < cutPoints.length; c++) {
      if (cutPoints[c] >= i) {
        cutIndex = cutPoints[c];
        break;
      }
    }
    break;
  }
}
```

**Why relevant:** Cut-point finding is **deterministic and explicit**, not a vague "summarize when something feels long." Walk backwards from the newest message accumulating tokens; once you've crossed `keepRecentTokens`, snap to the nearest safe boundary (turn boundaries are pre-computed as `cutPoints`). This avoids cutting in the middle of a turn or splitting a tool call from its result.

### Compaction main entry — LLM-summarize, retain file structure

```ts
// packages/coding-agent/src/core/compaction/compaction.ts (excerpted)
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
```

**Why relevant:** Two things that distinguish this from naive "summarize and forget":

1. **`previousSummary` is forwarded** so a second compaction can build on the first.
2. **File operation lists** are extracted from tool calls and persisted in `details.readFiles` / `details.modifiedFiles`. This is **reversible compression**: after lossy summarization the agent still knows which files it touched, and can re-read any of them by path. This is the "drop file contents, keep paths" move from Anthropic's context-engineering essay.

### Stable summary envelope

```ts
// packages/agent/src/harness/messages.ts
export const COMPACTION_SUMMARY_PREFIX = `The conversation history before this point was compacted into the following summary:\n\n<summary>\n`;
export const COMPACTION_SUMMARY_SUFFIX = `\n</summary>`;

// In the renderer:
case "compactionSummary":
  return {
    role: "user",
    content: [{ type: "text", text: COMPACTION_SUMMARY_PREFIX + m.summary + COMPACTION_SUMMARY_SUFFIX }],
    timestamp: m.timestamp,
  };
```

**Why relevant:** Summaries are injected as user-role messages with a stable prefix/suffix envelope. The model knows the boundary; downstream tools could parse it. Consistency matters: every compacted session has the same shape.

### Branch summarization — preserve abandoned branches

```ts
// packages/coding-agent/src/core/compaction/branch-summarization.ts (excerpted)
export async function collectEntriesForBranchSummary(
  session, oldLeafId, targetId,
): CollectEntriesResult {
  // ... walks from oldLeafId back to the common ancestor with targetId
  while (current && current !== commonAncestorId) {
    const entry = session.getEntry(current);
    if (!entry) break;
    entries.push(entry);
    current = entry.parentId;
  }
  entries.reverse();
  return { entries, commonAncestorId };
}
```

```ts
// packages/coding-agent/src/core/agent-session.ts (excerpted)
const { entries: entriesToSummarize, commonAncestorId } = collectEntriesForBranchSummary(
  this.sessionManager, oldLeafId, targetId,
);
// ...
const result = await generateBranchSummary(entriesToSummarize, { /* ... */ });
// ...
const summaryId = this.sessionManager.branchWithSummary(
  newLeafId, summaryText, summaryDetails, fromExtension,
);
```

**Why relevant:** When the user navigates from one branch of the session tree to another, Pi collects the entries on the abandoned branch (from old leaf back to the common ancestor with the new target), summarizes them, and attaches that summary as a `BranchSummaryEntry`. This is context management across **branching histories**, not just along a linear timeline. Abandoned branch work is preserved in condensed form rather than discarded.

### Parallel summarization for split turns

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

**Why relevant:** Compaction in the middle of a multi-tool turn is the hard case. Pi handles it with two parallel LLM calls (history-up-to-turn-boundary + turn-prefix), concatenated. Parallelism keeps latency down.

### Summarization prompt — explicit no-roleplay-forward

```ts
// packages/coding-agent/src/core/compaction/utils.ts
export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;
```

**Why relevant:** The summarization model is told **not to roleplay forward** — a known failure mode when feeding a partial conversation to a model. Small detail, real impact.

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

**Why relevant:** Lets the consumer detect "next turn would overflow" and exit cleanly rather than waiting for the provider to error mid-turn. Pi delegates the *decision* to the consumer rather than enforcing built-in policy at the framework layer — appropriate for a foundation package.

## Tradeoffs and limitations

- **Token estimation is heuristic, not exact.** Chars/4 overestimates for code-heavy content. Fine for window management; insufficient if you need precise per-turn cost forecasting.
- **Compaction is sync-blocking per session.** When `shouldCompact` returns true, the next turn waits on `generateSummary` (one LLM call, two for split turns). No background pre-compaction.
- **Compaction settings are duplicated** at the framework and app layer (`DEFAULT_COMPACTION_SETTINGS` exists in both). Drift hazard.
- **Project context files are appended wholesale.** No relevance ranking before they enter the prompt — if `AGENTS.md` is 50k tokens, all 50k go in every turn.
- **One summarization strategy per session.** Prompt is fixed (with optional `customInstructions`). Extensions can hand-roll via `fromHook`, but there's no built-in hierarchical / topic-clustered summarization.
- **No semantic relevance scoring of older messages.** Compaction is "summarize everything before the cut point." There's no "keep these specific old messages because they're still relevant."
- **No built-in retriever.** `transformContext` *enables* retrieval-style injection, but Pi ships no retriever — that's a coding-agent-app concern.
- **The `transformContext` hook has a strict no-throw contract.** If your implementation can fail, you must catch internally and return a safe-fallback message list.
- **Branch summaries are local to one session.** They preserve context along explicit session branches; they are not a global retrieval system.

## "Not implemented" caveats

- ❌ Exact tokenization via provider-specific tokenizers
- ❌ Background / async compaction
- ❌ Per-message importance / relevance scoring
- ❌ Built-in retrieval injection (RAG-style "fetch top-k docs and prepend")
- ❌ Hierarchical / topic-clustered summarization out of the box
- ❌ Selective inclusion of project-context files (it's all-or-nothing per file)

What Pi does ship that the context-engineering pattern asks for:

- ✅ Project-context file discovery (`AGENTS.md` / `CLAUDE.md` walk from CWD to root)
- ✅ Explicit system-prompt assembly (skills + context + tool snippets + cwd/date)
- ✅ Compaction triggered on configurable window thresholds (proactive + reactive)
- ✅ LLM-driven summarization with prior-summary chaining
- ✅ Reversible compression — file paths preserved after content summarized away
- ✅ Branch summarization for tree navigation, persisted as session entries
- ✅ Mid-turn split-summary handling (parallel)
- ✅ Compaction summaries persisted as `CompactionEntry` records — survive resume
- ✅ Framework hook (`transformContext`) for custom strategies
- ✅ Graceful-stop hook (`shouldStopAfterTurn`) and turn-replacement hook (`prepareNextTurn`)
- ✅ Mid-run steering message injection (`getSteeringMessages`)
- ✅ Stable summary envelope (`COMPACTION_SUMMARY_PREFIX`/`SUFFIX`)
