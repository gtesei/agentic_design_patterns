# Pi implementation notes — Context Management

**Accessed on:** 2026-05-15  
**Scope:** `earendil-works/pi` (inspected local clone)

## 1) Implementation status
- **Implemented in core:** context budgeting, automatic/manual compaction, branch summarization, deterministic context reconstruction.

## 2) Code evidence (with line refs)

### A. Context budget and compaction thresholds
**Source:** `packages/agent/src/harness/compaction/compaction.ts:112-115,196-199`

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

Why this matters: Pi has explicit, code-level token policy for when context must be compressed.

### B. Reconstructing LLM context after compaction
**Source:** `packages/agent/src/harness/session/session.ts:57-58,66`

```ts
if (compaction) {
  messages.push(createCompactionSummaryMessage(compaction.summary, compaction.tokensBefore, compaction.timestamp));
  ...
  for (let i = compactionIdx + 1; i < pathEntries.length; i++) {
    appendMessage(pathEntries[i]!);
  }
}
```

Why this matters: this is the operational mechanism that keeps recent context while replacing old context with a summary.

### C. Branch-context preservation on tree navigation
**Source:** `packages/agent/src/harness/compaction/branch-summarization.ts:69-96`

```ts
export async function collectEntriesForBranchSummary(session, oldLeafId, targetId): CollectEntriesResult {
  ...
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

Why this matters: context from abandoned branches is preserved as summarized state instead of being silently lost.

### D. Stable prompt envelope for summaries
**Source:** `packages/agent/src/harness/messages.ts:4-10,147,151`

```ts
export const COMPACTION_SUMMARY_PREFIX = `The conversation history before this point was compacted into the following summary:\n\n<summary>\n`;
...
case "compactionSummary":
  return {
    role: "user",
    content: [{ type: "text", text: COMPACTION_SUMMARY_PREFIX + m.summary + COMPACTION_SUMMARY_SUFFIX }],
    timestamp: m.timestamp,
  };
```

Why this matters: summaries are injected consistently and parseably into model input.

## 3) Tradeoffs / limitations
- Compaction is intentionally lossy for active context.
- Token estimation may use heuristics (`chars/4`) when provider usage is unavailable.
- Quality of retained context depends on summary quality.

## 4) Pattern mapping
Pi is a strong implementation reference for context management patterns (budgeting + summarization + replay across branches).