# Pi implementation notes — Memory Management

**Accessed on:** 2026-05-15  
**Scope:** `earendil-works/pi` (inspected local clone)

## 1) Implementation status
- **Implemented in core:** session persistence, branchable history, summary checkpoints (compaction/branch summary entries).
- **Not implemented in core:** semantic long-term memory (vector retrieval/salience ranking across historical facts).

## 2) Code evidence (with line refs)

### A. Persistent session memory (JSONL repo)
**Source:** `packages/agent/src/harness/session/jsonl-repo.ts:38,75,92,102,133`

```ts
export class JsonlSessionRepo implements JsonlSessionRepoApi {
  async create(options: JsonlSessionCreateOptions): Promise<Session<JsonlSessionMetadata>> { ... }
  async open(metadata: JsonlSessionMetadata): Promise<Session<JsonlSessionMetadata>> { ... }
  async list(options: JsonlSessionListOptions = {}): Promise<JsonlSessionMetadata[]> { ... }
  async fork(sourceMetadata: JsonlSessionMetadata, options: ...): Promise<Session<JsonlSessionMetadata>> { ... }
}
```

Why this matters: this is the durable memory substrate Pi uses across runs.

### B. Memory artifacts represented as tree entries
**Source:** `packages/coding-agent/src/core/session-manager.ts:66-85`

```ts
export interface CompactionEntry<T = unknown> extends SessionEntryBase {
  type: "compaction";
  summary: string;
  firstKeptEntryId: string;
  tokensBefore: number;
  details?: T;
}

export interface BranchSummaryEntry<T = unknown> extends SessionEntryBase {
  type: "branch_summary";
  fromId: string;
  summary: string;
  details?: T;
}
```

Why this matters: memory is append-only and branch-aware, not mutable hidden state.

### C. Rebuilding active memory from current path
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

Why this matters: active model context uses summary + retained recent path, while full raw history remains stored.

## 3) Constraints / limitations
**Source:** `packages/coding-agent/README.md:270`

> “Compaction is lossy. The full history remains in the JSONL file; use `/tree` to revisit.”

Implication: model-visible memory is compressed and lossy, even though session memory on disk is complete.

## 4) Tradeoffs
- **Pros:** simple and auditable local persistence; robust resume/fork/navigation story.
- **Cons:** no built-in semantic recall layer; recall quality depends on summarization.

## 5) Pattern mapping
For this repo’s memory-management pattern, Pi maps best to **durable session memory + compression checkpoints**, not semantic memory retrieval systems.