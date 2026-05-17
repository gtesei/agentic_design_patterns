# Pi — Memory Management (revised)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates findings from `pi.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, accessed 2026-05-15), and `pi_codex.md` (Codex agent read, accessed 2026-05-17). Every excerpt below was re-verified against a fresh `git clone` of the repo on 2026-05-17.

## Summary

**Partial — but stronger than my first read suggested.** Pi implements **branchable session persistence** as a first-class concern at the framework layer (`packages/agent/src/harness/session/`), not only at the app layer. A session is a tree of typed append-only entries with a leaf pointer; the harness exposes `create` / `open` / `list` / `fork` / `delete` operations on a `SessionRepo` interface, with two concrete implementations (`JsonlSessionRepo` for on-disk durability, `InMemorySessionRepo` for ephemeral / SDK / test use).

What Pi does **not** implement is the canonical 2026 long-term-memory pattern (LangMem / Letta / Mem0 / Zep style): no vector store, no embedding-based retrieval, no episodic/semantic/procedural scope distinction, no automatic write-to-memory after each turn, no recall API exposed to the LLM. Memory primitives are not skills the model can invoke. Cross-session and cross-project recall is not automatic.

The shape is: *durable structured transcript with branching and replay* — not *queryable knowledge base*.

> Correction to my earlier `pi.md`: I claimed the agent-core framework "knows nothing about persistence." That was wrong. The harness layer in `packages/agent/src/harness/session/` is the framework's persistence story; only the *in-process* `AgentState.messages` is unpersisted.

## Where it lives

| Concern | File |
|---|---|
| Session repo interface, create/open/list/fork | `packages/agent/src/harness/session/jsonl-repo.ts` |
| JSONL append-only storage (header + entries + leaf) | `packages/agent/src/harness/session/jsonl-storage.ts` |
| In-memory repo for ephemeral runs | `packages/agent/src/harness/session/memory-repo.ts` |
| In-memory storage backing it | `packages/agent/src/harness/session/memory-storage.ts` |
| Active-session class with tree navigation | `packages/agent/src/harness/session/session.ts` |
| App-layer session manager (compaction, branch labels, replay) | `packages/coding-agent/src/core/session-manager.ts` (1458 lines) |
| App-layer agent session runtime | `packages/coding-agent/src/core/agent-session.ts` (3110 lines) |
| CLI resume entry points | `packages/coding-agent/src/cli/args.ts`, `src/main.ts` |
| In-process agent state (one run, no persistence) | `packages/agent/src/types.ts` (`AgentState`) |

There is no `memory` package, no `store` package, no `recall` module. Branchable session storage is the only memory primitive.

## Key code excerpts

### Framework-level session API — repo + storage interfaces

```ts
// packages/agent/src/harness/session/jsonl-repo.ts
export class JsonlSessionRepo implements JsonlSessionRepoApi {
  async create(options: JsonlSessionCreateOptions): Promise<Session<JsonlSessionMetadata>> { /* ... */ }
  async open(metadata: JsonlSessionMetadata): Promise<Session<JsonlSessionMetadata>> { /* ... */ }
  async list(options: JsonlSessionListOptions = {}): Promise<JsonlSessionMetadata[]> { /* ... */ }
  async fork(
    sourceMetadata: JsonlSessionMetadata,
    options: JsonlSessionCreateOptions & {
      entryId?: string;
      position?: "before" | "at";
      id?: string;
    },
  ): Promise<Session<JsonlSessionMetadata>> { /* ... */ }
}
```

**Why relevant:** This is the canonical memory API at the framework layer. Any consumer of `@earendil-works/pi-agent-core` can persist, list, and fork sessions without depending on the coding-agent app. The `fork` method takes an `entryId` plus a `position: "before" | "at"` — the caller picks an exact point in the existing session and a new branch is created starting at (or just before) that entry. This is the "branchable memory" capability: prior work is not overwritten, it is preserved as an alternate branch.

### JSONL storage — append-only with header + entries + leaf pointer

```ts
// packages/agent/src/harness/session/jsonl-storage.ts
interface SessionHeader {
  type: "session";
  version: 3;
  id: string;
  timestamp: string;
  cwd: string;
  parentSession?: string;
}

const header: SessionHeader = {
  type: "session",
  version: 3,
  id: options.sessionId,
  timestamp: new Date().toISOString(),
  cwd: options.cwd,
  parentSession: options.parentSessionPath,
};
await fs.writeFile(filePath, `${JSON.stringify(header)}\n`);
// ...
async appendEntry(entry: SessionTreeEntry): Promise<void> {
  await this.fs.appendFile(this.filePath, `${JSON.stringify(entry)}\n`);
  this.entries.push(entry);
  this.byId.set(entry.id, entry);
  this.currentLeafId = leafIdAfterEntry(entry);
}
```

**Why relevant:** Memory is an auditable log: one header line, then JSON-per-line entries. No mutable hidden state — every change is an entry, every entry has an `id` and `parentId` that builds the tree. Pi uses **uuidv7** for entry IDs (`packages/agent/src/harness/session/uuid.ts`), so IDs are time-ordered and collision-resistant. `cwd` lives on the header — sessions are keyed to the project they were started in.

### Branch navigation — explicit leaf pointer

```ts
// packages/agent/src/harness/session/jsonl-storage.ts
async setLeafId(leafId: string | null): Promise<void> {
  if (leafId !== null && !this.byId.has(leafId)) {
    throw new SessionError("not_found", `Entry ${leafId} not found`);
  }
  const entry: LeafEntry = {
    type: "leaf",
    id: generateEntryId(this.byId),
    parentId: this.currentLeafId,
    timestamp: new Date().toISOString(),
    targetId: leafId,
  };
  await this.fs.appendFile(this.filePath, `${JSON.stringify(entry)}\n`);
  // ...
  this.currentLeafId = leafId;
}
```

**Why relevant:** Switching branches is itself an event in the log. A `LeafEntry` records "we jumped from current leaf to `targetId`." This is auditable rewind/replay — you can see in the file exactly when the user navigated away from one conversation branch and into another.

### CWD-keyed session directory layout

```ts
// packages/agent/src/harness/session/jsonl-repo.ts
function encodeCwd(cwd: string): string {
  return `--${cwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
}
```

**Why relevant:** Sessions are organized on disk under a directory derived from the working directory they were created in. `--Users-AG62216-var-agentic_design_patterns--/` etc. Listing sessions for the current project means listing one directory; cross-project memory means walking other directories (no built-in API for that).

### Compaction summary — lossy checkpoint, persisted as an entry

```ts
// packages/coding-agent/src/core/session-manager.ts
export interface CompactionEntry<T = unknown> extends SessionEntryBase {
  type: "compaction";
  summary: string;
  firstKeptEntryId: string;
  tokensBefore: number;
  /** Extension-specific data (e.g., ArtifactIndex, version markers for structured compaction) */
  details?: T;
  /** True if generated by an extension, undefined/false if pi-generated (backward compatible) */
  fromHook?: boolean;
}
```

```ts
// packages/coding-agent/src/core/agent-session.ts (excerpted)
this.sessionManager.appendCompaction(summary, firstKeptEntryId, tokensBefore, details, fromExtension);
const sessionContext = this.sessionManager.buildSessionContext();
this.agent.state.messages = sessionContext.messages;
```

**Why relevant:** Compaction summaries are part of the same append-only log. On reload, `buildSessionContext` replays the entries — when it hits a compaction entry it inserts a special user-role "the conversation before this was summarized" message, then continues from `firstKeptEntryId`. The model-visible memory is the summary plus everything after; the raw history before the cut still lives in the file and is accessible via `/tree` (the CLI quote from the README: *"Compaction is lossy. The full history remains in the JSONL file; use `/tree` to revisit."*).

### Replay-on-reload — `buildSessionContext`

```ts
// packages/agent/src/harness/session/session.ts (excerpted)
if (compaction) {
  messages.push(createCompactionSummaryMessage(compaction.summary, compaction.tokensBefore, compaction.timestamp));
  const compactionIdx = pathEntries.findIndex((e) => e.type === "compaction" && e.id === compaction.id);
  let foundFirstKept = false;
  for (let i = 0; i < compactionIdx; i++) {
    const entry = pathEntries[i]!;
    if (entry.id === compaction.firstKeptEntryId) foundFirstKept = true;
    if (foundFirstKept) appendMessage(entry);
  }
  for (let i = compactionIdx + 1; i < pathEntries.length; i++) {
    appendMessage(pathEntries[i]!);
  }
} else {
  for (const entry of pathEntries) {
    appendMessage(entry);
  }
}
```

**Why relevant:** The replay algorithm is deterministic. Given the same JSONL file and the same compaction entry, you get the same model-visible context. This is what makes Pi's session memory auditable: there's no hidden interpretation layer.

### Ephemeral mode — `InMemorySessionRepo`

```ts
// packages/agent/src/harness/session/memory-repo.ts
export class InMemorySessionRepo implements SessionRepo<SessionMetadata, { id?: string }, void> {
  private sessions = new Map<string, Session<SessionMetadata>>();
  // ...
}
```

```ts
// packages/coding-agent/src/core/session-manager.ts
static inMemory(cwd: string = process.cwd()): SessionManager {
  return new SessionManager(cwd, "", undefined, false);
}
```

**Why relevant:** Memory persistence is configurable. Tests, SDK use cases that don't want files on disk, and one-shot agent runs can swap in `InMemorySessionRepo` — same API, no JSONL file. The same shape (`SessionRepo<...>` interface) means application code is decoupled from the storage choice.

### Extension memory hook — `CustomEntry`

```ts
// packages/coding-agent/src/core/session-manager.ts
/**
 * Custom entry for extensions to store extension-specific data in the session.
 * Use customType to identify your extension's entries.
 *
 * Purpose: Persist extension state across session reloads. On reload, extensions can
 * scan entries for their customType and reconstruct internal state.
 *
 * Does NOT participate in LLM context (ignored by buildSessionContext).
 * For injecting content into context, see CustomMessageEntry.
 */
export interface CustomEntry<T = unknown> extends SessionEntryBase {
  type: "custom";
  customType: string;
  data?: T;
}
```

**Why relevant:** This is the closest Pi gets to a "memory write" API. An extension can append a `CustomEntry` with arbitrary `data` to the session log and recover it on reload by scanning entries for its `customType`. Durable, but scoped to one session lineage and **not retrieved or summarized for the LLM** — it's a private extension-state sidecar, not a queryable knowledge base. `CustomMessageEntry` is the variant that *does* re-inject content into LLM context on reload.

### In-process agent state — still no persistence

```ts
// packages/agent/src/types.ts
export interface AgentState {
  systemPrompt: string;
  model: Model<any>;
  thinkingLevel: ThinkingLevel;
  set tools(tools: AgentTool<any>[]);
  get tools(): AgentTool<any>[];
  /** Conversation transcript. Assigning a new array copies the top-level array. */
  set messages(messages: AgentMessage[]);
  get messages(): AgentMessage[];
  readonly isStreaming: boolean;
  readonly streamingMessage?: AgentMessage;
  readonly pendingToolCalls: ReadonlySet<string>;
  readonly errorMessage?: string;
}
```

**Why relevant:** `AgentState` is the live, in-memory state of *one running agent loop*. It is intentionally separate from `Session` — agent loops don't persist themselves; persistence is the harness's job. This separation is what allows the same agent loop to run with `JsonlSessionRepo`, with `InMemorySessionRepo`, or with a future custom backend.

### `sessionId` is for provider caching, not memory

The agent-core `AgentLoopConfig` accepts a `sessionId` field forwarded to the LLM provider for prompt caching (Anthropic / OpenAI). This is **not** memory — it's a hint to the provider that the prefix of this call should hit cache from a prior call with the same `sessionId`. Easy to conflate with persistent memory because of the name; worth flagging.

## Tradeoffs and limitations

- **Branchable but not semantic.** Memory is tree-of-entries with explicit branches. There is no "given this query, retrieve top-k semantically similar past messages." Fork and replay are the navigation primitives; vector search is not in scope.
- **CWD-scoped, not cross-project.** Sessions are organized under a directory derived from their creation `cwd`. Cross-project recall ("what did I tell the agent about this library a year ago in a different repo?") would require the user to know which other session file to open.
- **No write-during-conversation memory tool.** The LLM cannot call `remember(fact)` that persists outside the current session. Extensions can do this via `CustomEntry`, but again only within one session lineage.
- **No procedural memory.** No agent rewriting its own system prompt based on what worked. System prompts are static (and refreshed per turn from the resource loader; see context_management notes).
- **Compaction is lossy.** As the Pi README explicitly states. The full history survives in the file, but the model-visible context is the summary + recent suffix.
- **Cross-session knowledge transfer is manual.** Fork copies entries from a source session into a new one (selectable by `entryId` / `position`). There's no "this session learned X; propagate X to all future sessions in this project."
- **No pluggable backends today.** `JsonlSessionRepo` (disk) and `InMemorySessionRepo` (RAM) are the two concrete implementations. The `SessionRepo` interface allows a Postgres/Redis/SQLite backend in principle but none ship.

## "Not implemented" caveats

- ❌ Vector store / embedding-based retrieval
- ❌ Episodic / semantic / procedural memory scopes
- ❌ Background memory consolidation
- ❌ Memory-as-tool (`store_fact`, `recall_facts`) exposed to the LLM
- ❌ Per-user profile / cross-session memory by default
- ❌ Memory aging, decay, or relevance scoring
- ❌ Postgres/Redis/Pinecone/Qdrant backends — only JSONL on disk and in-memory

What Pi does provide:

- ✅ Durable per-session transcript with full fidelity (JSONL append-only)
- ✅ `SessionRepo` abstraction at the framework layer (`JsonlSessionRepo`, `InMemorySessionRepo`)
- ✅ `fork(metadata, { entryId, position })` for explicit branching at chosen entries
- ✅ `setLeafId` for explicit branch navigation, recorded as its own entry
- ✅ Tree-structured history with `id` / `parentId` per entry (uuidv7)
- ✅ Resume by metadata across processes / machine restarts
- ✅ Extension sidecar (`CustomEntry`) and context-replaying variant (`CustomMessageEntry`)
- ✅ Compaction summaries persisted as durable typed entries (lossy but auditable)
- ✅ Provider-side prompt caching via `sessionId` (orthogonal to memory)
- ✅ Ephemeral mode (`SessionManager.inMemory`, `InMemorySessionRepo`) for tests and SDK
