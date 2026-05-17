# Pi — Memory Management

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17

## Summary

**Partial.** Pi implements **session persistence** — a session's full transcript and metadata are durably written to disk as an append-only log, and the user can resume any prior session by ID via `--continue` / `--resume`. This gives the agent a working notion of "remember our last conversation."

What Pi does **not** implement is the canonical 2026 long-term-memory pattern (LangMem / Letta / Mem0 / Zep style): there is no vector store, no episodic/semantic/procedural scope distinction, no automatic write-to-memory after each turn, and no retrieval API the LLM can call to recall facts learned in a previous session. Memory primitives are not exposed as tools.

If a pattern reader expects "agent learns about me across sessions and surfaces relevant facts," Pi does not ship that. If they expect "the conversation I had yesterday is still there and I can pick it up exactly where I left off," Pi does ship that.

## Where it lives

| Concern | File |
|---|---|
| Session schema, on-disk format, read/write/resume | `packages/coding-agent/src/core/session-manager.ts` (1458 lines) |
| Live session runtime (mutations, compaction triggers) | `packages/coding-agent/src/core/agent-session.ts` (3110 lines) |
| Session resume from CLI | `packages/coding-agent/src/cli/args.ts`, `src/main.ts` |
| Agent-core state (in-memory representation per run) | `packages/agent/src/types.ts` (`AgentState`) |

There is no `memory` package, no `store` package, no `recall` module. Session persistence is the only memory primitive.

## Key code excerpts

### Session entry schema — append-only log of typed records

```ts
// packages/coding-agent/src/core/session-manager.ts
export const CURRENT_SESSION_VERSION = 3;

export interface SessionHeader {
  type: "session";
  version?: number;
  id: string;
  timestamp: string;
  cwd: string;
  parentSession?: string;
}

export interface SessionMessageEntry extends SessionEntryBase {
  type: "message";
  message: AgentMessage;
}

export interface CompactionEntry<T = unknown> extends SessionEntryBase {
  type: "compaction";
  summary: string;
  firstKeptEntryId: string;
  tokensBefore: number;
  details?: T;
  fromHook?: boolean;
}

export type SessionEntry =
  | SessionMessageEntry
  | ThinkingLevelChangeEntry
  | ModelChangeEntry
  | CompactionEntry
  | BranchSummaryEntry
  | CustomEntry
  | CustomMessageEntry
  | LabelEntry
  | SessionInfoEntry;
```

**Why relevant:** This is the entire memory model. A session is a flat list of typed entries — every message, every model change, every compaction summary — with stable `id`/`parentId` for tree structure. Persistence is JSONL append on disk; nothing more sophisticated. There is no separate "memory" object decoupled from the transcript.

### Extension memory hook — `CustomEntry` and `CustomMessageEntry`

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

**Why relevant:** This is the closest Pi gets to a "memory write" API. An extension can append a `CustomEntry` with arbitrary `data` to the session log and recover it on reload. This is durable but scoped to one session and **not retrieved or summarized** for the LLM — it's a private extension-state sidecar, not a queryable knowledge base.

`CustomMessageEntry` does inject content back into the LLM context on reload, but again, only within the same session lineage.

### In-process agent state — just a message array, no persistence

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

**Why relevant:** The agent-core package — Pi's framework foundation — knows nothing about persistence. `messages: AgentMessage[]` is in-process state. All persistence is handled at the coding-agent layer via `session-manager.ts`. There is no `Store`, no `Memory`, no `Recall` abstraction in the framework itself.

### `sessionId` is for provider caching, not memory

The agent-core `AgentLoopConfig` accepts a `sessionId` field that gets forwarded to the LLM provider for prompt caching (Anthropic / OpenAI). This is **not** memory — it's caching of the input prefix on the provider side to reduce token cost. Easy to conflate; worth flagging.

### CLI resume — re-hydrates a session from its log

```ts
// packages/coding-agent/src/cli/args.ts (relevant excerpts)
} else if (arg === "--continue" || arg === "-c") {
  parsed.continue = true;
} else if (arg === "--resume" || arg === "-r") {
  parsed.resume = true;
}

// help text:
//   --continue, -c                 Continue previous session
//   --resume, -r                   Select a session to resume
```

**Why relevant:** Resume reads the JSONL session file from disk, parses entries (`parseSessionEntries`), runs migrations if the version changed (`migrateSessionEntries`), rebuilds the LLM context with `buildSessionContext`, and hands the agent the reconstructed transcript. The "memory" the user perceives is whatever was in that file.

## Tradeoffs and limitations

- **Session-scoped, not cross-session.** Each session is an isolated transcript. Nothing flows from session A to session B unless the user explicitly `--resume`s A. There is no "every user has a profile that all sessions can read."
- **No semantic retrieval.** Memory is flat. To find "what did I tell the agent about my deploy preferences last week?" you would scroll the session file or `--resume` it. There is no vector store, no embedding, no `recall(query)` tool.
- **No write-during-conversation memory tool.** The LLM cannot call a `remember(fact)` tool that persists outside the current session. Extensions can do this via `CustomEntry`, but the entry only survives within the same session lineage.
- **No procedural memory** in the LangMem sense (agent rewriting its own system prompt based on what worked). System prompts are static or template-driven, set by the user / agent definition file.
- **Compaction is not memory.** Compaction summaries (see `pi.md` for `context_management`) are *intra-session* — they let one long session keep going. They are not a "memory" surface across sessions.
- **The agent-core framework has no memory concept at all.** Persistence is a coding-agent concern. A non-coding-agent app built on `@earendil-works/pi-agent-core` would need to BYO storage.

## "Not implemented" caveats

What Pi does not have that the 2026 memory_management pattern implies:

- ❌ Vector store / embedding-based retrieval
- ❌ Episodic / semantic / procedural memory scopes
- ❌ Background memory consolidation ("dreaming" in the Anthropic May 2026 sense)
- ❌ Memory-as-tool (`store_fact`, `recall_facts`) exposed to the LLM
- ❌ Per-user profile / cross-session memory
- ❌ Memory aging, decay, or relevance scoring
- ❌ Pluggable backends (Postgres / Redis / Pinecone / Qdrant) — only the on-disk session log

What Pi does provide that is memory-adjacent:

- ✅ Durable per-session transcript with full fidelity
- ✅ Resume by ID across processes / machine restarts
- ✅ Extension hook (`CustomEntry`) for sidecar state
- ✅ Provider-side prompt caching via `sessionId` (orthogonal to memory)
- ✅ Project-scoped working directory (`cwd` on `SessionHeader`) preserved on resume
