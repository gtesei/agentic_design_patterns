# Pi implementation notes: memory management

Accessed on: 2026-05-17

Verdict: Pi meaningfully implements session memory, but mostly as transcript persistence plus compression checkpoints. I did not find a separate long-term semantic memory layer such as embeddings, vector retrieval, or a user-profile store in the inspected `packages/agent` and `packages/coding-agent` sources.

## Relevant Pi code

- `packages/agent/src/harness/session/jsonl-repo.ts`
- `packages/agent/src/harness/session/jsonl-storage.ts`
- `packages/agent/src/harness/session/memory-repo.ts`
- `packages/agent/src/harness/types.ts`
- `packages/coding-agent/src/core/session-manager.ts`
- `packages/coding-agent/src/core/compaction/compaction.ts`

## What Pi actually stores as memory

Pi's strongest memory primitive is a persisted session tree. In the lower-level harness, sessions are stored as JSONL files under a per-working-directory session root:

Source: `packages/agent/src/harness/session/jsonl-repo.ts`

```ts
function encodeCwd(cwd: string): string {
	return `--${cwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
}

async create(options: JsonlSessionCreateOptions): Promise<Session<JsonlSessionMetadata>> {
	const id = options.id ?? createSessionId();
	const createdAt = createTimestamp();
	const sessionDir = await this.getSessionDir(options.cwd);
	...
	const filePath = await this.createSessionFilePath(options.cwd, id, createdAt);
	const storage = await JsonlSessionStorage.create(this.fs, filePath, {
		cwd: options.cwd,
		sessionId: id,
		parentSessionPath: options.parentSessionPath,
	});
	return toSession(storage);
}
```

Why this matters: Pi's "memory" is not an abstract memory API first. It is concrete session state keyed to a working directory, with explicit create/open/list/delete/fork operations.

The JSONL storage is append-oriented. Pi writes a session header once, then appends tree entries as work continues:

Source: `packages/agent/src/harness/session/jsonl-storage.ts`

```ts
const header: SessionHeader = {
	type: "session",
	version: 3,
	id: options.sessionId,
	timestamp: new Date().toISOString(),
	cwd: options.cwd,
	parentSession: options.parentSessionPath,
};
await fs.writeFile(filePath, `${JSON.stringify(header)}\n`);
...
async appendEntry(entry: SessionTreeEntry): Promise<void> {
	await this.fs.appendFile(this.filePath, `${JSON.stringify(entry)}\n`);
	this.entries.push(entry);
	this.byId.set(entry.id, entry);
	this.currentLeafId = leafIdAfterEntry(entry);
}
```

Why this matters: Pi treats memory as an auditable log of entries, not as mutable hidden state. That design makes replay and branching straightforward.

## Memory is branch-aware, not just linear chat history

Pi persists a leaf pointer and supports forking:

Source: `packages/agent/src/harness/session/jsonl-storage.ts`

```ts
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
	...
	this.currentLeafId = leafId;
}
```

Source: `packages/agent/src/harness/session/jsonl-repo.ts`

```ts
async fork(
	sourceMetadata: JsonlSessionMetadata,
	options: JsonlSessionCreateOptions & { entryId?: string; position?: "before" | "at"; id?: string },
): Promise<Session<JsonlSessionMetadata>> {
	const source = await this.open(sourceMetadata);
	const forkedEntries = await getEntriesToFork(source.getStorage(), options);
	...
	for (const entry of forkedEntries) {
		await storage.appendEntry(entry);
	}
	return toSession(storage);
}
```

Why this matters: Pi remembers alternate conversational branches. That is a real memory-management capability, because prior work is kept structurally instead of overwritten.

## Compaction summaries act like compressed memory checkpoints

When the coding agent compacts a session, it writes a durable summary entry rather than dropping history outright:

Source: `packages/coding-agent/src/core/session-manager.ts`

```ts
export interface CompactionEntry<T = unknown> extends SessionEntryBase {
	type: "compaction";
	summary: string;
	firstKeptEntryId: string;
	tokensBefore: number;
	details?: T;
	fromHook?: boolean;
}
```

Source: `packages/coding-agent/src/core/agent-session.ts`

```ts
this.sessionManager.appendCompaction(summary, firstKeptEntryId, tokensBefore, details, fromExtension);
const sessionContext = this.sessionManager.buildSessionContext();
this.agent.state.messages = sessionContext.messages;
```

Why this matters: Pi's memory model includes lossy checkpoints. Old dialogue becomes a structured summary that is persisted and then replayed into future context.

That replay happens explicitly when session context is rebuilt:

Source: `packages/coding-agent/src/core/session-manager.ts`

```ts
if (compaction) {
	messages.push(createCompactionSummaryMessage(compaction.summary, compaction.tokensBefore, compaction.timestamp));
	...
	for (let i = compactionIdx + 1; i < path.length; i++) {
		const entry = path[i];
		appendMessage(entry);
	}
} else {
	for (const entry of path) {
		appendMessage(entry);
	}
}
```

Why this matters: compacted summaries are not metadata only. They become part of the active reconstructed memory for the current branch.

## Pi supports ephemeral memory too

Pi exposes an in-memory session mode for SDK and ephemeral runs:

Source: `packages/agent/src/harness/session/memory-repo.ts`

```ts
export class InMemorySessionRepo implements SessionRepo<SessionMetadata, { id?: string }, void> {
	private sessions = new Map<string, Session<SessionMetadata>>();
	...
}
```

Source: `packages/coding-agent/src/core/session-manager.ts`

```ts
static inMemory(cwd: string = process.cwd()): SessionManager {
	return new SessionManager(cwd, "", undefined, false);
}
```

Why this matters: memory persistence is configurable. Pi can operate with durable session memory or with test-friendly/ephemeral in-memory state.

## Architectural tradeoffs and limitations

- Pi's memory is strongest at transcript durability, branching, and replay. It is weaker at semantic recall.
- I did not find a dedicated vector store, embedding pipeline, retrieval-by-similarity layer, or separate long-term user memory subsystem in the inspected packages.
- Session storage is scoped to working directories and session files. Cross-project recall is not automatic.
- Compaction summaries are LLM-generated and therefore lossy. They improve survivability of long sessions, but they are not exact state snapshots.
- Because memory is append-log based, Pi is auditable and easy to fork, but not optimized for high-level "remember this fact forever" behaviors out of the box.
