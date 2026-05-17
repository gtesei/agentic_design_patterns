# Pi — Graph of Thoughts

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Not meaningfully implemented.** Graph of Thoughts (GoT) needs a graph-shaped reasoning structure: arbitrary edges, recombination, merges, and graph-level traversal or evaluation. Pi does not expose anything like that in its core runtime.

The closest thing Pi has is its **session tree**. That is explicitly a tree, not a general graph.

## Where it lives (or doesn't)

| Concern | Status in Pi |
|---|---|
| General graph-shaped reasoning state | ❌ none found |
| Recombination / merged thought nodes | ❌ none found |
| Arbitrary graph edges between reasoning states | ❌ none found |
| Tree-shaped session structure | ✅ `packages/coding-agent/src/core/session-manager.ts` |
| Tree navigation UI (`/tree`) | ✅ documented in `packages/coding-agent/README.md` |

## Key code excerpts

Source: `packages/coding-agent/src/core/session-manager.ts:136-159`

```ts
/** Session entry - has id/parentId for tree structure (returned by "read" methods in SessionManager) */
export type SessionEntry = ...

/** Tree node for getTree() - defensive copy of session structure */
export interface SessionTreeNode {
	entry: SessionEntry;
	children: SessionTreeNode[];
	label?: string;
	labelTimestamp?: string;
}
```

Why this matters: Pi's foundational branching structure is explicitly parent/child only.

Source: `packages/coding-agent/src/core/session-manager.ts:1108-1145`

```ts
 * Get the session as a tree structure. Returns a shallow defensive copy of all entries.
 * A well-formed session has exactly one root (first entry with parentId === null).
 * Orphaned entries (broken parent chain) are also returned as roots.
 */
getTree(): SessionTreeNode[] {
	...
	if (entry.parentId === null || entry.parentId === entry.id) {
		roots.push(node);
	} else {
		const parent = nodeMap.get(entry.parentId);
		if (parent) {
			parent.children.push(node);
		}
	}
}
```

Why this matters: the implementation literally builds a rooted tree. There is no mechanism for shared children, merge nodes, or non-hierarchical edges.

Source: `packages/coding-agent/README.md:228-248`

```md
Sessions are stored as JSONL files with a tree structure. Each entry has an `id` and `parentId`, enabling in-place branching without creating new files.

**`/tree`** - Navigate the session tree in-place. Select any previous point, continue from there, and switch between branches.
```

Why this matters: the product documentation matches the code. Pi presents branching as a session tree, not as a thought graph.

## Tradeoffs and limitations

- Trees are much simpler to persist, render, and reason about than general graphs.
- That simplicity also rules out GoT-style recombination and graph-level search.
- If Graph of Thoughts were needed, it would require a new orchestration layer above Pi's session model rather than a thin wrapper around the existing tree API.

## Final word

Pi implements **tree-shaped session branching**, not Graph of Thoughts.
