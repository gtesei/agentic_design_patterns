# Pi — Tree of Thoughts

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Not meaningfully implemented as a reasoning pattern.** Tree of Thoughts (ToT) is a reasoning strategy: generate multiple candidate thoughts, evaluate them, expand promising branches, and sometimes backtrack. Pi does not ship that algorithm in its core packages.

What Pi **does** have is a tree-shaped **session** model:

- sessions are append-only trees with `id` and `parentId`
- users can branch and revisit old points with `/tree`
- Pi can summarize the branch being left behind

That is useful infrastructure for conversation management, but it is not the same thing as model-side ToT search.

## Where it lives (or doesn't)

| Concern | Status in Pi |
|---|---|
| Generate multiple candidate thoughts | ❌ none found |
| Score / rank candidate thoughts | ❌ none found |
| Prune or backtrack over reasoning branches | ❌ none found |
| Tree-shaped conversation/session storage | ✅ `packages/coding-agent/src/core/session-manager.ts` |
| Preserve abandoned branch context via summaries | ✅ `packages/coding-agent/src/core/compaction/branch-summarization.ts` |

## Key code excerpts

Source: `packages/coding-agent/src/core/session-manager.ts:696-704`

```ts
 * Manages conversation sessions as append-only trees stored in JSONL files.
 *
 * Each session entry has an id and parentId forming a tree structure. The "leaf"
 * pointer tracks the current position. Appending creates a child of the current leaf.
 * Branching moves the leaf to an earlier entry, allowing new branches without
 * modifying history.
 *
 * Use buildSessionContext() to get the resolved message list for the LLM, which
 * handles compaction summaries and follows the path from root to current leaf.
```

Why this matters: this is the strongest tree-shaped primitive in Pi, but it is a **session-history tree**, not a reasoning tree.

Source: `packages/coding-agent/src/core/session-manager.ts:309-318`

```ts
/**
 * Build the session context from entries using tree traversal.
 * If leafId is provided, walks from that entry to root.
 * Handles compaction and branch summaries along the path.
 */
export function buildSessionContext(
	entries: SessionEntry[],
	leafId?: string | null,
	byId?: Map<string, SessionEntry>,
): SessionContext {
```

Why this matters: Pi resolves **one active path** for the next model call. It does not run multiple branches, compare them, and keep the best.

Source: `packages/coding-agent/src/core/compaction/branch-summarization.ts:1-6`

```ts
/**
 * Branch summarization for tree navigation.
 *
 * When navigating to a different point in the session tree, this generates
 * a summary of the branch being left so context isn't lost.
 */
```

Why this matters: Pi preserves information when a human switches branches, which is adjacent to branch-aware workflows, but still not ToT-style search and evaluation.

## Tradeoffs and limitations

- Pi's session tree is practical for human-controlled branching, recovery, and history preservation.
- It does not implement automatic branch expansion, branch scoring, or branch pruning.
- If someone wanted real Tree of Thoughts in Pi, they would need to build it externally on top of session forking, repeated model calls, and a separate scoring policy.

## Final word

The honest characterization is:

Pi has a **tree-shaped memory/navigation substrate**, but it does **not** implement Tree of Thoughts as a reasoning algorithm.
