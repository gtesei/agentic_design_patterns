# Pi — Exploration & Discovery

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Partially implemented, mostly as workflow guidance plus extensions.** Pi does not have a dedicated "exploration engine," but it does meaningfully support exploratory work in three ways:

1. the core system prompt nudges the agent toward efficient codebase exploration
2. Pi exposes a read-only exploration tool set
3. the `subagent` example ships an explicit `scout` role and a `scout -> planner` workflow

So the pattern is real in Pi, but it is distributed across prompting, tool design, and example extensions rather than packaged as one subsystem.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Exploration guidance in the default prompt | ✅ `packages/coding-agent/src/core/system-prompt.ts` |
| Read-only exploration tool bundle | ✅ `packages/coding-agent/src/core/tools/index.ts` |
| Specialized reconnaissance agent (`scout`) | ✅ `packages/coding-agent/examples/extensions/subagent/agents/scout.md` |
| Discovery handoff into later planning | ✅ `packages/coding-agent/examples/extensions/subagent/prompts/scout-and-plan.md` |

## Key code excerpts

Source: `packages/coding-agent/src/core/system-prompt.ts:111-116`

```ts
// File exploration guidelines
if (hasBash && !hasGrep && !hasFind && !hasLs) {
	addGuideline("Use bash for file operations like ls, rg, find");
} else if (hasBash && (hasGrep || hasFind || hasLs)) {
	addGuideline("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)");
}
```

Why this matters: Pi explicitly teaches the agent how to explore a codebase efficiently, rather than leaving exploration behavior fully implicit.

Source: `packages/coding-agent/src/core/tools/index.ts:147-153`

```ts
export function createReadOnlyToolDefinitions(cwd: string, options?: ToolsOptions): ToolDef[] {
	return [
		createReadToolDefinition(cwd, options?.read),
		createGrepToolDefinition(cwd, options?.grep),
		createFindToolDefinition(cwd, options?.find),
		createLsToolDefinition(cwd, options?.ls),
	];
}
```

Why this matters: Pi has a concrete, first-class read-only tool bundle for discovery work. That makes safe exploration a recognizable operating mode.

Source: `packages/coding-agent/examples/extensions/subagent/agents/scout.md:2-21`

```md
name: scout
description: Fast codebase recon that returns compressed context for handoff to other agents
...
You are a scout. Quickly investigate a codebase and return structured findings that another agent can use without re-reading everything.
...
Strategy:
1. grep/find to locate relevant code
2. Read key sections (not entire files)
3. Identify types, interfaces, key functions
4. Note dependencies between files
```

Why this matters: Pi's example assets make exploration explicit as a specialized agent role with a clear reconnaissance contract.

Source: `packages/coding-agent/examples/extensions/subagent/prompts/scout-and-plan.md:4-9`

```md
Use the subagent tool with the chain parameter to execute this workflow:

1. First, use the "scout" agent to find all code relevant to: $@
2. Then, use the "planner" agent to create an implementation plan for "$@" using the context from the previous step (use {previous} placeholder)

Execute this as a chain, passing output between steps via {previous}. Do NOT implement - just return the plan.
```

Why this matters: this is the clearest in-repo example of exploration feeding into downstream reasoning.

## Tradeoffs and limitations

- Exploration is supported, but mostly through conventions and extensions rather than one core controller.
- Discovery results are handed off as text, not as a typed evidence graph or durable retrieval index.
- The approach is flexible and easy to extend, but different Pi setups may expose very different discovery behaviors.

## Final word

Pi meaningfully supports exploration and discovery, especially for codebase reconnaissance, but it implements the pattern as **prompt guidance + tool design + extension workflows**, not as a dedicated core module.
