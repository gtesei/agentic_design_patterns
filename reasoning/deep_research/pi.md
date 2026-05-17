# Pi — Deep Research

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Not implemented as a first-class pattern.** Pi does not ship a built-in deep-research workflow: no native web search tool, no citation accumulator, no iterative research controller, and no report synthesizer specialized for evidence-backed research.

What Pi does have is **adjacent infrastructure**:

- model registry entries for deep-research-capable models
- a plan-mode prompt that mentions using a `brave-search` skill through `bash`
- extension examples that show chained recon and context handoff

That is enough substrate for an integrator to build deep research on top, but not enough to say Pi meaningfully implements the pattern out of the box.

## Where it lives (or doesn't)

| Concern | Status in Pi |
|---|---|
| Built-in web search tool | ❌ none found |
| Citation tracking / citation store | ❌ none found |
| Multi-round search-refine-synthesize loop | ❌ none found |
| Deep-research model IDs in registry | ✅ `packages/ai/src/models.generated.ts` |
| Prompt-level mention of web research via skill | ✅ `packages/coding-agent/examples/extensions/plan-mode/index.ts` |
| Chained recon/synthesis workflow | ✅ `packages/coding-agent/examples/extensions/subagent/prompts/scout-and-plan.md` |

## Key code excerpts

Source: `packages/ai/src/models.generated.ts:2496-2512,2564-2578`

```ts
"o3-deep-research": {
	id: "o3-deep-research",
	name: "o3-deep-research",
	...
	contextWindow: 200000,
	maxTokens: 100000,
},

"o4-mini-deep-research": {
	id: "o4-mini-deep-research",
	name: "o4-mini-deep-research",
	...
	contextWindow: 200000,
```

Why this matters: Pi can target research-oriented models at the provider/model layer. That is model availability, not workflow orchestration.

Source: `packages/coding-agent/examples/extensions/plan-mode/index.ts:168-175`

```ts
Restrictions:
- You can only use: read, bash, grep, find, ls, questionnaire
- You CANNOT use: edit, write (file modifications are disabled)
- Bash is restricted to an allowlist of read-only commands

Ask clarifying questions using the questionnaire tool.
Use brave-search skill via bash for web research.

Create a detailed numbered plan under a "Plan:" header:
```

Why this matters: this is the clearest in-repo hint that Pi users may compose web research through a skill, but the research skill itself is not part of core Pi.

Source: `packages/coding-agent/examples/extensions/subagent/prompts/scout-and-plan.md:4-9`

```md
Use the subagent tool with the chain parameter to execute this workflow:

1. First, use the "scout" agent to find all code relevant to: $@
2. Then, use the "planner" agent to create an implementation plan for "$@" using the context from the previous step (use {previous} placeholder)
```

Why this matters: Pi does show the architectural shape of "gather evidence, then synthesize," but the example is codebase reconnaissance, not full deep research.

Source: `packages/coding-agent/examples/extensions/handoff.ts:20-27`

```ts
const SYSTEM_PROMPT = `You are a context transfer assistant. Given a conversation history and the user's goal for a new thread, generate a focused prompt that:

1. Summarizes relevant context from the conversation (decisions made, approaches taken, key findings)
2. Lists any relevant files that were discussed or modified
3. Clearly states the next task based on the user's goal
4. Is self-contained - the new thread should be able to proceed without the old conversation
```

Why this matters: handoff is adjacent to long-horizon research workflows, because it helps package findings for the next iteration, but it is not a research loop by itself.

## Tradeoffs and limitations

- Pi is intentionally extensible, so higher-order workflows like deep research are expected to live in extensions, skills, or external packages.
- That keeps the core small, but it also means there is no standard research stack every Pi installation shares.
- In practice, Pi is currently much better described as "deep-research-capable substrate" than "deep research implementation."

## Final word

Pi does **not** implement deep research as a first-class pattern. It exposes enough primitives to build one, but the repo itself stops at model support, skill hooks, and adjacent chain/handoff examples.
