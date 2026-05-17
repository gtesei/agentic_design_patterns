# Pi — Planning (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Partial — implemented as an opt-in extension, not as a core abstraction.** Pi has no planner node, no `Plan-and-Act` graph, no hierarchical decomposition primitive in the framework. The framework's `AgentLoopConfig` exposes `prepareNextTurn` and `getSteeringMessages` hooks that *could* be used to implement a planner-as-policy, but no built-in implementation uses them this way.

Two reference implementations ship in the repo:

1. **`plan-mode/` extension** (340 lines + 168 lines of utils): a "read-only exploration + numbered plan extraction + step-tracking + `[DONE:n]` completion markers" workflow. Toggleable via `/plan` slash command or `Ctrl+Alt+P`. Restricts the agent to read-only tools during planning; on approval, switches to execution mode with mutating tools. The extension implements planning **as a constrained mode**, not just a prompt — it injects planning instructions via `before_agent_start` and **enforces tool restrictions** via `tool_call` hook with `block: true`.
2. **`subagent/` extension's `chain` mode** + workflow templates: `scout-and-plan.md` chains scout → planner (recon then plan); `implement.md` chains scout → planner → worker (recon, plan, execute). There's a dedicated `planner.md` agent definition with a system prompt that mandates read-only analysis and structured plan output.

Pi explicitly lists `plan mode` among features it "intentionally does not include" in core:

> It intentionally does not include built-in MCP, sub-agents, permission popups, **plan mode**, to-dos, or background bash. You can build or install those workflows as extensions or packages.
> *(`packages/coding-agent/docs/usage.md:275`)*

The maintainer's stance: **planning is a workflow, not a primitive — kept out of core, shipped as a reference extension.**

## Where it lives

| Concern | File:line |
|---|---|
| Explicit "not in core" stance | `packages/coding-agent/docs/usage.md:275`, `packages/coding-agent/README.md:478` |
| Plan-mode extension entry | `packages/coding-agent/examples/extensions/plan-mode/index.ts` (340 lines) |
| Plan-mode read-only/normal tool sets | `packages/coding-agent/examples/extensions/plan-mode/index.ts:23-24` |
| Plan-mode tool-call blocker (enforces read-only) | `packages/coding-agent/examples/extensions/plan-mode/index.ts:121-132` |
| Plan-mode planning-prompt injection via `before_agent_start` | `packages/coding-agent/examples/extensions/plan-mode/index.ts:158-205` |
| Plan-mode utilities (step extraction, completion parsing) | `packages/coding-agent/examples/extensions/plan-mode/utils.ts:129-168` |
| Planner subagent role definition | `packages/coding-agent/examples/extensions/subagent/agents/planner.md:8-37` |
| `scout-and-plan` workflow template | `packages/coding-agent/examples/extensions/subagent/prompts/scout-and-plan.md:4-9` |
| `implement` workflow template (scout → planner → worker) | `packages/coding-agent/examples/extensions/subagent/prompts/implement.md` |
| Framework hooks that could host a planner | `packages/agent/src/types.ts:213-232` (`prepareNextTurn`, `getSteeringMessages`) |

## Key code excerpts

### Tool-set restriction — plan mode is "read-only mode with a plan-extraction overlay"

```ts
// packages/coding-agent/examples/extensions/plan-mode/index.ts:23-24
const PLAN_MODE_TOOLS = ["read", "bash", "grep", "find", "ls", "questionnaire"];
const NORMAL_MODE_TOOLS = ["read", "bash", "edit", "write"];
```

**Why relevant:** The first thing the extension does when planning is enabled: shrink the tool surface. The agent literally **cannot** call `write` or `edit` while in plan mode. The "exploration / planning" phase is intrinsically safe — worst case is read files and run an allowlisted bash command. Capability-restricted reasoning, not just prompted planning.

### Bash allowlist enforcement during plan mode

```ts
// packages/coding-agent/examples/extensions/plan-mode/index.ts:121-132
pi.on("tool_call", async (event) => {
    if (!planModeEnabled || event.toolName !== "bash") return;

    const command = event.input.command as string;
    if (!isSafeCommand(command)) {
        return {
            block: true,
            reason: `Plan mode: command blocked (not allowlisted). Use /plan to disable plan mode first.\nCommand: ${command}`,
        };
    }
});
```

**Why relevant:** Plan mode uses the **same `tool_call` extension event** that drives HITL gates (`permission-gate.ts`, `protected-paths.ts`) — see `10_hitl/pi.md`. The plan-first architecture is enforced by *blocking destructive bash commands*, not just by polite prompting. The error message tells the user (and the model) exactly how to escape if they need full bash access.

### Plan-mode prompt injection — explicit planning instructions

```ts
// packages/coding-agent/examples/extensions/plan-mode/index.ts:158-205 (excerpted)
pi.on("before_agent_start", async () => {
    if (planModeEnabled) {
        return {
            message: {
                customType: "plan-mode-context",
                content: `[PLAN MODE ACTIVE]
You are in plan mode - a read-only exploration mode for safe code analysis.

Restrictions:
- You can only use: read, bash, grep, find, ls, questionnaire
- You CANNOT use: edit, write (file modifications are disabled)
- Bash is restricted to an allowlist of read-only commands

Ask clarifying questions using the questionnaire tool.
Use brave-search skill via bash for web research.

Create a detailed numbered plan under a "Plan:" header:

Plan:
1. First step description
2. Second step description
...

Do NOT attempt to make changes - just describe what you would do.`,
                display: false,
            },
        };
    }
```

**Why relevant:** Plan creation is implemented as a concrete *mode* with prompt injection and tool restrictions, not just user convention. The instructions tell the model exactly what format to use (`Plan:` header + numbered steps), which is what the parser downstream depends on.

### Plan extraction + completion tracking — regex over assistant text

```ts
// packages/coding-agent/examples/extensions/plan-mode/utils.ts:129-168 (excerpted)
export function extractTodoItems(message: string): TodoItem[] {
    const items: TodoItem[] = [];
    const headerMatch = message.match(/\*{0,2}Plan:\*{0,2}\s*\n/i);
    if (!headerMatch) return items;

    const planSection = message.slice(message.indexOf(headerMatch[0]) + headerMatch[0].length);
    const numberedPattern = /^\s*(\d+)[.)]\s+\*{0,2}([^*\n]+)/gm;
    // ...
}

export function extractDoneSteps(message: string): number[] {
    const steps: number[] = [];
    for (const match of message.matchAll(/\[DONE:(\d+)\]/gi)) {
        const step = Number(match[1]);
        if (Number.isFinite(step)) steps.push(step);
    }
    return steps;
}

export function markCompletedSteps(text: string, items: TodoItem[]): number {
    const doneSteps = extractDoneSteps(text);
    for (const step of doneSteps) {
        const item = items.find((t) => t.step === step);
        if (item) item.completed = true;
    }
    return doneSteps.length;
}
```

**Why relevant:** Planning here is **emergent from prompting plus regex**, not structured output. The agent is told to emit `Plan:` + numbered steps and `[DONE:n]` completion markers (via the injected prompt above); the utilities regex-parse the assistant message text and update a UI widget. Trades structural rigor for prompt-engineering flexibility. The fragility lives in the formatting convention.

### Planner subagent role definition

```md
<!-- packages/coding-agent/examples/extensions/subagent/agents/planner.md:8-37 (excerpted) -->
You are a planning specialist. You receive context (from a scout) and requirements, then produce a clear implementation plan.

You must NOT make any changes. Only read, analyze, and plan.

Output format:

## Goal
One sentence summary of what needs to be done.

## Plan
Numbered steps, each small and actionable:
1. Step one - specific file/function to modify
2. Step two - what to add/change
...
```

**Why relevant:** Pi has a **dedicated planning role** as a markdown agent definition. The system prompt mandates read-only analysis and a structured output format. This is the second planning shape Pi ships: not in-process mode-switching (like `plan-mode`), but multi-process role specialization (via `subagent` chain mode).

### Scout-and-plan workflow template

```md
<!-- packages/coding-agent/examples/extensions/subagent/prompts/scout-and-plan.md:4-9 -->
Use the subagent tool with the chain parameter to execute this workflow:

1. First, use the "scout" agent to find all code relevant to: $@
2. Then, use the "planner" agent to create an implementation plan for "$@" using the context from the previous step (use {previous} placeholder)

Execute this as a chain, passing output between steps via {previous}. Do NOT implement - just return the plan.
```

**Why relevant:** The workflow is a markdown prompt loaded into the orchestrator's context. The orchestrator LLM reads it and invokes the `subagent` tool in chain mode with `scout` → `planner`. Reconnaissance feeds planning; planning produces a plan; nothing executes. Pi expresses workflow recipes as **prompts**, not as code.

## Tradeoffs and limitations

- **All planning lives in extensions.** Stock `pi` has no planning capability until you install `plan-mode/` or the `subagent/` extension.
- **Plan format is brittle.** `plan-mode/` depends on the agent emitting `Plan:` headers and `[DONE:n]` markers in *exact* shapes the regex can parse. A model that decides to format slightly differently breaks step tracking.
- **No typed plan schema.** Plans are markdown the extension parses. Structured-output (`structured-output.ts` pattern, see `11_structured_outputs/pi.md`) could be used to produce typed plans but the existing plan-mode extension does not.
- **No automatic replan on failure.** Plan-mode tracks completion but doesn't react to failures — the user or orchestrator LLM decides what to do.
- **Plan-mode and subagent-chain planning are non-composable.** They are two different patterns; nothing in core unifies them.
- **No hierarchical decomposition framework.** Plans are flat lists of steps; no plan-of-plans, no sub-step trees.
- **In-process plan-mode mutates state across turns;** subagent-chain planning is fresh-process per step. Different cost / consistency tradeoffs but Pi doesn't help you reason about which to pick.

## "Not implemented" caveats

- ❌ A `Plan`, `Planner`, `PlanAndAct`, or `HiPlan` class in core
- ❌ Graph-based plan representation (no DAG of steps with dependencies)
- ❌ Replan-on-failure primitive at the framework level
- ❌ Hierarchical decomposition
- ❌ Structured plan schema (TypeBox-typed `Plan[]`) — plans are markdown
- ❌ Tool-typed step completion — completion is regex over assistant text
- ❌ Multi-plan composition

What Pi does ship:

- ✅ `plan-mode/` reference extension: tool-set restriction + plan extraction + `[DONE:n]` completion tracking + UI widget
- ✅ Bash allowlist enforcement during plan mode (real `block: true` policy, not just prompting)
- ✅ Prompt injection via `before_agent_start` carrying the planning system instructions
- ✅ Dedicated `planner.md` subagent role definition
- ✅ `scout-and-plan.md` and `implement.md` workflow templates (subagent chain)
- ✅ Framework hooks (`prepareNextTurn`, `getSteeringMessages`) that an integrator could use to build a custom planner-as-policy
- ✅ Slash-command + keybinding + flag registration APIs for plan-mode-style toggles
- ✅ Status / widget UI APIs for surfacing plan progress
- ✅ Session-persistent extension state (so plan-mode survives `--resume`)
