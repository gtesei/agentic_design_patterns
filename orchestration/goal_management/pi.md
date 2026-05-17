# Pi — Goal Management

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Partially implemented, mostly via extensions.** Pi's core coding agent explicitly does **not** ship built-in plan mode or built-in to-dos. At the same time, the repo includes example extensions that add both:

- a `todo` tool for persistent task state
- a `plan-mode` workflow for read-only exploration, plan extraction, and tracked execution

So Pi supports goal management, but it does so as opt-in orchestration rather than universal core behavior.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Core stance: no built-in plan mode / to-dos | ✅ documented in `packages/coding-agent/README.md` |
| Persisted task list tool | ✅ `packages/coding-agent/examples/extensions/todo.ts` |
| Plan/execution workflow | ✅ `packages/coding-agent/examples/extensions/plan-mode/index.ts` |
| Plan extraction from assistant output | ✅ `packages/coding-agent/examples/extensions/plan-mode/utils.ts` |

## Key code excerpts

Source: `packages/coding-agent/README.md:470-480`

```md
Pi is aggressively extensible so it doesn't have to dictate your workflow.
...
**No plan mode.** Write plans to files, or build it with [extensions](#extensions), or install a package.

**No built-in to-dos.** They confuse models. Use a TODO.md file, or build your own with [extensions](#extensions).
```

Why this matters: Pi is explicit that goal management is not part of the minimal core contract.

Source: `packages/coding-agent/examples/extensions/todo.ts:110-139`

```ts
/**
 * Reconstruct state from session entries.
 * Scans tool results for this tool and applies them in order.
 */
const reconstructState = (ctx: ExtensionContext) => {
	todos = [];
	nextId = 1;
	...
	if (msg.role !== "toolResult" || msg.toolName !== "todo") continue;
	...
};

pi.registerTool({
	name: "todo",
	label: "Todo",
	description: "Manage a todo list. Actions: list, add (text), toggle (id), clear",
	parameters: TodoParams,
```

Why this matters: the `todo` extension turns task state into something durable and tool-accessible, reconstructed from session history.

Source: `packages/coding-agent/examples/extensions/plan-mode/index.ts:38-97`

```ts
let planModeEnabled = false;
let executionMode = false;
let todoItems: TodoItem[] = [];
...
function togglePlanMode(ctx: ExtensionContext): void {
	planModeEnabled = !planModeEnabled;
	executionMode = false;
	todoItems = [];

	if (planModeEnabled) {
		pi.setActiveTools(PLAN_MODE_TOOLS);
		...
	} else {
		pi.setActiveTools(NORMAL_MODE_TOOLS);
		...
	}
}

function persistState(): void {
	pi.appendEntry("plan-mode", {
		enabled: planModeEnabled,
		todos: todoItems,
		executing: executionMode,
	});
}
```

Why this matters: Pi's plan-mode example makes goal state explicit, persists it, and changes available tools depending on planning vs execution mode.

Source: `packages/coding-agent/examples/extensions/plan-mode/index.ts:158-205`

```ts
pi.on("before_agent_start", async () => {
	if (planModeEnabled) {
		return {
			message: {
				customType: "plan-mode-context",
				content: `[PLAN MODE ACTIVE]
You are in plan mode - a read-only exploration mode for safe code analysis.
...
Create a detailed numbered plan under a "Plan:" header:
...
Do NOT attempt to make changes - just describe what you would do.`,
```

Why this matters: goal management is implemented not just as storage, but as turn-level behavior shaping. Planning and execution are treated as different phases.

Source: `packages/coding-agent/examples/extensions/plan-mode/index.ts:240-281` and `packages/coding-agent/examples/extensions/plan-mode/utils.ts:129-149`

```ts
const extracted = extractTodoItems(getTextContent(lastAssistant));
if (extracted.length > 0) {
	todoItems = extracted;
}
...
const choice = await ctx.ui.select("Plan mode - what next?", [
	todoItems.length > 0 ? "Execute the plan (track progress)" : "Execute the plan",
	"Stay in plan mode",
	"Refine the plan",
]);
```

```ts
export function extractTodoItems(message: string): TodoItem[] {
	const items: TodoItem[] = [];
	const headerMatch = message.match(/\*{0,2}Plan:\*{0,2}\s*\n/i);
	if (!headerMatch) return items;
	...
}
```

Why this matters: Pi's example goal-management flow can parse a plan from model output, turn it into tracked steps, and transition into execution.

## Tradeoffs and limitations

- Goal management is real in Pi, but not standardized across every installation.
- The extension approach is flexible: teams can decide whether they want todo tools, planning mode, or plain markdown files.
- The downside is portability: "goal management in Pi" depends heavily on which extensions are installed.

## Final word

Pi partially implements goal management, and it does so most clearly through the `todo` and `plan-mode` extensions rather than through the core agent loop.
