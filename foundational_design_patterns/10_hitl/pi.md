# Pi — Human-in-the-Loop (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Yes — comprehensive, multi-shape implementation.** Pi treats HITL as a first-class concern at both the framework layer and the extension layer. Five cooperating mechanisms:

1. **`beforeToolCall` hook (framework primitive).** Returns `{ block: true, reason: "..." }` to prevent a tool from executing. The framework synthesizes an error tool result.
2. **Session-event hooks (`session_before_switch`, `session_before_fork`, etc.).** Allow extensions to gate destructive session operations on user confirmation.
3. **UI primitives (`ctx.ui.confirm`, `ctx.ui.select`, `ctx.ui.input`, `ctx.ui.editor`, `ctx.ui.notify`, `ctx.ui.custom`)** — typed interaction APIs available to extensions, with **timeout support** built in (`{ timeout: 5000 }` for auto-cancel) and AbortSignal as an alternative.
4. **Mid-run steering (`AgentSession.steer()` and `AgentSession.followUp()`).** The human (or an external controller) types while the agent is running; the message is injected at the next safe point. This is *interruptive collaboration*, not just gate-keeping. Both methods are also exposed over the RPC protocol so external controllers can inject HITL guidance.
5. **`getSteeringMessages` framework hook** that the agent loop polls for queued user input.

The repository ships several reference HITL extensions: `confirm-destructive.ts`, `permission-gate.ts`, `protected-paths.ts`, `timed-confirm.ts`, `dirty-repo-guard.ts`, `question.ts`, `questionnaire.ts`.

Steering ⟂ approval. Pi distinguishes "approval-gated execution" (the framework asks the human before doing X) from "human-injected steering" (the human interrupts to redirect the agent). Both are first-class.

## Where it lives

| Concern | File:line |
|---|---|
| `beforeToolCall` framework hook | `packages/agent/src/types.ts:50-58, 262-268` |
| `BeforeToolCallResult` (`{ block, reason }`) | `packages/agent/src/types.ts:50-58` |
| `getSteeringMessages` framework hook | `packages/agent/src/types.ts:222-232` |
| `AgentSession.steer()` / `followUp()` | `packages/coding-agent/src/core/agent-session.ts:1181-1245` |
| RPC mode `steer` + `follow_up` commands | `packages/coding-agent/src/modes/rpc/rpc-mode.ts:403-410` |
| Session-event hooks (`session_before_switch`, `session_before_fork`) | `packages/coding-agent/src/core/extensions/runner.ts` |
| Extension UI APIs (`select`/`confirm`/`input`/`editor`/`notify`/`custom`) | `packages/coding-agent/src/core/extensions/types.ts:124-135` |
| `confirm-destructive.ts` — session-event confirmation | `packages/coding-agent/examples/extensions/confirm-destructive.ts:11-22` |
| `permission-gate.ts` — dangerous-command detection | `packages/coding-agent/examples/extensions/permission-gate.ts:13-29` |
| `protected-paths.ts` — write/edit path allowlist | `packages/coding-agent/examples/extensions/protected-paths.ts` |
| `timed-confirm.ts` — timeout-aware confirmation | `packages/coding-agent/examples/extensions/timed-confirm.ts` |
| `dirty-repo-guard.ts` — git-state guard | `packages/coding-agent/examples/extensions/dirty-repo-guard.ts` |
| `question.ts` / `questionnaire.ts` — agent-asks-human tools | `packages/coding-agent/examples/extensions/{question,questionnaire}.ts` |
| Maintainer's "no built-in permission popups" stance | `packages/coding-agent/README.md:476` |

## Key code excerpts

### Framework primitive — `beforeToolCall` returns `{ block, reason }`

```ts
// packages/agent/src/types.ts:50-58
/**
 * Returning `{ block: true }` prevents the tool from executing. The loop emits an error tool result instead.
 * `reason` becomes the text shown in that error result. If omitted, a default blocked message is used.
 */
export interface BeforeToolCallResult {
    block?: boolean;
    reason?: string;
}
```

```ts
// packages/agent/src/types.ts:262-268
/**
 * Called before a tool is executed, after arguments have been validated.
 *
 * Return `{ block: true }` to prevent execution. The loop emits an error tool result instead.
 * The hook receives the agent abort signal and is responsible for honoring it.
 */
beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
```

**Why relevant:** The *single* framework primitive that enables every HITL approval pattern. An extension's `beforeToolCall` handler can block synchronously (no UI), prompt the user via `ctx.ui.confirm`, check filesystem state, inspect tool arguments for dangerous patterns — and return `block: true` if anything fails. The blocked result becomes a synthesized error message the model sees, so the model can adapt rather than crash.

### Extension UI primitives — typed interaction APIs

```ts
// packages/coding-agent/src/core/extensions/types.ts:124-135 (excerpted)
export interface ExtensionUIContext {
    /** Show a selector and return the user's choice. */
    select(title: string, options: string[], opts?: ExtensionUIDialogOptions): Promise<string | undefined>;

    /** Show a confirmation dialog. */
    confirm(title: string, message: string, opts?: ExtensionUIDialogOptions): Promise<boolean>;

    /** Show a text input dialog. */
    input(title: string, placeholder?: string, opts?: ExtensionUIDialogOptions): Promise<string | undefined>;

    /** Show a notification to the user. */
    notify(message: string, type?: "info" | "warning" | "error"): void;
    // also: editor, custom
}
```

**Why relevant:** HITL is not just "pause and print text." Pi has concrete typed APIs for confirmation, multi-choice, text input, full-editor input, notifications, and arbitrary custom UI components. All support optional `{ timeout }` for auto-cancel countdowns. All return `undefined` on cancel — explicit failure shape, no exception throwing.

### Permission-gate — dangerous bash command detection

```ts
// packages/coding-agent/examples/extensions/permission-gate.ts:13-29 (full handler)
pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "bash") return undefined;

    const command = event.input.command as string;
    const isDangerous = dangerousPatterns.some((p) => p.test(command));

    if (isDangerous) {
        if (!ctx.hasUI) {
            // In non-interactive mode, block by default
            return { block: true, reason: "Dangerous command blocked (no UI for confirmation)" };
        }

        const choice = await ctx.ui.select(`⚠️ Dangerous command:\n\n  ${command}\n\nAllow?`, ["Yes", "No"]);

        if (choice !== "Yes") {
            return { block: true, reason: "Blocked by user" };
        }
    }

    return undefined;
});
```

**Why relevant:** The canonical HITL extension shape. Pattern-based detection (regex, no LLM call). **UI-aware fallback** — `ctx.hasUI` check: in non-interactive mode (e.g., `--mode json` SDK path used by subagents), there's no human to ask, so block by default. The framework distinguishes these modes and the extension respects it.

### Confirm-destructive — session-event hooks

```ts
// packages/coding-agent/examples/extensions/confirm-destructive.ts:11-22 (excerpted)
pi.on("session_before_switch", async (event: SessionBeforeSwitchEvent, ctx) => {
    if (!ctx.hasUI) return;

    if (event.reason === "new") {
        const confirmed = await ctx.ui.confirm(
            "Clear session?",
            "This will delete all messages in the current session.",
        );

        if (!confirmed) {
            ctx.ui.notify("Clear cancelled", "info");
            return { cancel: true };
        }
        return;
    }
    // ... reason === "resume" - check for unsaved work
});
```

**Why relevant:** HITL isn't only about tool calls. Pi has a parallel event surface for **session-level destructive operations**: switching sessions, forking, clearing. Each emits a `_before_*` event that handlers can cancel by returning `{ cancel: true }`. This is the right shape for "the user is about to wipe their work" — distinct from "the agent is about to do X."

### Mid-run steering — `steer()` and `followUp()` methods

```ts
// packages/coding-agent/src/core/agent-session.ts:1181-1245 (excerpted)
async steer(text: string, images?: ImageContent[]): Promise<void> {
    // ...
    let expandedText = this._expandSkillCommand(text);
    expandedText = expandPromptTemplate(expandedText, [...this.promptTemplates]);

    await this._queueSteer(expandedText, images);
}

async followUp(text: string, images?: ImageContent[]): Promise<void> {
    // ...
    let expandedText = this._expandSkillCommand(text);
    expandedText = expandPromptTemplate(expandedText, [...this.promptTemplates]);

    await this._queueFollowUp(expandedText, images);
}

private async _queueSteer(text: string, images?: ImageContent[]): Promise<void> {
    // ...
    this.agent.steer({ role: "user", content, timestamp: Date.now() });
}
```

**Why relevant:** `steer()` is "inject this user message at the next safe point in the running loop." `followUp()` is "inject this when the agent would otherwise stop." Both go through the same expansion pipeline as regular prompts (`/skill:` and `/template` get expanded — see `1_prompt_chain/pi.md`). HITL via message injection, not via blocking.

### RPC exposure — HITL works for external controllers too

```ts
// packages/coding-agent/src/modes/rpc/rpc-mode.ts:403-410
case "steer": {
    await session.steer(command.message, command.images);
    return success(id, "steer");
}

case "follow_up": {
    await session.followUp(command.message, command.images);
    return success(id, "follow_up");
}
```

**Why relevant:** HITL is not limited to the local TUI. External controllers (web UI, IDE plugin, automation harness) can push human guidance via RPC. Same `steer` / `follow_up` semantics, just over JSON-RPC instead of through the local keyboard.

### Timed confirm — timeout-aware UI primitive

```ts
// packages/coding-agent/examples/extensions/timed-confirm.ts (usage example)
const confirmed = await ctx.ui.confirm(
    "Timed Confirmation",
    "This dialog will auto-cancel in 5 seconds. Confirm?",
    { timeout: 5000 },
);

// Or with AbortSignal for finer control:
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 5000);
```

**Why relevant:** `ctx.ui.confirm` / `ctx.ui.select` support `{ timeout: number }` for live-countdown auto-cancel. Critical for production HITL: a confirmation that hangs forever blocks the agent's loop and consumes context cache; bounded waits are the production-grade pattern. Default-on-timeout is **cancel**, not approve — fail-safe.

### Protected-paths — non-interactive blocking

```ts
// packages/coding-agent/examples/extensions/protected-paths.ts (excerpted)
const protectedPaths = [".env", ".git/", "node_modules/"];

pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "write" && event.toolName !== "edit") return undefined;

    const path = event.input.path as string;
    const isProtected = protectedPaths.some((p) => path.includes(p));

    if (isProtected) {
        if (ctx.hasUI) {
            ctx.ui.notify(`Blocked write to protected path: ${path}`, "warning");
        }
        return { block: true, reason: `Path "${path}" is protected` };
    }
    return undefined;
});
```

**Why relevant:** Different policy shape. Protected paths are **always blocked** — user is just notified, not asked. A useful escalation pattern: combine `protected-paths.ts` (hard block) with `permission-gate.ts` (interactive gate) for layered defense.

### Agent-asks-human tools (`question.ts` / `questionnaire.ts`)

These extensions go the other direction: they register tools the *agent* can call to ask the *human* a question. The agent emits a tool call like `question({ prompt: "Should I delete X?" })`, the tool implementation prompts the user via `ctx.ui.confirm` / `ctx.ui.select`, and the answer comes back as the tool result.

**Why relevant:** HITL bidirectional. Three patterns coexist:

- Framework asks the human → `beforeToolCall` returns `block` (agent doesn't know it was asked).
- Human interrupts the agent → `steer()` / `getSteeringMessages` (human chose to interject).
- Agent asks the human → `question` / `questionnaire` tool (agent chose to ask).

Each is the right shape for a different scenario.

## Tradeoffs and limitations

- **Framework hook returns `block: true`, never "approve later."** A `beforeToolCall` hook that blocks does so synchronously; there's no "create an approval request, agent waits indefinitely, human approves out-of-band later" workflow. For async approval queues, build it on top of `block: true` + retry logic.
- **No queuing of approval requests across multiple users.** HITL in Pi is single-user-at-the-terminal (or single RPC controller). Multi-reviewer flows are out of scope.
- **`ctx.hasUI` is binary.** Either there's a TUI or there isn't. No "headless web UI" interpretation by default — extensions that want web-based approval would need to build that channel themselves.
- **Steering injects at "safe points," not arbitrarily.** The human can't preempt a long-running tool call (the current turn finishes its tool calls first). Aborting mid-tool requires Ctrl+C (fires the `AbortSignal`).
- **No first-class audit log of approve/deny decisions.** Extensions that need audit have to log to disk themselves (e.g., into `CustomEntry` session entries — see `memory/memory_management/pi.md`).
- **All HITL extensions are opt-in.** Stock `pi` has no built-in confirmation prompts (per the maintainer's documented stance: `packages/coding-agent/README.md:476`).
- **`timed-confirm` defaults to cancel on timeout.** Fail-safe but means a forgotten dialog blocks the dangerous action — agent gets blocked, not a "yes by default" surprise. Different operational tradeoff than auto-approve.

## "Not implemented" caveats

- ❌ Built-in approval UI in stock `pi` (per the maintainer's documented stance)
- ❌ Asynchronous approval queues (request-now, approve-out-of-band)
- ❌ Multi-reviewer workflows
- ❌ Web-based HITL channels (would require a custom extension wiring `ctx.ui.custom` to a web UI, or use the RPC mode)
- ❌ Built-in audit log of approve/deny decisions
- ❌ Preemption of in-flight tool calls (Ctrl+C aborts, doesn't "ask first")

What Pi does ship that the HITL pattern asks for:

- ✅ `beforeToolCall` framework primitive with `{ block, reason }` contract
- ✅ Session-level event hooks with cancellation (`session_before_switch`, `session_before_fork`, etc.)
- ✅ Typed UI primitives: `confirm`, `select`, `input`, `editor`, `notify`, `custom`
- ✅ Built-in `{ timeout }` support on confirm / select for bounded waits
- ✅ AbortSignal-based timeout for finer control
- ✅ Mid-run steering — `AgentSession.steer()` and `followUp()` methods
- ✅ RPC exposure of `steer` / `follow_up` for external controllers
- ✅ Framework `getSteeringMessages` hook for human-initiated injection
- ✅ Agent-asks-human tools (`question`, `questionnaire`)
- ✅ Mode-aware UI fallback (`ctx.hasUI`) — block-by-default in non-interactive contexts
- ✅ 7 reference HITL extensions demonstrating the patterns
