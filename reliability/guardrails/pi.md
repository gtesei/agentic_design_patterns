# Pi — Guardrails

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Meaningfully implemented, mostly as a policy layer built on extensions.** Pi does not centralize guardrails into one always-on safety subsystem. Instead, it exposes strong interception hooks and ships example extensions that block, confirm, or sandbox risky actions.

That makes guardrails a real pattern in Pi, but one that is intentionally deployment-specific.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Pre-tool blocking hook | ✅ `packages/coding-agent/src/core/extensions/types.ts` |
| UI confirmation primitives | ✅ `packages/coding-agent/src/core/extensions/types.ts` |
| Command approval example | ✅ `packages/coding-agent/examples/extensions/permission-gate.ts` |
| Protected-path write blocking | ✅ `packages/coding-agent/examples/extensions/protected-paths.ts` |
| OS-level sandboxed tool wrapper | ✅ `packages/coding-agent/examples/extensions/sandbox/index.ts` |

## Key code excerpts

Source: `packages/coding-agent/src/core/extensions/types.ts:124-129`

```ts
export interface ExtensionUIContext {
	select(title: string, options: string[], opts?: ExtensionUIDialogOptions): Promise<string | undefined>;
	confirm(title: string, message: string, opts?: ExtensionUIDialogOptions): Promise<boolean>;
```

Why this matters: Pi guardrails are not limited to hard denials. Extensions can ask for explicit human approval.

Source: `packages/coding-agent/src/core/extensions/types.ts:816-820,984-988`

```ts
/**
 * Fired before a tool executes. Can block.
 *
 * `event.input` is mutable. Mutate it in place to patch tool arguments before execution.
 */

export interface ToolCallEventResult {
	block?: boolean;
	reason?: string;
}
```

Why this matters: this is the main guardrail enforcement point. Extensions can inspect a tool call and stop it before execution.

Source: `packages/coding-agent/examples/extensions/permission-gate.ts:11-29`

```ts
const dangerousPatterns = [/\brm\s+(-rf?|--recursive)/i, /\bsudo\b/i, /\b(chmod|chown)\b.*777/i];

pi.on("tool_call", async (event, ctx) => {
	if (event.toolName !== "bash") return undefined;
	...
	if (isDangerous) {
		if (!ctx.hasUI) {
			return { block: true, reason: "Dangerous command blocked (no UI for confirmation)" };
		}
		...
		if (choice !== "Yes") {
			return { block: true, reason: "Blocked by user" };
		}
	}
});
```

Why this matters: Pi ships a concrete example of command-level approval gating, not just abstract hook types.

Source: `packages/coding-agent/examples/extensions/protected-paths.ts:11-26`

```ts
const protectedPaths = [".env", ".git/", "node_modules/"];

pi.on("tool_call", async (event, ctx) => {
	if (event.toolName !== "write" && event.toolName !== "edit") {
		return undefined;
	}
	...
	if (isProtected) {
		...
		return { block: true, reason: `Path "${path}" is protected` };
	}
});
```

Why this matters: Pi guardrails can be path-aware and tool-specific, which is useful for repo safety and secret protection.

Source: `packages/coding-agent/examples/extensions/sandbox/index.ts:214-279`

```ts
pi.registerTool({
	...localBash,
	label: "bash (sandboxed)",
	async execute(id, params, signal, onUpdate, _ctx) {
		if (!sandboxEnabled || !sandboxInitialized) {
			return localBash.execute(id, params, signal, onUpdate);
		}
		...
		return sandboxedBash.execute(id, params, signal, onUpdate);
	},
});

pi.on("session_start", async (_event, ctx) => {
	...
	await SandboxManager.initialize({
		network: config.network,
		filesystem: config.filesystem,
		...
	});
	...
});
```

Why this matters: Pi can move beyond soft policy checks and actually replace tool execution with sandboxed operations.

## Tradeoffs and limitations

- Guardrails are powerful because they are composable and local to the environment where Pi is deployed.
- They are not universal: one Pi installation may have strong guardrails, another may have none.
- The extension-first model keeps core Pi small, but it means guardrail strength depends on what the operator chooses to enable.

## Final word

Pi meaningfully implements guardrails, but as an **extension-driven policy surface** rather than a single mandatory safety subsystem.
