# Pi — Agent Communication

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Partially implemented, mostly through extension conventions rather than a formal agent-to-agent protocol.** Pi does not ship a typed, first-class A2A transport. What it does provide is:

- subprocess-based subagent delegation
- chained communication via text handoff
- focused session/thread handoff
- an inter-extension event bus

So communication exists, but it is pragmatic and heterogeneous rather than standardized.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Subagent subprocess execution and message capture | ✅ `packages/coding-agent/examples/extensions/subagent/index.ts` |
| Sequential agent-to-agent text passing | ✅ `packages/coding-agent/examples/extensions/subagent/index.ts` |
| Session/thread handoff | ✅ `packages/coding-agent/examples/extensions/handoff.ts` |
| Extension-to-extension event bus | ✅ `packages/coding-agent/examples/extensions/event-bus.ts` |

## Key code excerpts

Source: `packages/coding-agent/examples/extensions/subagent/index.ts:304-338`

```ts
const exitCode = await new Promise<number>((resolve) => {
	const invocation = getPiInvocation(args);
	const proc = spawn(invocation.command, invocation.args, {
		cwd: cwd ?? defaultCwd,
		shell: false,
		stdio: ["ignore", "pipe", "pipe"],
	});
	...
	const processLine = (line: string) => {
		...
		event = JSON.parse(line);
		...
		if (event.type === "message_end" && event.message) {
			const msg = event.message as Message;
			currentResult.messages.push(msg);
			...
		}
	};
});
```

Why this matters: Pi's `subagent` example communicates with child agents through a real subprocess/event stream boundary. That is a concrete communication mechanism, even though it is extension-level.

Source: `packages/coding-agent/examples/extensions/subagent/index.ts:435-438,501-549`

```ts
description: [
	"Delegate tasks to specialized subagents with isolated context.",
	"Modes: single (agent + task), parallel (tasks array), chain (sequential with {previous} placeholder).",
].join(" "),
...
if (params.chain && params.chain.length > 0) {
	const results: SingleResult[] = [];
	let previousOutput = "";
	...
	const taskWithContext = step.task.replace(/\{previous\}/g, previousOutput);
	...
	previousOutput = getFinalOutput(result.messages);
}
```

Why this matters: Pi supports an explicit sequential communication pattern where one agent's output becomes the next agent's input.

Source: `packages/coding-agent/examples/extensions/handoff.ts:20-27,100-111,175-184`

```ts
const SYSTEM_PROMPT = `You are a context transfer assistant. Given a conversation history and the user's goal for a new thread, generate a focused prompt that:
1. Summarizes relevant context from the conversation ...
2. Lists any relevant files ...
3. Clearly states the next task ...
4. Is self-contained ...
`;
...
const messages = getHandoffMessages(ctx.sessionManager.getBranch());
const llmMessages = convertToLlm(messages);
const conversationText = serializeConversation(llmMessages);
...
const newSessionResult = await ctx.newSession({
	parentSession: currentSessionFile,
	withSession: async (replacementCtx) => {
		replacementCtx.ui.setEditorText(editedPrompt);
	},
});
```

Why this matters: Pi can package the current branch into a focused communication artifact for a fresh session, which is a useful form of inter-agent or inter-thread handoff.

Source: `packages/coding-agent/examples/extensions/event-bus.ts:1-8,20-32`

```ts
/**
 * Inter-extension event bus example.
 *
 * Shows pi.events for communication between extensions.
 */
...
pi.events.on("my:notification", (data) => {
	...
});
...
pi.registerCommand("emit", {
	...
	pi.events.emit("my:notification", { message, from: "/emit command" });
});
```

Why this matters: Pi has an explicit communication channel for extensions, even though it is not an agent protocol in the stricter A2A sense.

## Tradeoffs and limitations

- Communication is mostly text-based and convention-based, not a versioned typed protocol.
- The subagent system is an example extension, not a mandatory core capability.
- That keeps Pi simple and inspectable, but leaves semantics like trust, schema, and routing to integrators.

## Final word

Pi does implement useful agent communication patterns, but mainly as **subprocess handoff, text chaining, session transfer, and extension events**, not as a unified agent-to-agent protocol.
