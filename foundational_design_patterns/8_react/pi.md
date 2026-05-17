# Pi — ReAct (Reason + Act) (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Yes — Pi's agent loop *is* a ReAct loop.** The framework's main execution unit (`runLoop` in `packages/agent/src/agent-loop.ts`) implements the Thought → Action → Observation cycle directly: an outer loop polls for steering/follow-up messages, an inner loop alternates between streaming an assistant response (which may contain reasoning + tool calls) and executing those tool calls (with results appended to context for the next assistant turn). The cycle continues until either no more tool calls are emitted, a hook signals termination, or an error/abort fires.

The label "ReAct" is not used as a formal API name in the codebase, but the behavior matches the pattern closely. Pi is built around an agent that thinks and acts in alternation — that *is* ReAct, expressed in TypeScript. Reasoning gets first-class support too: many providers expose a `thinking` content block (separate from `text`), and Pi propagates those through the assistant message structure so reasoning is visible and persistable but distinguishable from final output.

There is **no separate "ReAct" abstraction** — Pi doesn't have a `ReactAgent` class because the agent loop itself enforces the pattern. Every agent in Pi is a ReAct agent by construction.

Pi also extends classic ReAct in two ways:
- **Mid-run steering** — humans (or automated controllers) can inject messages between turns without breaking the loop, via `getSteeringMessages`.
- **Follow-up messages** — queued user messages that fire after the agent would otherwise stop, via `getFollowUpMessages`.

## Where it lives

| Concern | File:line |
|---|---|
| Outer loop (`while (true)` over steering/follow-up cycles) | `packages/agent/src/agent-loop.ts:170` |
| Inner loop (`while (hasMoreToolCalls || pendingMessages.length > 0)`) | `packages/agent/src/agent-loop.ts:174` |
| `streamAssistantResponse` — the "Thought + Action" step | `packages/agent/src/agent-loop.ts:193, 275` |
| Tool execution — the "Observation" step | `packages/agent/src/agent-loop.ts:208-217` |
| Event protocol (`turn_start`, `message_*`, `tool_execution_*`, `turn_end`) | `packages/agent/src/agent-loop.ts:108-118, 176-218` |
| Context build per call (`transformContext` then `convertToLlm`) | `packages/agent/src/agent-loop.ts:282-296` |
| Provider event stream (`thinking_*` vs `text_*` vs `toolcall_*`) | `packages/ai/src/types.ts:347-360` |
| Continuation entry point (resume from saved state) | `packages/agent/src/agent-loop.ts:121-143` (`runAgentLoopContinue`) |
| Steering + follow-up cycle wiring | `packages/agent/src/agent-loop.ts:253-260` |

## Key code excerpts

### Inner loop — the ReAct cycle itself

```ts
// packages/agent/src/agent-loop.ts:170-220 (excerpted)
while (true) {
    let hasMoreToolCalls = true;

    while (hasMoreToolCalls || pendingMessages.length > 0) {
        if (!firstTurn) await emit({ type: "turn_start" });
        else firstTurn = false;

        // Process pending messages (inject before next assistant response)
        if (pendingMessages.length > 0) {
            for (const message of pendingMessages) {
                await emit({ type: "message_start", message });
                await emit({ type: "message_end", message });
                currentContext.messages.push(message);
                newMessages.push(message);
            }
            pendingMessages = [];
        }

        // "Thought" + "Action" — assistant streams a response that may include tool calls
        const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
        newMessages.push(message);

        if (message.stopReason === "error" || message.stopReason === "aborted") {
            await emit({ type: "turn_end", message, toolResults: [] });
            await emit({ type: "agent_end", messages: newMessages });
            return;
        }

        // "Observation" — execute tool calls and fold results back into the context
        const toolCalls = message.content.filter((c) => c.type === "toolCall");
        const toolResults: ToolResultMessage[] = [];
        hasMoreToolCalls = false;
        if (toolCalls.length > 0) {
            const executedToolBatch = await executeToolCalls(currentContext, message, config, signal, emit);
            toolResults.push(...executedToolBatch.messages);
            hasMoreToolCalls = !executedToolBatch.terminate;
            for (const result of toolResults) {
                currentContext.messages.push(result);
                newMessages.push(result);
            }
        }

        await emit({ type: "turn_end", message, toolResults });
        // ... prepareNextTurn / steering / follow-up handling
    }
}
```

**Why relevant:** This **is** the ReAct loop. Map it onto the canonical pattern:

- **Thought + Action**: `streamAssistantResponse` produces an assistant message that may contain `thinking` blocks (reasoning) and `toolCall` blocks (actions).
- **Observation**: `executeToolCalls` runs the actions and produces tool result messages pushed onto the context.
- **Loop**: `hasMoreToolCalls = true` keeps the cycle running; the loop only terminates when the assistant emits no tool calls (or sets `terminate`).
- **Steering**: `pendingMessages` (from `getSteeringMessages()`) lets the human inject mid-cycle — a Pi addition that the original ReAct paper doesn't address.

### Context build per call — `transformContext` then `convertToLlm`

```ts
// packages/agent/src/agent-loop.ts:282-296
let messages = context.messages;
if (config.transformContext) {
    messages = await config.transformContext(messages, signal);
}

const llmMessages = await config.convertToLlm(messages);

const llmContext: Context = {
    systemPrompt: context.systemPrompt,
    messages: llmMessages,
    tools: context.tools,
};
```

**Why relevant:** Before every LLM call, Pi (1) optionally lets a consumer transform the message list (pruning, retrieval injection, summarization — see `memory/context_management/pi.md`), then (2) converts AgentMessages to provider-native Messages, then (3) hands the model the system prompt + transcript + available tools. ReAct in Pi is **tool-native**: the model sees both the running transcript and the available tools every turn.

### Thinking vs text vs tool calls — first-class reasoning support

```ts
// packages/ai/src/types.ts:347-360 (excerpted from AssistantMessageEvent)
export type AssistantMessageEvent =
    | { type: "start"; partial: AssistantMessage }
    | { type: "text_start"; contentIndex: number; partial: AssistantMessage }
    | { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
    | { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
    | { type: "thinking_start"; contentIndex: number; partial: AssistantMessage }
    | { type: "thinking_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
    | { type: "thinking_end"; contentIndex: number; content: string; partial: AssistantMessage }
    | { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
    | { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
    | { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
    | { type: "done"; reason: Extract<StopReason, "stop" | "length" | "toolUse">; message: AssistantMessage }
    | { type: "error"; reason: Extract<StopReason, "aborted" | "error">; error: AssistantMessage };
```

**Why relevant:** Reasoning is a **separately-typed** content block (`thinking_*` events) at the provider abstraction layer. Pi distinguishes:

- **`thinking`** content: model's internal reasoning (Anthropic extended thinking, OpenAI o-series, etc.). Captured, persisted, often hidden from the user by default but available for inspection.
- **`text`** content: user-facing assistant output.
- **`toolCall`** content: actions requested by the model.

This separation is the "explicit reasoning trace" half of ReAct — enforced by the type system, not just convention. `AgentLoopConfig` even has a `thinkingLevel` field (`"off" | "minimal" | "low" | "medium" | "high" | "xhigh"`) to tune reasoning depth.

### Steering — a Pi-specific extension of ReAct

```ts
// packages/agent/src/types.ts:222-232
/**
 * Returns steering messages to inject into the conversation mid-run.
 *
 * Called after the current assistant turn finishes executing its tool calls, unless `shouldStopAfterTurn` exits first.
 * If messages are returned, they are added to the context before the next LLM call.
 * Tool calls from the current assistant message are not skipped.
 *
 * Use this for "steering" the agent while it's working.
 */
getSteeringMessages?: () => Promise<AgentMessage[]>;
```

```ts
// packages/agent/src/agent-loop.ts:253-260 (steering + follow-up cycle wiring)
pendingMessages = (await config.getSteeringMessages?.()) || [];
}

const followUpMessages = (await config.getFollowUpMessages?.()) || [];
if (followUpMessages.length > 0) {
    pendingMessages = followUpMessages;
    continue;
}
```

**Why relevant:** A human typing while the agent is in its inner loop gets injected at the next safe point (after the current turn's tool calls finish, before the next assistant call). This is a *real-time human-in-the-loop* extension of ReAct that the classic pattern doesn't include — the agent's loop isn't fire-and-forget. **Follow-up** messages are similar but fire only after the agent would otherwise stop, letting queued work resume the loop.

### Continuation — start ReAct from a saved state

```ts
// packages/agent/src/agent-loop.ts:121-143
export async function runAgentLoopContinue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal?: AbortSignal,
    streamFn?: StreamFn,
): Promise<AgentMessage[]> {
    if (context.messages.length === 0) {
        throw new Error("Cannot continue: no messages in context");
    }

    if (context.messages[context.messages.length - 1].role === "assistant") {
        throw new Error("Cannot continue from message role: assistant");
    }

    const newMessages: AgentMessage[] = [];
    const currentContext: AgentContext = { ...context };

    await emit({ type: "agent_start" });
    await emit({ type: "turn_start" });

    await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
    return newMessages;
}
```

**Why relevant:** Second entry point that picks up an existing context and continues the ReAct loop. Requires the last message to be a user / tool-result / custom message (not assistant — that would mean the model already responded). This is what powers `--continue` and `--resume` from the CLI: re-hydrate context from a saved session and re-enter the ReAct loop.

### Event stream — the ReAct trace is observable

The loop emits these events in order per cycle:

1. `turn_start`
2. (`message_start` / `message_end` for pending/steering messages)
3. `text_start` / `text_delta` / `text_end` (assistant reasoning + final text)
4. `thinking_start` / `thinking_delta` / `thinking_end` (model reasoning, if exposed)
5. `toolcall_start` / `toolcall_delta` / `toolcall_end` (per tool call)
6. `tool_execution_start` / `tool_execution_end` (per tool execution)
7. `turn_end` (carries the assistant message + tool result messages)

**Why relevant:** A subscriber sees the entire Thought-Action-Observation trace, in order, with structural distinction between reasoning and output and action and result. This is the "transparent reasoning" half of ReAct's value proposition, available as a structured event stream rather than a flat text log.

## Tradeoffs and limitations

- **No explicit "Thought:" / "Action:" / "Observation:" string markers** in the prompt. Pi relies on provider-native tool-call APIs (every modern provider supports them), not on parsing tagged text. Correct for 2026 but means Pi's ReAct doesn't look like the textbook 2022 ReAct prompts — the structure is in the message types, not in the prompt prose.
- **Reasoning visibility depends on the provider.** Models that don't expose a separate thinking channel just emit text + tool calls. Framework handles both gracefully.
- **No iteration limit at the framework level.** The inner loop continues as long as the model emits tool calls. A consumer who wants "max 10 iterations" wires it via `shouldStopAfterTurn`.
- **No built-in retry on tool failure.** If a tool fails, the error result goes back to the model; the model decides whether to retry. (Provider-level retry on transient errors is a separate mechanism — see `4_reflection/pi.md`.)
- **Streaming with tool calls is a constraint.** Tool calls happen at well-defined points in the message structure — partial streaming of tool args is supported (`toolcall_delta`), but tool execution waits for `toolcall_end`.
- **One agent at a time.** Pi's ReAct loop is single-agent. Concurrent multi-agent ReAct requires the subagent extension (one ReAct loop per subprocess).

## "Not implemented" caveats

- ❌ Text-tagged ReAct (no `Thought:` / `Action:` / `Observation:` parsing — Pi uses native provider APIs)
- ❌ Built-in iteration cap (consumer's responsibility via `shouldStopAfterTurn`)
- ❌ Built-in tool-failure retry policy
- ❌ Step-level reward/scoring (no PRM integration)

What Pi does ship that the ReAct pattern asks for:

- ✅ Single agent loop alternating between assistant streaming and tool execution
- ✅ Typed separation of reasoning (`thinking_*`) from output (`text_*`) from action (`toolcall_*`)
- ✅ `thinkingLevel` configuration for tuning reasoning depth
- ✅ Observation as first-class message type (tool results flow back into the next call's context)
- ✅ Loop termination on no-more-tool-calls or `terminate` flag from a tool result
- ✅ Continuation entry point (`runAgentLoopContinue`) — resume ReAct from a saved state
- ✅ Event stream exposing the full Thought-Action-Observation trace structurally
- ✅ Mid-run steering — human can inject messages between turns without breaking the loop
- ✅ Follow-up messages — queued messages that fire when the agent would otherwise stop
- ✅ Graceful-stop hook (`shouldStopAfterTurn`) for consumer-defined termination policies
- ✅ `transformContext` hook for custom message-list modification before each LLM call
