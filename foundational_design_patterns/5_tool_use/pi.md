# Pi — Tool Use (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Yes — tool use is Pi's core competency.** Pi is built around the agent-with-tools model: a single agent loop that emits tool calls, the framework validates and dispatches them, results flow back into the next LLM call. Three layers cooperate:

1. **`packages/ai`** — provider-agnostic `Tool<TParameters>` type using **TypeBox** schemas. The LLM-facing contract.
2. **`packages/agent`** — the runtime `AgentTool` interface extending `Tool` with `execute`, `label`, `prepareArguments`, `executionMode`, plus the agent loop that drives dispatch with pre/post hooks.
3. **`packages/coding-agent`** — seven concrete built-in tools (`bash`, `read`, `write`, `edit`, `grep`, `find`, `ls`) assembled via `createAllToolDefinitions`, plus the extension system that lets users register custom tools via `pi.registerTool()`.

Strong properties beyond the basics:

- **Strict schema validation** at the framework level (TypeBox runtime + TypeScript static types). Schemas carry **all the way to provider-facing function definitions** with no conversion step (TypeBox produces JSON Schema natively).
- **Parallel-by-default execution** (see `3_parallelization/pi.md`) with per-tool sequential override.
- **`beforeToolCall` / `afterToolCall` hooks** for confirmation, redaction, override, blocking. Extension events `tool_call` / `tool_result` are wired through these hooks.
- **Multimodal tool results** — tools can return text or image content blocks (`read` returns inline base64 images when the file is an image).
- **Streaming updates** — tools can call `onUpdate` mid-execution to surface partial progress via `tool_execution_update` events.
- **`prepareArguments` shim** for tolerating malformed model output without losing strict typing.
- **`terminate: true` flag** on tool results to end the loop without a follow-up LLM turn (used by `structured-output.ts` extension).

## Where it lives

| Concern | File:line |
|---|---|
| LLM-facing `Tool` type | `packages/ai/src/types.ts:325-330` |
| Runtime `AgentTool` interface (extends `Tool`) | `packages/agent/src/types.ts:361-385` |
| Tool result type (text + image content + structured details) | `packages/agent/src/types.ts:344-354` |
| Pre/post hooks | `packages/agent/src/types.ts:262-281` |
| Sequential vs parallel dispatch | `packages/agent/src/agent-loop.ts:380-388` |
| Argument preparation + validation + execute + post-hook | `packages/agent/src/agent-loop.ts:568-684` |
| Built-in tool inventory (constants + assembly) | `packages/coding-agent/src/core/tools/index.ts:83-95, 156-163` |
| `bash` tool (definition + schema) | `packages/coding-agent/src/core/tools/bash.ts:23-32` |
| `read` tool with image content support | `packages/coding-agent/src/core/tools/read.ts:246-275` |
| Extension `tool_call` / `tool_result` → framework hooks wiring | `packages/coding-agent/src/core/agent-session.ts:378-427` |
| Extension API `ToolDefinition` (parameters: TSchema) | `packages/coding-agent/src/core/extensions/types.ts:426-461` |
| Schema preservation into provider function definitions | `packages/ai/src/providers/openai-responses-shared.ts:268-276` |

## Key code excerpts

### LLM-facing `Tool` — provider-neutral, TypeBox-typed

```ts
// packages/ai/src/types.ts:325-330
import type { TSchema } from "typebox";

export interface Tool<TParameters extends TSchema = TSchema> {
    name: string;
    description: string;
    parameters: TParameters;
}
```

**Why relevant:** The contract every LLM provider in Pi's `ai` package consumes — Anthropic, OpenAI Responses, OpenAI Completions, Bedrock, Google, Google Vertex, Mistral, Azure OpenAI. The `TParameters extends TSchema` generic preserves the schema's TypeScript shape end-to-end: `Static<TParameters>` gives the validated input type at runtime with zero `any` leaks.

### Runtime `AgentTool` — execute, label, mode, shim

```ts
// packages/agent/src/types.ts:361-385
export interface AgentTool<TParameters extends TSchema = TSchema, TDetails = any> extends Tool<TParameters> {
    /** Human-readable label for UI display. */
    label: string;
    /**
     * Optional compatibility shim for raw tool-call arguments before schema validation.
     * Must return an object that matches `TParameters`.
     */
    prepareArguments?: (args: unknown) => Static<TParameters>;
    /** Execute the tool call. Throw on failure instead of encoding errors in `content`. */
    execute: (
        toolCallId: string,
        params: Static<TParameters>,
        signal?: AbortSignal,
        onUpdate?: AgentToolUpdateCallback<TDetails>,
    ) => Promise<AgentToolResult<TDetails>>;
    /**
     * Per-tool execution mode override.
     */
    executionMode?: ToolExecutionMode;
}
```

**Why relevant:** Runtime contract documents the **exception model**: throw on failure, don't encode errors in `content`. The loop wraps the throw and synthesizes an error tool result. Tools don't have to manually format error envelopes for every provider. `prepareArguments` is the compatibility seam for models that emit slightly-off JSON.

### Tool result — text or image, plus structured `details`

```ts
// packages/agent/src/types.ts:344-354 (AgentToolResult)
export interface AgentToolResult<T> {
    content: (TextContent | ImageContent)[];
    details: T;
    /**
     * Hint that the agent should stop after the current tool batch.
     * Early termination only happens when every finalized tool result in the batch sets this to true.
     */
    terminate?: boolean;
}
```

**Why relevant:** Three things matter: (1) `content` is what the model sees; (2) `details` is for the UI / logs / extensions and is NOT sent back to the model; (3) `terminate` lets a tool tell the loop "no need for another LLM turn" — but requires unanimous batch agreement (see `3_parallelization/pi.md`).

### Tool lifecycle in the agent loop — prepare → validate → before → execute → after

```ts
// packages/agent/src/agent-loop.ts:568-684 (excerpted)
const preparedToolCall = prepareToolCallArguments(tool, toolCall);
const validatedArgs = validateToolArguments(tool, preparedToolCall);
if (config.beforeToolCall) {
    const beforeResult = await config.beforeToolCall(
        { assistantMessage, toolCall, args: validatedArgs, context: currentContext },
        signal,
    );
    if (beforeResult?.block) {
        return {
            kind: "immediate",
            result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
            isError: true,
        };
    }
}
// ...
const result = await prepared.tool.execute(
    prepared.toolCall.id,
    prepared.args as never,
    signal,
    (partialResult) => {
        updateEvents.push(Promise.resolve(emit({
            type: "tool_execution_update",
            toolCallId: prepared.toolCall.id,
            toolName: prepared.toolCall.name,
            args: prepared.toolCall.arguments,
            partialResult,
        })));
    },
);
// ...
if (config.afterToolCall) {
    const afterResult = await config.afterToolCall(
        { assistantMessage, toolCall: prepared.toolCall, args: prepared.args, result, isError, context: currentContext },
        signal,
    );
```

**Why relevant:** The complete five-step tool lifecycle. `prepareArguments` runs **before** validation; `beforeToolCall` runs **after** validation but before execution (so the hook sees typed args); `onUpdate` callbacks emit `tool_execution_update` events for streaming partial results back to the UI; `afterToolCall` can override the final result.

### Multimodal tool result — `read` returns inline images

```ts
// packages/coding-agent/src/core/tools/read.ts:246-275 (excerpted)
let content: (TextContent | ImageContent)[];
// ... when the file is an image:
const resized = await resizeImage({ type: "image", data: base64, mimeType });
// ...
{ type: "image", data: resized.data, mimeType: resized.mimeType },
// ...
{ type: "image", data: base64, mimeType },
```

**Why relevant:** The `read` tool inspects file MIME type; if it's an image, it embeds the binary as an inline `ImageContent` block (with automatic resize to 2000x2000 max). Real multimodal tool use — the model receives the image directly and can reason over it on its next turn.

### Built-in tool inventory — seven file/shell tools

```ts
// packages/coding-agent/src/core/tools/index.ts:83-95
export type ToolName = "read" | "bash" | "edit" | "write" | "grep" | "find" | "ls";
export const allToolNames: Set<ToolName> = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);

export interface ToolsOptions {
    read?: ReadToolOptions;
    bash?: BashToolOptions;
    write?: WriteToolOptions;
    edit?: EditToolOptions;
    grep?: GrepToolOptions;
    find?: FindToolOptions;
    ls?: LsToolOptions;
}
```

```ts
// packages/coding-agent/src/core/tools/index.ts:156-163
export function createAllToolDefinitions(cwd: string, options?: ToolsOptions): Record<ToolName, ToolDef> {
    return {
        read: createReadToolDefinition(cwd, options?.read),
        bash: createBashToolDefinition(cwd, options?.bash),
        edit: createEditToolDefinition(cwd, options?.edit),
        write: createWriteToolDefinition(cwd, options?.write),
        grep: createGrepToolDefinition(cwd, options?.grep),
        find: createFindToolDefinition(cwd, options?.find),
        ls: createLsToolDefinition(cwd, options?.ls),
    };
}
```

**Why relevant:** Pi's built-in tool surface is small and explicitly coding-agent-oriented: file system + shell. No HTTP, no browser, no DB. Narrowness is by design — broader tooling is delegated to extensions (~70 examples in `examples/extensions/`).

### Bash tool — typed schema (one source of truth)

```ts
// packages/coding-agent/src/core/tools/bash.ts:23-32
const bashSchema = Type.Object({
    command: Type.String({ description: "Bash command to execute" }),
    timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (optional, no default timeout)" })),
});

export type BashToolInput = Static<typeof bashSchema>;
```

**Why relevant:** Schema serves three purposes simultaneously: (1) runtime validation, (2) JSON Schema for the LLM, (3) `BashToolInput` static type for the implementation. One source of truth.

### Pre/post hooks — block, override, redact

```ts
// packages/agent/src/types.ts:50-58 (BeforeToolCallResult)
export interface BeforeToolCallResult {
    block?: boolean;
    reason?: string;
}

// packages/agent/src/types.ts:262-281 (the two hooks)
beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
// ...
afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
```

**Why relevant:** `beforeToolCall` is the policy seam — confirmation, dangerous-command detection, dry-run, sandbox enforcement. `afterToolCall` is the redaction/override seam. See `10_hitl/pi.md` for extensions that wire these.

### Extension events → framework hooks wiring

```ts
// packages/coding-agent/src/core/agent-session.ts:378-427 (excerpted)
private _installAgentToolHooks(): void {
    this.agent.beforeToolCall = async ({ toolCall, args }) => {
        const runner = this._extensionRunner;
        if (!runner.hasHandlers("tool_call")) return undefined;
        // ...
        return await runner.emitToolCall({
            type: "tool_call",
            toolName: toolCall.name,
            toolCallId: toolCall.id,
            input: args as Record<string, unknown>,
        });
    };

    this.agent.afterToolCall = async ({ toolCall, args, result, isError }) => {
        const runner = this._extensionRunner;
        if (!runner.hasHandlers("tool_result")) return undefined;
        // ...
        return { content: hookResult.content, details: hookResult.details, isError: hookResult.isError ?? isError };
    };
}
```

**Why relevant:** Extensions don't register `beforeToolCall` directly — they subscribe to `tool_call` / `tool_result` events via `pi.on(...)`. This `_installAgentToolHooks` wires the extension event runner into the framework hooks. Multiple extensions can subscribe; the runner aggregates their responses.

### Schemas carried all the way to provider function definitions

```ts
// packages/ai/src/providers/openai-responses-shared.ts:268-276
export function convertResponsesTools(tools: Tool[], options?: ConvertResponsesToolsOptions): OpenAITool[] {
    const strict = options?.strict === undefined ? false : options.strict;
    return tools.map((tool) => ({
        type: "function",
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters as any, // TypeBox already generates JSON Schema
        strict,
    }));
}
```

**Why relevant:** TypeBox schema goes **straight through** to the provider's function-call API — no `zodToJsonSchema` step, no runtime conversion. The comment is the point: TypeBox is JSON-Schema-native. This is why Pi picked TypeBox over Zod: zero-cost provider-facing schemas.

### Extension-registered tool example

```ts
// packages/coding-agent/examples/extensions/subagent/index.ts:432-466 (excerpted)
export default function (pi: ExtensionAPI) {
    pi.registerTool({
        name: "subagent",
        label: "Subagent",
        description: [
            "Delegate tasks to specialized subagents with isolated context.",
            // ...
        ].join(" "),
        parameters: SubagentParams,
        async execute(_toolCallId, params, signal, onUpdate, ctx) { /* ... */ },
    });
}
```

**Why relevant:** Custom tools are first-class. An extension is an ES module exporting a default function taking `ExtensionAPI`; calling `pi.registerTool(...)` uses the same `AgentTool`-shape signature core tools use. Extension tools and core tools are interchangeable from the agent loop's perspective.

## Tradeoffs and limitations

- **TypeBox, not Zod.** Most of the 2026 TypeScript agent ecosystem standardized on Zod. Pi picked TypeBox for native JSON Schema (no conversion step). Slightly less developer mindshare; `Type.Object({...})` is unfamiliar to Zod users.
- **Tools must throw on failure** — not return an error envelope. Easy to get wrong if porting from frameworks that prefer the latter. The error becomes synthesized content with a default message; the original exception is captured into `errorMessage`.
- **`prepareArguments` is a foot-cannon** if used carelessly — runs *before* schema validation, so a sloppy shim can produce invalid args that then fail validation with a confusing error.
- **No tool-result streaming back to the LLM mid-execution.** `onUpdate` / `tool_execution_update` streams partials to the UI, but the model only sees the final result when the tool finishes. Long-running tools block the next LLM call.
- **Per-tool `executionMode: "sequential"` poisons the whole batch** (see `3_parallelization/pi.md`).
- **No tool-call telemetry standard.** Per-tool latency / cost is observable in the event stream but not aggregated.
- **No automatic tool retry.** A tool that fails (throws) produces an error result; the model decides whether to retry, not the framework. (Distinct from provider-level retry on transient errors — see `4_reflection/pi.md`.)
- **Built-in tool surface is narrow by design.** Seven file/shell tools. Anything else (HTTP, DB, browser) is extension territory.

## "Not implemented" caveats relative to the 2026 pattern

- ❌ MCP integration in core (Pi docs explicitly say "intentionally does not include built-in MCP")
- ❌ Built-in tool-call telemetry / metrics aggregation
- ❌ Automatic retry on tool failure (model handles, not framework)
- ❌ Streaming partial tool results into the model context mid-execution
- ❌ Zod schemas (TypeBox only)

What Pi does ship:

- ✅ Strict-schema tool parameters (TypeBox, runtime + static typing in one source)
- ✅ Parallel-by-default tool execution with per-tool sequential override
- ✅ Pre/post hooks (`beforeToolCall`, `afterToolCall`) for confirmation, redaction, override, block
- ✅ Multimodal tool results (text + image content)
- ✅ Streaming progress updates from tools to the UI (`onUpdate` / `tool_execution_update`)
- ✅ `terminate: true` flag for tool-result-driven loop termination
- ✅ Extension-registered tools (interchangeable with core tools)
- ✅ Provider-neutral tool schema (works across Anthropic / OpenAI / Google / Bedrock / Mistral / Azure)
- ✅ Schema preservation end-to-end (TypeBox → provider function definition, no conversion)
- ✅ Seven first-class file/shell tools plus 70+ example extensions
- ✅ Extension `tool_call` / `tool_result` events wired through framework hooks
