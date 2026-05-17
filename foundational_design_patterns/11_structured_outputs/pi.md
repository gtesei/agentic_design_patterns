# Pi — Structured Outputs (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Partial — structured *input* (tool parameters) is first-class; structured *transport-level event stream* is built-in; structured *final answer* is a tool-pattern, not a separate API.** Three layers of structured-output story in Pi:

1. **TypeBox-typed tool parameters (`Tool<TParameters extends TSchema>`)** — every tool's `parameters` field is a TypeBox schema that serves three purposes simultaneously: runtime validation of model arguments, JSON Schema sent to the LLM, TypeScript static type via `Static<TParameters>`. This is the structured-output story for **inputs to tools**, and it's how Pi gets schema-validated structured data out of the model: by asking the model to call a tool with typed parameters. The schema flows **directly to provider-facing function definitions** with no conversion step (TypeBox is JSON-Schema-native).

2. **`--mode json` JSON event stream** — Pi's print-mode `json` output serializes every event from the agent loop (`message_end`, `tool_execution_*`, `turn_end`, etc.) as JSONL on stdout. This is a **transport-level structured output channel** for any external consumer (the SDK, the subagent extension, scripts piping `pi --mode json | jq`). I missed this in my first read.

3. **The `structured-output.ts` extension** — a 65-line reference extension that registers a `structured_output` tool with `terminate: true`. The agent calls this tool as its final action; the tool returns immediately (no follow-up LLM turn) and surfaces the structured payload via the tool result's `details` field. This is the *structured-final-answer* pattern, implemented as a tool because Pi's framework has no separate "respond with this schema" mode.

Pi has no `generateObject` / `withStructuredOutput` / `response_format: json_schema` equivalent at the framework level. Structured-output-via-tool-calls is the dominant idiom — which is what OpenAI / Anthropic / Google all converged on for function calling. Pi doesn't paper over it with a separate shorthand.

> Addition to my first read: I missed `--mode json` entirely. It's a real first-class structured-output surface Pi ships at the transport layer.

## Where it lives

| Concern | File:line |
|---|---|
| `Tool<TParameters extends TSchema>` — LLM-facing tool with typed params | `packages/ai/src/types.ts:325-330` |
| `AgentTool` — runtime extension with `Static<TParameters>` in `execute` | `packages/agent/src/types.ts:361-385` |
| `Static`, `TSchema`, `Type` re-exports for consumers | `packages/ai/src/index.ts:1-2` |
| Example tool with `Type.Object` schema (bash) | `packages/coding-agent/src/core/tools/bash.ts:23-32` |
| `--mode json` JSON event stream | `packages/coding-agent/src/modes/print-mode.ts:6, 18, 104-114` |
| `structured-output.ts` reference extension | `packages/coding-agent/examples/extensions/structured-output.ts:18-43` |
| Extension API `ToolDefinition` (parameters: TSchema) | `packages/coding-agent/src/core/extensions/types.ts:426-461` |
| `terminate: true` field on `AgentToolResult` | `packages/agent/src/types.ts:344-354` |
| Batch-aware termination policy (unanimous `terminate`) | `packages/agent/src/agent-loop.ts:534-535` |
| Schema preservation into provider function definitions | `packages/ai/src/providers/openai-responses-shared.ts:268-276` |

## Key code excerpts

### Tool parameters as a TypeBox schema

```ts
// packages/ai/src/types.ts:325-330
import type { TSchema } from "typebox";

export interface Tool<TParameters extends TSchema = TSchema> {
    name: string;
    description: string;
    parameters: TParameters;
}
```

```ts
// packages/ai/src/index.ts:1-2
export type { Static, TSchema } from "typebox";
export { Type } from "typebox";
```

**Why relevant:** Pi re-exports TypeBox at the `pi-ai` package level. Consumers don't need to depend on TypeBox directly — they import `Type`, `TSchema`, `Static` from `pi-ai`. The schema-authoring story is uniform across the ecosystem.

### Strict-schema tool — the canonical structured-input pattern

```ts
// packages/coding-agent/src/core/tools/bash.ts:23-32
const bashSchema = Type.Object({
    command: Type.String({ description: "Bash command to execute" }),
    timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (optional, no default timeout)" })),
});

export type BashToolInput = Static<typeof bashSchema>;
```

**Why relevant:** *The* structured-output pattern in Pi. The bash tool declares `Type.Object({...})`; that schema is sent to the LLM (as JSON Schema — model knows the shape it must produce), validated at runtime when the model emits a tool call (Pi rejects malformed args before `execute()` is called), and inferred into `BashToolInput` (TypeScript) for the implementation. One source of truth.

### `--mode json` — built-in transport-level structured output

```ts
// packages/coding-agent/src/modes/print-mode.ts:6, 18, 104-114 (excerpted)
// `pi --mode json "prompt"` - JSON event stream
mode: "text" | "json";
// ...
if (mode === "json") {
    writeRawStdout(`${JSON.stringify(event)}\n`);
}
// ...
if (mode === "json") {
    // ...
    writeRawStdout(`${JSON.stringify(header)}\n`);
}
```

**Why relevant:** Built-in JSON event stream mode. Every agent event — assistant messages, tool calls, tool results, turn boundaries, errors — gets serialized as JSONL on stdout. This is the structured-output surface for **machine consumers**: pipe `pi --mode json` to another process and you get a parseable stream of typed events. The subagent extension uses exactly this to read subagent results (see `7_multi_agent_collaboration/pi.md`).

### Extension API `ToolDefinition` — schema-based contract

```ts
// packages/coding-agent/src/core/extensions/types.ts:426-461 (excerpted)
export interface ToolDefinition<TParams extends TSchema = TSchema, TDetails = unknown, TState = any> {
    name: string;
    label: string;
    description: string;
    // ...
    /** Parameter schema (TypeBox) */
    parameters: TParams;
    // ...
    execute(
        toolCallId: string,
        params: Static<TParams>,
        signal: AbortSignal | undefined,
        onUpdate: AgentToolUpdateCallback<TDetails> | undefined,
        ctx: ExtensionContext,
    ): Promise<AgentToolResult<TDetails>>;
}
```

**Why relevant:** Structure is enforced through the tool interface itself, not only through prompt wording. Extension-authored tools get the same TypeBox / `Static<>` story as core tools — `params` is typed.

### `structured-output.ts` extension — final-answer-as-tool-call

```ts
// packages/coding-agent/examples/extensions/structured-output.ts:18-43 (excerpted)
const structuredOutputTool = defineTool({
    name: "structured_output",
    label: "Structured Output",
    description:
        "Return a final structured answer. Use this as your last action when the user asks for structured output or a machine-readable summary.",
    promptSnippet: "Emit a final structured answer as a terminating tool result",
    promptGuidelines: [
        "Use structured_output as your final action when the user asks for structured output, JSON-like output, or a machine-readable summary.",
        "After calling structured_output, do not emit another assistant response in the same turn.",
    ],
    parameters: Type.Object({
        headline: Type.String({ description: "Short title for the result" }),
        summary: Type.String({ description: "One-paragraph summary" }),
        actionItems: Type.Array(Type.String(), { description: "Concrete next steps or key bullets" }),
    }),

    async execute(_toolCallId, params) {
        return {
            content: [{ type: "text", text: `Saved structured output: ${params.headline}` }],
            details: {
                headline: params.headline,
                summary: params.summary,
                actionItems: params.actionItems,
            } satisfies StructuredOutputDetails,
            terminate: true,
        };
    },
    // ... renderResult for custom TUI rendering
});
```

**Why relevant:** The cleanest reference for "I want the agent to return a typed object as its final answer." Three things to notice:

1. **`terminate: true`** ends the agent loop after this tool result. No follow-up LLM turn is needed; the agent's last act is the structured tool call itself. Saves tokens and latency.
2. **`details`** holds the structured payload. `content` is what goes back into the model's context (a short ack); `details` is for downstream consumers. The structured data doesn't have to round-trip through model context.
3. **`promptGuidelines`** — instructions appended to the system prompt that teach the model when to use this tool and to **not emit another assistant response in the same turn**. Critical: without the guideline the model might call `structured_output` and *then* try to respond again, wasting a turn.

### Batch-aware termination — `terminate: true` requires unanimity

```ts
// packages/agent/src/agent-loop.ts:534-535
function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
    return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}
```

**Why relevant:** A single `terminate: true` doesn't kill the loop if other parallel tool calls in the same batch don't agree. The `structured_output` pattern works cleanly when called alone (one-tool batch); if the LLM emits `structured_output` alongside other tool calls in the same turn, the loop continues. This is a deliberate design — protects against accidental early termination.

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

**Why relevant:** Pi preserves the schema all the way to the provider-facing tool contract. The comment `TypeBox already generates JSON Schema` is the point: no `zodToJsonSchema` step, no runtime conversion. This is why Pi picked TypeBox over Zod — zero-cost provider-facing schemas.

### Comparison to canonical APIs in other frameworks

| Need | Other frameworks (2026) | Pi |
|---|---|---|
| Structured tool input | `tool(zodSchema)` / `@tool(args_schema=Pydantic)` | `parameters: Type.Object({...})` (TypeBox) |
| Structured final answer | `generateObject({ schema })` / `withStructuredOutput()` | Tool with `terminate: true` + `details` (the `structured-output.ts` pattern) |
| JSON-mode response (no tools) | `response_format: { type: "json_object" }` | Not exposed at the agent layer; drop to `ai` package's `complete`/`stream` directly |
| Constrained decoding | XGrammar / Outlines integration | Not implemented |
| Machine-consumable event stream | Usually per-framework | **Built-in: `pi --mode json`** |

## Tradeoffs and limitations

- **TypeBox, not Zod.** Most of the 2026 TypeScript agent ecosystem standardized on Zod. Pi uses TypeBox because it produces JSON Schema natively (no `zodToJsonSchema` step) and the runtime is smaller. Slightly less developer mindshare; `Type.Object({...})` syntax is unfamiliar to Zod users.
- **No `generateObject` shorthand.** A consumer who wants "extract this struct from this text in one call" must register a single-purpose tool, prompt the model to call it, and check the tool result's `details`. More boilerplate than `generateObject({ schema })`.
- **No provider-native JSON-mode passthrough at the agent layer.** Some providers (OpenAI Responses, Anthropic) support strict JSON output as a model setting. Pi's agent loop is tool-call-shaped, so it doesn't expose this as a first-class option. Drop to the `ai` package's `complete()` or `stream()` and bypass the agent.
- **No grammar-constrained decoding** (Outlines, XGrammar, llama.cpp grammars). Not in scope.
- **`structured-output.ts` is an example, not a built-in.** The `terminate: true` pattern is in core (any tool can use it), but the convention of "use a tool named `structured_output`" is just an extension.
- **No Instructor-style retry-on-validation-failure.** If a tool's parameters fail schema validation, the agent gets an error result and the model decides whether to retry on its next turn. No framework-level "retry with the validation error inlined into the prompt" loop.
- **The `promptGuidelines` field on the tool is critical for correct behavior.** Without it, the model may call `structured_output` and then also emit text — paying for a turn the `terminate` flag was supposed to save.
- **`terminate: true` requires unanimous batch agreement.** If the model calls `structured_output` alongside other tools, the loop continues — even though the structured payload is captured.

## "Not implemented" caveats

- ❌ `generateObject` / `withStructuredOutput` equivalent at the framework level
- ❌ JSON-mode response-format passthrough at the agent layer (drop to `ai` package directly)
- ❌ Constrained-grammar decoding (Outlines / XGrammar)
- ❌ Zod (TypeBox only)
- ❌ Built-in retry-on-validation-failure (the model handles it via the error result)
- ❌ Schema-versioned final-answer migration

What Pi does ship that the structured-outputs pattern asks for:

- ✅ TypeBox-typed tool parameters end-to-end (runtime validation + JSON Schema + static TS type)
- ✅ `Static<TParameters>` for type-safe `execute()` signatures
- ✅ `prepareArguments` shim for tolerating slightly-off model output
- ✅ `terminate: true` on `AgentToolResult` for ending the loop without a follow-up LLM turn
- ✅ `details` field on `AgentToolResult` for surfacing structured data without round-tripping through model context
- ✅ Built-in `--mode json` transport-level event stream for machine consumers
- ✅ `structured-output.ts` reference extension demonstrating the final-answer-as-tool-call pattern
- ✅ `promptGuidelines` field on tools for instructing the model when (and how) to call the tool
- ✅ Provider-neutral schemas — the same TypeBox schema works across Anthropic / OpenAI / Google / Bedrock / Mistral / Azure tool-use APIs
- ✅ Schema preserved verbatim into provider function definitions (no conversion overhead)
