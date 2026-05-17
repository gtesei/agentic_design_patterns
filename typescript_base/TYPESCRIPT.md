# TypeScript Track

The TypeScript track currently mirrors the **foundational patterns** in this catalog. **Python remains canonical**; TypeScript ports exist to preserve scenario-level parity while making the same design patterns accessible in the Bun/Node ecosystem.

## Current coverage

| Pattern | Anchor scenario | Framework | Path |
|---|---|---|---|
| `1_prompt_chain` | Support Ops | Vercel AI SDK (`generateObject`) | `foundational_design_patterns/1_prompt_chain/typescript/` |
| `2_routing` | Support Ops routing | LangChain.js LCEL (`RunnableBranch`) | `foundational_design_patterns/2_routing/typescript/` |
| `3_parallelization` | Parallel research synthesis | LangChain.js LCEL (`RunnableMap`) | `foundational_design_patterns/3_parallelization/typescript/` |
| `4_reflection` | Reflection loops | LangChain.js + LangGraph.js | `foundational_design_patterns/4_reflection/typescript/` |
| `5_tool_use` | Coding Agent | Vercel AI SDK (`tools`, `stepCountIs`) | `foundational_design_patterns/5_tool_use/typescript/` |
| `6_planning` | Incident Response | LangGraph.js (`StateGraph`) | `foundational_design_patterns/6_planning/typescript/` |
| `7_multi_agent_collaboration` | Research coordination | LangGraph.js + explicit orchestration | `foundational_design_patterns/7_multi_agent_collaboration/typescript/` |
| `8_react` | Research assistant | LangChain v1 agent + LangGraph.js | `foundational_design_patterns/8_react/typescript/` |
| `9_rag` | Retrieval-augmented generation | LangChain.js + local retrieval logic | `foundational_design_patterns/9_rag/typescript/` |
| `10_hitl` | Human review checkpoints | OpenAI SDK + LangGraph.js | `foundational_design_patterns/10_hitl/typescript/` |
| `11_structured_outputs` | Schema-constrained extraction | LangChain.js + AI SDK | `foundational_design_patterns/11_structured_outputs/typescript/` |
| `12_computer_use` | UI/browser automation framing | LangChain.js + optional Playwright | `foundational_design_patterns/12_computer_use/typescript/` |

Patterns outside the foundational track remain Python-first for now.

## Conventions

- **Runtime**: [Bun](https://bun.sh) (≥ 1.3). Runs `.ts` directly — no build step, no tsx.
- **Package manager**: Bun (workspaces declared in root `package.json`, lockfile `bun.lock`).
- **Test runner**: `bun test` (built-in, Jest-like API — no vitest dep).
- **Schemas**: Zod, end-to-end. Tool inputs, structured outputs, graph state all share Zod schemas.
- **Module system**: ESM (`"type": "module"`), strict TS via `tsconfig.base.json`.
- **Provider abstraction**: `@ai-sdk/openai` for AI-SDK examples; `@langchain/openai` for LangGraph examples. Default model `gpt-4o-mini`, overridable via `OPENAI_MODEL` (and `OPENAI_ADVANCED_MODEL` where the Python demo uses a higher-capability variant).

## Framework picks

- **Vercel AI SDK v6** — substrate for tool-loop and prompt-shaping patterns. Carries `generateObject` for schema-enforced output, `tools` for Zod-typed function calling, `stopWhen` for agent-loop termination.
- **LangGraph.js** — used only where the *state machine itself* is the lesson (planning, future: subagents, multi-agent orchestration). For tool-loop patterns, AI SDK is more direct.
- Not currently used: Mastra (will appear for memory/workflow patterns), Trigger.dev (will appear as an optional durability overlay for HITL).

## Running

### From the workspace root (`typescript_base/`)

```bash
cd typescript_base
bun install                              # install all workspace members
bash ../scripts/run_demos_smoke_typescript.sh --mode basic
bash ../scripts/run_demos_smoke_typescript.sh --mode full     # requires OPENAI_API_KEY
```

### From a single pattern dir

```bash
cd foundational_design_patterns/5_tool_use/typescript
bash run.sh                              # auto-bootstraps workspace install, sources repo-root .env
bun test                                 # offline smoke (no API key)
bun run demo                             # full LLM demo (needs OPENAI_API_KEY in repo-root .env)
```

### Smoke runner

`scripts/run_demos_smoke_typescript.sh` mirrors `scripts/run_demos_smoke.sh` (the Python runner). Same flags (`--mode basic|full`, `--timeout`, `--pattern`), same PASS/FAIL/SKIP_INFRA classification, same `.demo-smoke-logs-ts/<timestamp>/` log layout. See the script's `--help` for details.

## Adding a new TS pattern

1. Mirror the existing skeleton: `<pattern>/typescript/{package.json, tsconfig.json, run.sh, README.md, src/, tests/}`. No `.env.example` — repo-root `.env` is the single source.
2. Name the package `@adp/<short-name>` so the workspace picks it up.
3. Extend `typescript_base/tsconfig.base.json` from each pattern `tsconfig.json` with `"extends": "../../../typescript_base/tsconfig.base.json"`. Don't repeat the strict-mode flags.
4. Write the demo as a **direct port** of the matching Python file: same scenario, same example data, same identifiers, same structural primitives. See AGENTS.md §7 for the parity rule.
5. Add a `tests/<name>.test.ts` smoke file that imports the module and asserts shape. No LLM calls in smoke tests.
6. Run `bun --bun tsc --noEmit` locally; then `bun test`; then `bash scripts/run_demos_smoke_typescript.sh --mode basic --pattern <substring>`.

## Why Bun (not Node + tsx + pnpm)

User decision. Bun gives us: one binary for runtime + install + test, native `.ts` execution, faster install/test, no `tsx` dep, no `vitest` dep, native ESM, native `.env` loading. Trade-off: Bun is less ubiquitous than Node in 2026 enterprise but mature enough; readers comfortable with Node can substitute `pnpm install && pnpm tsx src/<file>.ts` with no source changes.
