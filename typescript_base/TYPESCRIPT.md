# TypeScript Track

The TypeScript track is a curated subset of the catalog. **Python remains canonical**; TS exists where it adds pedagogical value (type-safe tool schemas, AI SDK idioms, edge-deploy readiness) or where the reader plausibly works in Node/Bun day-to-day.

## Current coverage

| Pattern | Anchor scenario | Framework | Path |
|---|---|---|---|
| `1_prompt_chain` | Support Ops | Vercel AI SDK (`generateObject`) | `foundational_design_patterns/1_prompt_chain/typescript/` |
| `5_tool_use` | Coding Agent | Vercel AI SDK (`tools`, `stepCountIs`) | `foundational_design_patterns/5_tool_use/typescript/` |
| `6_planning` | Incident Response | LangGraph.js (`StateGraph`) | `foundational_design_patterns/6_planning/typescript/` |

More patterns will land in subsequent passes (target subset: routing, parallelization, reflection, ReAct, RAG, HITL, simple multi-agent, observability, plus new HIGH-priority chapters Structured Outputs, Subagents, Skills).

## Conventions

- **Runtime**: [Bun](https://bun.sh) (≥ 1.3). Runs `.ts` directly — no build step, no tsx.
- **Package manager**: Bun (workspaces declared in root `package.json`, lockfile `bun.lock`).
- **Test runner**: `bun test` (built-in, Jest-like API — no vitest dep).
- **Schemas**: Zod, end-to-end. Tool inputs, structured outputs, graph state all share Zod schemas.
- **Module system**: ESM (`"type": "module"`), strict TS via `tsconfig.base.json`.
- **Provider abstraction**: `@ai-sdk/openai` for AI-SDK examples; `@langchain/openai` for LangGraph examples. Default model `gpt-4o-mini`, overridable via `MODEL` env var.

## Framework picks

- **Vercel AI SDK v6** — substrate for tool-loop and prompt-shaping patterns. Carries `generateObject` for schema-enforced output, `tools` for Zod-typed function calling, `stopWhen` for agent-loop termination.
- **LangGraph.js** — used only where the *state machine itself* is the lesson (planning, future: subagents, multi-agent orchestration). For tool-loop patterns, AI SDK is more direct.
- Not currently used: Mastra (will appear for memory/workflow patterns), Trigger.dev (will appear as an optional durability overlay for HITL).

## Running

### From the workspace root (`typescript_base/`)

```bash
cd typescript_base
bun install                              # install all workspace members
bun run smoke                            # equivalent to: bash ../scripts/run_demos_smoke_typescript.sh --mode basic
bun run demos                            # equivalent to: ... --mode full (requires OPENAI_API_KEY)
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
6. Run `bun test` locally; then `bash scripts/run_demos_smoke_typescript.sh --mode basic --pattern <substring>`.

## Why Bun (not Node + tsx + pnpm)

User decision. Bun gives us: one binary for runtime + install + test, native `.ts` execution, faster install/test, no `tsx` dep, no `vitest` dep, native ESM, native `.env` loading. Trade-off: Bun is less ubiquitous than Node in 2026 enterprise but mature enough; readers comfortable with Node can substitute `pnpm install && pnpm tsx src/<file>.ts` with no source changes.
