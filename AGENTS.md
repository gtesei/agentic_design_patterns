# AGENTS.md

Single source of truth for **any AI agent** working in this repository — Claude Code, OpenAI Codex, Aider, Cursor, Continue, etc. Repo guide + collaboration model + working conventions + project state. Read this first before making changes. Last updated 2026-05-17.

This file follows the [AGENTS.md convention](https://agents.md) so any compliant agent picks it up automatically. There is no `CLAUDE.md`, `CODEX.md`, or other vendor-specific variant — `AGENTS.md` is the only file.

---

## 1. Repository purpose & layout

Educational, hands-on catalog of agentic-AI design patterns. Originally Python (LangChain / LangGraph); a curated **TypeScript track** mirrors a subset of patterns 1:1. Each pattern is self-contained, runnable, and accompanied by a long-form `README.md`. Reference material, not a deployable app.

Patterns live in 7 top-level category directories:

- `foundational_design_patterns/{1_prompt_chain, 2_routing, 3_parallelization, 4_reflection, 5_tool_use, 6_planning, 7_multi_agent_collaboration, 8_react, 9_rag, 10_hitl}`
- `reasoning/{tree_of_thoughts, graph_of_thoughts, exploration_discovery}`
- `reliability/{error_recovery, guardrails}`
- `orchestration/{goal_management, agent_communication, mcp, prioritization}`
- `observability/{evaluation_monitoring, resource_optimization}`
- `memory/{memory_management, context_management}`
- `learning/adaptive_learning`

Each pattern dir has its own `pyproject.toml`, `uv.lock`, `run.sh`, `README.md`, `QUICK_START.md`, and `src/` with example scripts (`*_basic.py`, `*_advanced.py`, or variants like `_plan_and_act.py` / `_hiplan.py`). Pattern packages all declare `name = "agentic_design_patterns"` but they are **independent uv environments**, not a single workspace — `cd` into the specific dir to install/run/lint.

TypeScript ports live under `<pattern>/typescript/` (currently for `1_prompt_chain`, `5_tool_use`, `6_planning` only). A single **bun workspace** with shared config lives in `typescript_base/` (contains `package.json`, `tsconfig.base.json`, `TYPESCRIPT.md`) and covers all pattern packages via the `../foundational_design_patterns/*/typescript` glob.

---

## 2. Running examples

### Python
```bash
cd <category>/<pattern>
uv sync                            # first time / after dep changes
bash run.sh                        # runs every script the pattern ships
uv run python src/<script>.py      # run one script directly
```

Some `run.sh` scripts are interactive menus (`9_rag`, `10_hitl`). For non-interactive use, prefer `uv run python` directly. `7_multi_agent_collaboration` needs a local `.env` with `TAVILY_API_KEY` in the pattern dir.

### TypeScript
```bash
cd typescript_base && bun install                            # one-time, installs all workspace members
cd ../foundational_design_patterns/<N>_<pattern>/typescript
bash run.sh                                                  # runs every TS demo for the pattern
bun run demo                                                 # single demo (one variant only)
bun test                                                     # offline smoke
```

Or skip the manual install — each `run.sh` auto-bootstraps `typescript_base/node_modules/` on first run.

### Smoke runners (cross-cutting)
- `scripts/run_demos_smoke.sh --mode basic|full` — Python
- `scripts/run_demos_smoke_typescript.sh --mode basic|full` — TS

Same flag shape, same PASS/FAIL/SKIP_INFRA classification, separate log dirs (`.demo-smoke-logs/`, `.demo-smoke-logs-ts/`).

---

## 3. Environment & secrets — **single .env at repo root**

The repo-root `.env` is the only `.env`. Both Python and TS read keys from it.

- Python: scripts use `find_dotenv()` (walks up to find root `.env`).
- TS: each `run.sh` does `set -a; . "$ROOT_DIR/.env"; set +a` before invoking bun.

**Do NOT create per-pattern `.env.example` files** — explicitly rejected (single source of truth at the root). If a new key is needed, document it in the pattern README; values go in the root `.env`.

Required keys:
- `OPENAI_API_KEY` — universal.
- `OPENAI_MODEL` — optional model override (default `gpt-4o-mini`).
- `OPENAI_ADVANCED_MODEL` — optional, used by `*_advanced` variants (default `gpt-5.2`, falls back to `OPENAI_MODEL`).
- `TAVILY_API_KEY` — only `7_multi_agent_collaboration`.
- `AGENTIC_DISABLE_SSL=1` — opt-in SSL bypass (corporate networks; see §4).

Python 3.11+ required (3.13 has known dep issues).

---

## 4. SSL bypass (opt-in)

`ssl_fix.py` at repo root used to monkey-patch SSL verification on import. Per P0 cleanup it's now **opt-in**:

```bash
export AGENTIC_DISABLE_SSL=1    # before any Python or bun command
```

The bypass disables cert verification globally and patches `httpx.Client` / `AsyncClient` to `verify=False`. Only enable in dev/corporate-proxy environments — never in production. The `run.sh` scripts also forward `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` (Python) or `NODE_EXTRA_CA_CERTS` / `SSL_CERT_FILE` (TS) where set.

---

## 5. Lockfiles are NOT committed

Both `uv.lock` (Python) and `bun.lock` (TS) are gitignored. This is a deliberate departure from the standard "commit lockfiles" advice. Trade-off accepted: transitive deps may drift across machines/CI, but the repo stays lighter and avoids lockfile churn in PRs.

**Why this matters when adding deps:** there's no lockfile to vendor against. Reproducibility relies on `uv sync` / `bun install` resolving the same versions, which is mostly stable for the version ranges in this repo but not guaranteed.

---

## 6. Lint / format / tests

- **Ruff** per pattern (`pyproject.toml`): `line-length = 120`, `target-version = "py39"`. Run from inside the pattern dir.
- **TypeScript**: `bun --bun tsc --noEmit` per package, `bun test` for smoke.
- No repo-wide lint command exists.

---

## 7. Conventions when adding or modifying code

### Python ↔ TypeScript parity is strict
When a pattern exists in both languages, the TS port must **mirror the Python verbatim**: same scenario, same example data, same identifiers, same structural primitives (LCEL `\|` ↔ `RunnableSequence.pipe()`, `ThreadPoolExecutor` ↔ `Promise.all`, `StateGraph` ↔ `StateGraph`). No deviations.

When refactoring scenarios or restructuring, hit **both languages in the same PR**. Cross-language drift defeats the purpose of dual-language coverage and was the source of every parity bug in this repo's history.

### Make the pattern's structure visible
In an educational catalog, the *form* of the code matters as much as the behavior. If the chapter is "Prompt Chaining", the reader should *see* a chain — not implicit `await await`. Default framework picks by lesson:
- Prompt chaining → LangChain.js LCEL (`prompt.pipe(llm).pipe(parser)`, `RunnableMap.from(...)`)
- State machines / planners → LangGraph.js `StateGraph` with `Annotation.Root`
- Tool-loop agents → LangChain v1 `createAgent` (mirrors Python `create_agent`)
- Parallel fan-out → `Promise.all` (mirrors `ThreadPoolExecutor.map`)

Vercel AI SDK is fine where the AI SDK idiom IS the lesson (streaming UI, agent loops with `stopWhen`), not when a structural framework would make the pattern visible.

### TypeScript uses Bun, not Node/pnpm/tsx/vitest
- Runtime: `bun run <file>.ts` (no build step)
- Package manager: `bun install`
- Test runner: `bun test` (built-in, Jest-like API: `describe`/`test`/`expect` from `bun:test`)
- Workspaces: native via `"workspaces"` in root `package.json`

Don't introduce vitest/tsx/pnpm in new code.

### Python pattern dirs intentionally duplicate boilerplate
Colors classes, env loading, etc. — do NOT refactor toward a shared utility package. Self-containment is the point; each pattern dir should be readable in isolation. (Exception: `repo_support.py` at repo root is the one allowed shared helper, used for env loading and model resolution.)

### Examples are intentionally verbose
Docstrings on functions, ANSI-colored terminal output, `print`/`console.log` showing intermediate steps. Clarity > cleverness. This is teaching code.

### Pi analysis docs use a canonical filename
Implementation-focused Pi cross-reference docs live in `pi.md` inside the relevant pattern directory. Do **not** create or reintroduce alternate filenames like `pi_codex.md`, `pi_codexi.md`, or other vendor-specific variants.

These docs must stay evidence-based:
- Ground claims in the actual Pi codebase.
- Prefer inline code excerpts with line numbers over fragile GitHub links.
- Call out architectural tradeoffs or limitations.
- If Pi does **not** meaningfully implement a pattern, say so explicitly instead of forcing a match.

Current pattern directories that support Pi analysis:
- `foundational_design_patterns/1_prompt_chain`
- `foundational_design_patterns/2_routing`
- `foundational_design_patterns/3_parallelization`
- `foundational_design_patterns/4_reflection`
- `foundational_design_patterns/5_tool_use`
- `foundational_design_patterns/6_planning`
- `foundational_design_patterns/7_multi_agent_collaboration`
- `foundational_design_patterns/8_react`
- `foundational_design_patterns/10_hitl`
- `foundational_design_patterns/11_structured_outputs`
- `foundational_design_patterns/12_computer_use`
- `memory/memory_management`
- `memory/context_management`
- `orchestration/subagents`
- `orchestration/skills`

### Pattern skeleton (for new patterns)
```
<pattern>/
├── pyproject.toml
├── run.sh
├── README.md
├── QUICK_START.md
├── src/
│   ├── <name>_basic.py
│   └── <name>_advanced.py        # optional
├── typescript/                     # optional TS port
│   ├── package.json
│   ├── tsconfig.json
│   ├── run.sh
│   ├── README.md
│   ├── src/
│   └── tests/
```

`CONTRIBUTING.md` has the full template.

---

## 8. User & collaboration model

User is the maintainer of `gtesei/agentic_design_patterns` (public, MIT). Senior dev. Direct, terse feedback style — when something looks wrong, says so plainly and expects it fixed. Comfortable with strong opinions in responses; make calls, don't punt every decision.

Apple Silicon Mac (M3 Pro arm64), corporate-network setup (hence the SSL bypass story).

Runs **two AI agents in parallel** on this repo: **OpenAI Codex (CLI)** and **Claude Code**. Work is split:

- **Codex owns**: P0 reliability/hygiene (the opt-in `ssl_fix.py`, `repo_support.py` shared helper, packaging cleanup, model-name typos, Python pattern source refreshes including the `5_tool_use` and `6_planning` variants).
- **Claude owns**: planning docs (`IMPROVEMENT_v2.md`, `SCENARIOS.md`), the TypeScript track (`TYPESCRIPT.md`, `scripts/run_demos_smoke_typescript.sh`, the `*/typescript/` packages), audits.

When merging outputs from both (e.g., `IMPROVEMENT_CODEX.md` + `IMPROVEMENT.md` → `IMPROVEMENT_v2.md`), treat Codex contributions as **complementary, not competing**. Surface real conflicts honestly; never silently overwrite Codex's choices. Check recent git log before touching Python source to avoid clobbering in-flight Codex work.

---

## 9. Project state (May 2026)

Multi-week "2026 Edition" refresh. Canonical roadmap: `IMPROVEMENT_v2.md`.

| Phase | Status |
|---|---|
| **P0** reliability/hygiene | done (Codex) — ssl_fix is opt-in, `repo_support.py` exists, model typos fixed |
| **P1** anchor scenarios | done — committed as `SCENARIOS.md` |
| **P2** stack modernization sweep | partial — some Python files refreshed |
| **P3** refresh top-5 stale patterns | in-flight — `5_tool_use` and `6_planning` on anchor scenarios; `9_rag`, `1_prompt_chain` still on legacy toys |
| **P4** new HIGH chapters (Structured Outputs, Subagents, Skills, Deep Research, Computer Use) | not started |
| **P5** cross-cutting evals/traces/tests | partial — smoke runners exist |
| **P6** TypeScript track | initial subset done (`1_prompt_chain`, `5_tool_use`, `6_planning`) — direct Python ports |
| **P7** restructure (merges/splits/renames) | not started |
| **P8** Voice, Managed-vs-Custom, etc. | deferred |

When the user asks "what's next?" or "plan and execute P<N>", `IMPROVEMENT_v2.md` is the source. Its TL;DR ordering is the recommendation, not gospel — the user has overridden before (bun instead of pnpm; anchor scenarios over modernization first).

---

## 10. Canonical docs index (always consult before deciding in their domain)

| File | Purpose |
|---|---|
| `IMPROVEMENT_v2.md` | Modernization roadmap, P0..P8, the contract between Claude and Codex |
| `SCENARIOS.md` | Anchor-scenario decision + pattern-to-scenario map. Read before any refresh PR |
| `TYPESCRIPT.md` | TS track overview (frameworks per pattern, bun conventions, how to add a TS pattern) |
| `CONTRIBUTING.md` | Pattern template, code style, Ruff config |
| `repo_support.py` | Shared Python helper: `find_dotenv`, `get_default_model`, `get_advanced_model`, `configure_example` |
| `scripts/run_demos_smoke.sh` | Python smoke runner |
| `scripts/run_demos_smoke_typescript.sh` | TS smoke runner |
| `typescript_base/` | Bun workspace root: `package.json`, `tsconfig.base.json`, `TYPESCRIPT.md` |

---

## 11. What NOT to do

- **Don't commit lockfiles** (`uv.lock`, `bun.lock` — both ignored).
- **Don't create per-pattern `.env.example` files** — root `.env` only.
- **Don't refactor Python pattern dirs toward a shared utility package** (beyond `repo_support.py`).
- **Don't drift TS away from Python.** If you'd port a feature, you'd port the same scenario, same identifiers, same constructions. Refactor both languages in one PR.
- **Don't silently overwrite Codex's in-flight work.** Check `git log` and the relative dates on source files first.
- **Don't add vitest/tsx/pnpm.** Bun only.
- **Don't use the AI SDK** for patterns where the structural form (chain, graph) is the lesson — use LangChain.js / LangGraph.js so the form is visible.
