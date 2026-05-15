# SCENARIOS.md

This document defines the **anchor scenarios** used across every pattern in this repository. Each example draws from this set instead of inventing a fresh toy. The intent: a reader who already knows "Aurora Telecom support tickets" can compare routing vs reflection vs HITL vs guardrails without re-loading new domain framing for each pattern. The *pattern* becomes the variable.

This file is the single source of truth. If you are authoring or refreshing an example, look up your pattern in the table below, use the fixtures in `tooling/scenarios/<scenario>/`, and add a one-line "Anchor scenario:" pointer to your pattern README. If a pattern doesn't fit cleanly, open a PR — don't reach for a fresh toy.

---

## The four anchors + one sidecar

| # | Scenario | Setting | Why it earns a slot |
|---|---|---|---|
| 1 | **Support Ops** | Tier-1 customer support agent at "Aurora Telecom" (telecom co.) — triage, refunds, escalations | Universally relatable; natural fit for approval flows, PII, KB lookup, structured extraction |
| 2 | **Coding Agent** | Maintenance agent for a small Python project, "datapipe" — PR review, test repair, dep upgrades | Universally relatable for our reader (devs); natural fit for tool use, ReAct, planning, context engineering |
| 3 | **Deep Research** | Research agent at "Helix Labs" — multi-source synthesis with citations | Natural map-reduce shape (fan-out → read → synthesize); the canonical orchestrator-worker / subagent demo |
| 4 | **Incident Response** | On-call SRE agent at "Aurora Telecom" — alerts, runbooks, cross-team coordination | Covers prioritization, hierarchical goals, A2A, error recovery — patterns that otherwise lack a clear narrative |
| 5 | **Browser Ops** *(sidecar)* | Legacy back-office portal with no API | Used **only** for `12_computer_use`; no other pattern needs it |

**Consolidation note**: Support Ops and Incident Response share a fictional company ("Aurora Telecom") on purpose. The customer accounts in support tickets are the same accounts whose services page the on-call agent. This lets later examples cross-reference (e.g., a support ticket from a customer affected by an outage), and lets fixtures be shared.

We deliberately **dropped** two candidates from the original six:
- **Compliance / risk review** — overlaps too much with Support Ops + Guardrails. The patterns it would anchor are already covered.
- **Browser ops as a top-tier anchor** — promoted to sidecar. Only one pattern (`12_computer_use`) genuinely needs it.

---

## Pattern → scenario map

| Pattern | Anchor | The example does |
|---|---|---|
| `1_prompt_chain` | Support Ops | Extract complaint fields from a customer email → transform into ticket schema |
| `2_routing` | Support Ops | Classify ticket: billing / technical / cancellation / abuse / unclear |
| `3_parallelization` | Deep Research | Fan out 3 parallel sub-queries → merge results |
| `4_reflection` | Coding Agent | Draft PR review → critique → refine *(replaces blog-post polish)* |
| `5_tool_use` | Coding Agent | Tools: `read_file`, `grep_repo`, `run_pytest`, `git_diff` *(replaces weather mock)* |
| `6_planning` | Incident Response | Plan-and-Act for an outage: triage → diagnose → mitigate → postmortem *(replaces sunglasses inventory)* |
| `7_multi_agent_collaboration` (peer) | Deep Research | Researcher + Writer + Editor team (framed as legacy/peer topology) |
| `8_react` | Coding Agent | Debug a failing test via Thought-Action-Observation |
| `9_rag` (basic) | Support Ops | KB lookup for refund policy questions |
| `9_rag` (advanced/agentic) | Deep Research | Citation-first answers with query rewrite + grader + web fallback |
| `10_hitl` | Support Ops | Human approval for refunds > $500 |
| `11_structured_outputs` *(new)* | Support Ops | Pydantic `Ticket` schema; extract fields from raw email with Instructor retries |
| `12_computer_use` *(new)* | **Browser Ops** | Operate a legacy back-office portal (no API) for ticket lookup |
| `tree_of_thoughts` | Coding Agent | Explore multiple debug hypotheses for a flaky test *(verifiable: test passes or doesn't)* |
| `graph_of_thoughts` | Incident Response | Multi-perspective outage analysis: technical / customer-impact / business |
| `exploration_discovery` | *(merged into `deep_research`)* | — |
| `deep_research` *(new)* | Deep Research | Plan → search → read → reflect → cited synthesis |
| `error_recovery` | Incident Response | Retries + fallback when a runbook step fails mid-incident |
| `guardrails` | Support Ops | PII detection + policy enforcement on agent responses |
| `goal_management` | Incident Response | Hierarchical decomposition: incident goal → contain / communicate / remediate |
| `agent_communication` (A2A) | Incident Response | Oncall agent ↔ downstream service-owner agent across teams |
| `mcp` | Coding Agent | MCP server exposing repo tools (file ops, git, tests) over stdio |
| `subagents` *(new)* | Deep Research | Lead agent spawns 3 isolated research subagents in parallel |
| `skills` *(new)* | Coding Agent | `SKILL.md` files: `convert-csv`, `run-lint`, `generate-docs` |
| `prioritization` | Incident Response | Score & rank incoming alerts by severity × business impact |
| `evaluation_monitoring` | Support Ops | RAGAS + DeepEval on the support chatbot; LangSmith traces |
| `resource_optimization` | Support Ops | Prompt caching + semantic caching + model cascade on high-volume chatbot |
| `memory_management` | Support Ops | Customer-preference memory across sessions |
| `context_engineering` *(renamed from `context_management`)* | Coding Agent | Compaction on a large-codebase task |
| `adaptive_learning` | Support Ops | Agent learns from human approve/reject signals |

**Distribution**: Support Ops 11 · Coding Agent 7 · Deep Research 6 · Incident Response 6 · Browser Ops 1. Heavy on Support Ops and Coding, which is intentional — those are the most relatable to the reader.

---

## Per-scenario detail

Authors: read the section for your scenario, use the fixtures, don't invent new ones.

### 1. Support Ops — "Aurora Telecom"

**Storyline.** Aurora Telecom is a mid-sized telco. Their AI agent handles tier-1 customer tickets: billing questions, plan changes, refund requests, technical complaints. Higher-risk actions (refunds > $500, plan downgrades, account closures) require human approval. Volume is high; cost matters.

**Fixtures** (`tooling/scenarios/support_ops/`):
- `tickets.json` — 20 sample inbound tickets across 5 categories
- `kb/` — markdown KB articles (refund policy, plan tiers, abuse policy, SLA terms)
- `customers.json` — 10 mock customer records with plan, tenure, account balance
- Mock tools: `lookup_customer(id)`, `process_refund(ticket_id, amount)`, `update_plan(customer_id, plan)`, `escalate(ticket_id, reason)`

**Canonical happy path.** Customer emails about a $200 overcharge → agent classifies as `billing` → looks up customer + KB policy → drafts refund → guardrails check (no PII leakage, refund within policy) → routes to human if > $500 → executes refund → confirms to customer.

**Patterns using this scenario**: prompt_chain, routing, rag (basic), hitl, structured_outputs, guardrails, memory_management, adaptive_learning, evaluation_monitoring, resource_optimization.

### 2. Coding Agent — "datapipe"

**Storyline.** `datapipe` is a small open-source Python data-pipeline library (mocked: ~30 files, ~2000 LOC, ~80% test coverage, pinned deps). The agent maintains it: reviews incoming PRs, fixes failing tests, upgrades pinned deps when CVEs land, adds docstrings.

**Fixtures** (`tooling/scenarios/coding_agent/`):
- `datapipe_repo/` — a small mock repo (real files, real `pytest`-runnable)
- `tasks.json` — 5 canonical tasks: "fix failing test after pandas 3 upgrade", "review this PR", "add docstrings to module X", "upgrade requests to v3", "find unused imports"
- MCP server config exposing `read_file`, `write_file`, `grep`, `run_pytest`, `git_diff`, `git_apply`

**Canonical happy path.** Task: "tests fail after pandas 3 upgrade" → agent runs `pytest`, reads failure → greps for the failing symbol → reads relevant file → forms hypothesis (deprecated API) → patches → re-runs tests → opens PR with diff and explanation.

**Patterns using this scenario**: reflection, tool_use, react, tree_of_thoughts, mcp, skills, context_engineering.

### 3. Deep Research — "Helix Labs"

**Storyline.** Helix Labs is a small research org. The agent helps researchers produce technical briefs: fans out queries, reads top results from a curated corpus + web, identifies gaps, iterates, returns a synthesis with citations.

**Fixtures** (`tooling/scenarios/deep_research/`):
- `corpus/` — 50 markdown summaries on 3 topics (LLM agents, retrieval techniques, agent evaluation), each ~1k tokens with a stable URL frontmatter
- `web_search_mock.py` — deterministic mock returning seeded results so examples are reproducible
- `questions.json` — 3 canonical questions: "what changed in MCP between Q1 and Q4 2025?", "compare hybrid retrieval to dense-only in 2026", "state of agent evals beyond LLM-as-judge?"

**Canonical happy path.** Question → planner emits 3 sub-queries → parallel search across corpus + web mock → top hits read → "what's still missing?" → 2 follow-up queries → synthesized brief with inline citations.

**Patterns using this scenario**: parallelization, multi_agent_collaboration (peer), rag (advanced), exploration_discovery (merged), deep_research (new), subagents.

### 4. Incident Response — "Aurora Telecom SRE"

**Storyline.** Same company as Support Ops, different team. The agent supports the on-call SRE: prioritizes paging traffic, runs runbooks, coordinates with downstream service-owner agents via A2A, writes incident timelines.

**Fixtures** (`tooling/scenarios/incident_response/`):
- `runbooks/` — 5 markdown runbooks: `db-slow.md`, `cache-miss-spike.md`, `region-failover.md`, `ssl-expiry.md`, `cdn-purge.md`
- `alerts.json` — 15 sample alerts (severity, service, metric, timestamp)
- `services.json` — service map with team ownership (so A2A examples have real targets)
- Mock tools: `query_metric(name, range)`, `kick_runbook(name)`, `page_owner(team)`, `post_status(message)`

**Canonical happy path.** P2 alert: `api.latency_p99 > 2s` → agent prioritizes among 3 concurrent alerts → reads `db-slow.md` runbook → queries DB metrics → diagnoses (long-running query) → applies mitigation → A2A-notifies the data-team agent → writes incident timeline.

**Patterns using this scenario**: planning, graph_of_thoughts, error_recovery, goal_management, agent_communication (A2A), prioritization.

### 5. Browser Ops — "legacy back-office portal" *(sidecar)*

**Used only for `12_computer_use`.**

**Fixtures** (`tooling/scenarios/browser_ops/`):
- `portal/` — a minimal Flask app rendering a fake ticket-lookup UI with realistic friction (multi-step form, modal popups, no API endpoints)
- Runs locally on `localhost:5005`; example uses Anthropic's computer-use tool against a sandboxed browser

**Canonical happy path.** Agent gets ticket ID → navigates portal → logs in → searches → opens ticket → extracts the ticket fields back into structured JSON.

---

## Cross-scenario rules

These keep the catalog coherent.

1. **One scenario per example.** No mixing. If two could fit, pick one and stay there.
2. **Reuse fixtures.** Never invent new mock data if `tooling/scenarios/<scenario>/` already has it. Add to the shared fixture and update this file rather than baking data into your example.
3. **Customer / repo / service names stay consistent.** A customer in `1_prompt_chain` exists in `customers.json` and works in `2_routing` and `9_rag` too. Same for `datapipe` files and Aurora service names.
4. **Default model `gpt-4o-mini`, configurable via `MODEL` env var.** Document model swaps to Claude / Gemini in each README.
5. **No real API keys for the happy path.** Mocks serve from JSON. Optional `--live` toggles are fine but the example must run offline by default.
6. **Each pattern README opens with**: `Anchor scenario: <name>. See [SCENARIOS.md](../../SCENARIOS.md) for shared fixtures.`

---

## What replaces what (audit-trail for refresh PRs)

| Current toy | Anchor replacement |
|---|---|
| Weather lookup *(in tool_use, others)* | Coding-agent file ops *or* Support-Ops customer lookup |
| Blog-post polishing *(in reflection)* | Coding-agent PR review |
| Sunglasses inventory *(in planning)* | Incident-response runbook execution |
| Factorial / Game of 24 / arithmetic *(in tree_of_thoughts and others)* | Coding-agent debug exploration (test-pass is the verifiable state) |
| Research-report mocks with Tavily/arXiv *(in multi_agent)* | Deep-Research with the local corpus + deterministic web mock |
| Generic "stock market" tool demo *(in tool_use)* | Coding-agent repo tools |

Keep a toy domain *only* when the algorithm being taught (e.g., ToT search) depends on having a verifiable intermediate state and no anchor scenario provides one cheaply. In all other cases, refresh to the anchor.

---

## What this changes in practice

When you start a refresh PR:

1. Look up the pattern → find its anchor → read the per-scenario detail.
2. Open `tooling/scenarios/<scenario>/` — your fixtures are there.
3. Build the example using those fixtures. The pattern code is the only thing that varies.
4. Add the "Anchor scenario:" pointer to your README.
5. Delete the toy you replaced; don't leave it as a fallback.

This is what makes the next 6 weeks of refresh work tractable: every author opens this file, picks up the same fictional world, and ships an example that composes with everyone else's.
