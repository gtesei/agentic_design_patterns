# Pi implementation notes — Subagents

**Accessed on:** 2026-05-15  
**Scope:** `earendil-works/pi` (inspected local clone)

## 1) Implementation status
- **Not implemented as first-class core subsystem.**
- **Implemented as extension pattern / example.**

## 2) Core evidence (explicit non-core stance)

**Source:** `packages/coding-agent/README.md:474,478`

> “**No sub-agents.** ... build your own with extensions, or install a package.”

**Source:** `packages/coding-agent/docs/usage.md:275`

> “It intentionally does not include built-in ... sub-agents ...”

Why this matters: architecture choice is explicit; subagents are intentionally outside core runtime primitives.

## 3) What exists in code: extension-based implementation

### A. Subagent example uses subprocess isolation
**Source:** `packages/coding-agent/examples/extensions/subagent/index.ts:15`

```ts
import { spawn } from "node:child_process";
```

(also described in file header comments around lines 1-12)

Why this matters: delegated agents are external Pi processes, not internal scheduler entities.

### B. Subagent capability is exposed as a custom tool
**Source:** `packages/coding-agent/examples/extensions/subagent/index.ts:432-433`

```ts
pi.registerTool({
  name: "subagent",
  ...
});
```

Why this matters: orchestration is done via extension tool APIs.

### C. Operational modes are extension-level logic
**Source:** `packages/coding-agent/examples/extensions/subagent/index.ts:449-452,463,469`

- tracks `hasChain` / `hasTasks` / `hasSingle`
- computes `modeCount`
- rejects invalid calls with “Provide exactly one mode”

Why this matters: orchestration strategy is customizable and not standardized by core.

## 4) Tradeoffs
- **Pros:** maximum flexibility; teams can implement domain-specific orchestration.
- **Cons:** no core guarantees for behavior/UX; extension authors own process limits, failure handling, and safety.

## 5) Pattern mapping
For this repo’s `subagents` chapter, the accurate Pi mapping is:
- **“extension-supported pattern”**, not
- **“built-in core pattern.”**