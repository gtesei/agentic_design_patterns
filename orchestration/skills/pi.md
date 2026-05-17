# Pi — Skills

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17

## Summary

**Yes — built-in, spec-conformant.** Pi implements the open [Agent Skills](https://agentskills.io) spec as a first-class feature in `coding-agent`. Skills are discovered from `SKILL.md` files (one per skill, in their own directory) via `loadSkillsFromDir`. Each skill has frontmatter-constrained metadata (`name`, `description`, optional `disable-model-invocation`) and a markdown body. Names are validated against the spec (lowercase a-z, 0-9, hyphens; ≤ 64 chars; no leading/trailing/consecutive hyphens). Descriptions are length-capped at 1024 chars.

The spec contract — *progressive disclosure*: metadata always loaded, body loaded on demand when invoked — is the architectural point of the pattern. Pi implements the loading side cleanly; the invocation/loading-on-demand machinery is wired through the coding-agent runtime that builds the system prompt from skill metadata.

Pi's skills implementation is **distinct from its subagents**. Skills are static capability descriptors loaded into the LLM's prompt as tool-like entries; subagents are runtime delegations to a separate `pi` process. Both use a `.md` + YAML-frontmatter convention but they sit at different layers.

## Where it lives

| Concern | File |
|---|---|
| Skill schema, frontmatter parse, validation, loader | `packages/coding-agent/src/core/skills.ts` (500 lines) |
| Frontmatter parser used by skills + agents | `packages/coding-agent/src/utils/frontmatter.ts` |
| Skill diagnostics surfaced to the user | `packages/coding-agent/src/core/diagnostics.ts` |
| Skill source-info (where it came from on disk) | `packages/coding-agent/src/core/source-info.ts` |

`SKILL.md` files themselves live outside the repo — typically in `~/.pi/skills/<skill-name>/SKILL.md` or `<repo>/.pi/skills/<skill-name>/SKILL.md` (project scope).

## Key code excerpts

### Schema + validation per the Agent Skills spec

```ts
// packages/coding-agent/src/core/skills.ts
/** Max name length per spec */
const MAX_NAME_LENGTH = 64;

/** Max description length per spec */
const MAX_DESCRIPTION_LENGTH = 1024;

export interface SkillFrontmatter {
  name?: string;
  description?: string;
  "disable-model-invocation"?: boolean;
  [key: string]: unknown;
}

export interface Skill {
  name: string;
  description: string;
  filePath: string;
  baseDir: string;
  sourceInfo: SourceInfo;
  disableModelInvocation: boolean;
}
```

**Why relevant:** The `Skill` shape matches the open spec exactly: `name`, `description`, optional `disable-model-invocation`. The constants `MAX_NAME_LENGTH = 64` and `MAX_DESCRIPTION_LENGTH = 1024` are taken verbatim from the spec. No proprietary extensions.

### Name validation — strict regex matching the spec

```ts
// packages/coding-agent/src/core/skills.ts
function validateName(name: string): string[] {
  const errors: string[] = [];

  if (name.length > MAX_NAME_LENGTH) {
    errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
  }

  if (!/^[a-z0-9-]+$/.test(name)) {
    errors.push(`name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)`);
  }

  if (name.startsWith("-") || name.endsWith("-")) {
    errors.push(`name must not start or end with a hyphen`);
  }

  if (name.includes("--")) {
    errors.push(`name must not contain consecutive hyphens`);
  }

  return errors;
}
```

**Why relevant:** Pi enforces the spec at load time. Skills with invalid names are rejected and reported as diagnostics rather than silently included. This protects against name collisions and ensures interoperability with other Agent Skills consumers.

### Description validation — required, capped

```ts
// packages/coding-agent/src/core/skills.ts
function validateDescription(description: string | undefined): string[] {
  const errors: string[] = [];

  if (!description || description.trim() === "") {
    errors.push("description is required");
  } else if (description.length > MAX_DESCRIPTION_LENGTH) {
    errors.push(`description exceeds ${MAX_DESCRIPTION_LENGTH} characters (${description.length})`);
  }

  return errors;
}
```

**Why relevant:** Description is what the LLM sees in its always-loaded metadata index — that's the *progressive disclosure* contract. Capping at 1024 chars keeps the metadata index compact; requiring non-empty descriptions enforces that every skill explains itself.

### Discovery — recursive scan honoring `.gitignore` / `.ignore` / `.fdignore`

```ts
// packages/coding-agent/src/core/skills.ts
const IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"];

/**
 * Load skills from a directory.
 *
 * Discovery rules:
 * - if a directory contains SKILL.md, treat it as a skill root and do not recurse further
 * - otherwise, load direct .md children in the root
 * - recurse into subdirectories to find SKILL.md
 */
export function loadSkillsFromDir(options: LoadSkillsFromDirOptions): LoadSkillsResult {
  const { dir, source } = options;
  return loadSkillsFromDirInternal(dir, source, true);
}
```

**Why relevant:** Two important behaviors documented in the comment:

1. **Skill roots don't recurse** — once a directory has `SKILL.md`, that directory is treated as the skill's package. Subdirs are bundled assets (referenceable from the skill body), not nested skills.
2. **`.gitignore` is honored** — skills under ignored paths are skipped. This matters when scanning a project repo for skills: build artifacts, `node_modules`, etc. won't be scanned.

The `addIgnoreRules` helper walks the tree applying ignore files at each directory level, prefixing patterns appropriately — equivalent to how `ripgrep` and `fd` resolve nested ignore files.

### Source-info — every skill knows where it came from

```ts
// packages/coding-agent/src/core/skills.ts
function createSkillSourceInfo(filePath: string, baseDir: string, source: string): SourceInfo {
  switch (source) {
    case "user":
      return createSyntheticSourceInfo(filePath, { source: "local", scope: "user", baseDir });
    case "project":
      return createSyntheticSourceInfo(filePath, { source: "local", scope: "project", baseDir });
    case "path":
      return createSyntheticSourceInfo(filePath, { source: "local", baseDir });
    default:
      return createSyntheticSourceInfo(filePath, { source, baseDir });
  }
}
```

**Why relevant:** A skill carries provenance metadata (user-scoped vs project-scoped vs ad-hoc path). Same security pattern as subagents: user-scoped skills are trusted by default; project-scoped skills (i.e., skills committed in a repo you cloned) are typically gated by an opt-in or confirmation flow in higher-level coding-agent code. The `SourceInfo` is also surfaced in the UI so users can audit "where did this skill come from?"

### Diagnostics — non-fatal validation errors flow to the user

```ts
// packages/coding-agent/src/core/skills.ts (function signature)
export interface LoadSkillsResult {
  skills: Skill[];
  diagnostics: ResourceDiagnostic[];
}
```

**Why relevant:** Loading is partial-failure-tolerant. A skill with invalid frontmatter is **skipped** (not silently included with garbage data), and the error is recorded as a `ResourceDiagnostic` returned alongside the valid skills. The UI surfaces these so users can fix their skill files.

## Tradeoffs and limitations

- **Spec-conformant but spec-bounded.** Pi tracks the Agent Skills spec faithfully. If the spec adds capabilities (e.g., new frontmatter fields), Pi has to follow. If you want non-standard skill metadata, the `[key: string]: unknown` index signature on `SkillFrontmatter` permits extras at parse time but they're ignored by Pi's runtime.
- **`disable-model-invocation` exists but is not deeply integrated.** The flag is loaded and persisted on `Skill.disableModelInvocation` but the canonical interpretation (a skill that loads on demand based on a separate trigger, not the LLM choosing it) depends on how the consuming app wires the skill into its tool surface. Pi's coding-agent uses the standard "expose all skills as tool-like entries in the system prompt" path.
- **Skills are markdown, not code.** The body of a skill is a natural-language instruction set. Skills that need executable behavior typically reference scripts (`run-lint.sh`) that the agent invokes via its `bash` tool, rather than carrying code directly. This is the spec's intent — skills are *playbooks*, not *plugins*.
- **No skill composition primitives.** A skill body can reference other skills by name in prose ("after running this, use the `convert-csv` skill") but there is no first-class "this skill depends on / extends that skill" relationship. Pure progressive-disclosure: the LLM sees descriptions of all skills and chooses.
- **No registry / network distribution.** Skills are loaded from disk only. There is no `pi skills install <url>` or skills marketplace in the repo. Distribution is user-responsibility (git clone the skill, drop it under `~/.pi/skills/`).
- **No skill versioning.** A skill's `SKILL.md` is whatever's on disk. If a project ships a different version of the same-named skill, the loading order (`user` then `project`, or vice versa) decides which wins — there is no semver / range resolution.

## "Not implemented" caveats

- ❌ Skills registry / distribution mechanism
- ❌ Skill versioning / dependency resolution
- ❌ First-class skill composition (skills calling skills)
- ❌ Non-markdown skill formats (code-as-skill)
- ❌ Skill marketplace UI

What Pi does ship that the Skills pattern asks for:

- ✅ `SKILL.md` file convention with YAML frontmatter (name / description / disable-model-invocation)
- ✅ Spec-conformant validation (name regex, length limits)
- ✅ Recursive discovery with `.gitignore` honor
- ✅ User-scope vs project-scope provenance tracking
- ✅ Partial-failure-tolerant loading with surfaced diagnostics
- ✅ Progressive disclosure: metadata always loaded, full body referenced on demand
- ✅ Distinct from subagents (different layer, different lifecycle)
