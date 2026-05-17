# Pi — Skills (revised)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates findings from `pi.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, accessed 2026-05-15), and `pi_codex.md` (Codex agent read, accessed 2026-05-17). Every excerpt below was re-verified against a fresh `git clone` of the repo on 2026-05-17.

## Summary

**Yes — built-in, spec-conformant, at two layers, with both implicit and explicit invocation paths.** Skills exist at the framework level (`packages/agent/src/harness/skills.ts`) *and* at the coding-agent app level (`packages/coding-agent/src/core/skills.ts`). Both implement the open [Agent Skills](https://agentskills.io) spec: `SKILL.md` files with name/description/disable-model-invocation frontmatter, name validation (lowercase a-z, 0-9, hyphens; ≤ 64 chars; no leading/trailing/consecutive hyphens), 1024-char description cap.

The **progressive disclosure contract** is implemented operationally: only compact metadata (name + description + location) is rendered into the system prompt; the full body is loaded on demand via the `read` tool, *or* explicitly via the `/skill:name` slash command which expands into a structured `<skill>` block injected into the conversation.

Additional capabilities I missed in my first read but verified after merging notes: **collision detection** across multiple skill sources (with diagnostics), `disableModelInvocation` to keep a skill available by slash command but hidden from the auto-discovery prompt, and an `extendResources` API that lets extensions dynamically add skill paths at runtime.

> Addition to my earlier `pi.md`: I missed the framework-layer `harness/skills.ts` (which is the reusable skills primitive for non-coding-agent apps) and the `/skill:name` slash command path entirely. Both are merged below.

## Where it lives

| Concern | File |
|---|---|
| Framework-layer skill loading + `formatSkillInvocation` | `packages/agent/src/harness/skills.ts` |
| Framework-layer system-prompt assembly with `<available_skills>` | `packages/agent/src/harness/system-prompt.ts` |
| App-layer skill schema, validation, discovery | `packages/coding-agent/src/core/skills.ts` (500 lines) |
| `/skill:name` slash command expansion | `packages/coding-agent/src/core/agent-session.ts` (`_expandSkillCommand`) |
| Resource loader integration (CLI flags, extension paths) | `packages/coding-agent/src/core/resource-loader.ts` |
| Frontmatter parser | `packages/coding-agent/src/utils/frontmatter.ts` |
| User-facing docs | `packages/coding-agent/docs/skills.md` |

`SKILL.md` files themselves live outside the repo — typically in `~/.pi/skills/<skill-name>/SKILL.md` (user scope) or `<repo>/.pi/skills/<skill-name>/SKILL.md` (project scope).

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

**Why relevant:** The `Skill` shape matches the open spec exactly: `name`, `description`, optional `disable-model-invocation`. The constants `MAX_NAME_LENGTH = 64` and `MAX_DESCRIPTION_LENGTH = 1024` are taken verbatim from the spec. The framework `harness/skills.ts` uses identical constants — duplicated rather than shared, but consistent.

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

**Why relevant:** Pi enforces the spec at load time. Skills with invalid names are rejected and reported as diagnostics rather than silently included with garbage data.

### Lenient fallbacks where the spec allows

```ts
// packages/coding-agent/src/core/skills.ts (excerpted)
const name = frontmatter.name || parentDirName;
const nameErrors = validateName(name);
// ...
if (!frontmatter.description || frontmatter.description.trim() === "") {
  return { skill: null, diagnostics };
}
```

**Why relevant:** Pi falls back to the parent directory name if `frontmatter.name` is missing — a pragmatic shortcut so `~/.pi/skills/my-skill/SKILL.md` works even without an explicit `name:` field. Description, however, is **required** — a skill without a description is dropped entirely (still emits a diagnostic). The progressive-disclosure contract depends on description being present, so this is enforced.

### Discovery — recursive scan honoring ignore files, skipping `node_modules`

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

// ... in the recursive scan:
if (entry.name === "node_modules") {
  continue;
}
```

**Why relevant:** Two important behaviors:

1. **Skill roots don't recurse.** Once a directory has `SKILL.md`, that directory is treated as the skill's package — subdirs are bundled assets (referenceable from the skill body), not nested skills.
2. **`.gitignore` is honored** + `node_modules` is hardcoded-skipped. Important when scanning a project repo: build artifacts won't pollute the skill list.

### Framework-layer skill API

```ts
// packages/agent/src/harness/skills.ts
export async function loadSkills(
  env: ExecutionEnv,
  dirs: string | string[],
): Promise<{ skills: Skill[]; diagnostics: SkillDiagnostic[] }> {
  // ...
}

export function formatSkillInvocation(skill: Skill, additionalInstructions?: string): string {
  const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${dirnameEnvPath(skill.filePath)}.\n\n${skill.content}\n</skill>`;
  return additionalInstructions ? `${skillBlock}\n\n${additionalInstructions}` : skillBlock;
}
```

**Why relevant:** Skills are not just a coding-agent UI feature — they are a **reusable harness-level primitive** for any application built on `@earendil-works/pi-agent-core`. `loadSkills` is generic over an `ExecutionEnv`, and `formatSkillInvocation` is the standard way to inject a full skill body into a conversation as a `<skill>` block. The framework also has `loadSourcedSkills` for typed source tracking.

### System-prompt advertisement — compact metadata, not full bodies

```ts
// packages/coding-agent/src/core/skills.ts (formatSkillsForPrompt)
const lines = [
  "\n\nThe following skills provide specialized instructions for specific tasks.",
  "Use the read tool to load a skill's file when the task matches its description.",
  "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
  "",
  "<available_skills>",
];

for (const skill of visibleSkills) {
  lines.push("  <skill>");
  lines.push(`    <name>${escapeXml(skill.name)}</name>`);
  lines.push(`    <description>${escapeXml(skill.description)}</description>`);
  lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
  lines.push("  </skill>");
}
```

The framework harness has the same shape (`packages/agent/src/harness/system-prompt.ts:formatSkillsForSystemPrompt`).

**Why relevant:** This is the **progressive-disclosure contract** in concrete form. The model sees a compact XML index — name, description, location per skill — and is told to use the `read` tool to load the full body on demand. Full skill bodies never enter the system prompt automatically. Token-efficient: a project with 50 skills costs ~50 × ~150 tokens of metadata, not 50 × full-skill-body. The path-resolution instruction in the prompt prevents the model from getting confused about relative paths inside skill files.

### `/skill:name` — explicit, deterministic skill invocation

```ts
// packages/coding-agent/src/core/agent-session.ts
private _expandSkillCommand(text: string): string {
  if (!text.startsWith("/skill:")) return text;
  // ...
  const skill = this.resourceLoader.getSkills().skills.find((s) => s.name === skillName);
  // ...
  const content = readFileSync(skill.filePath, "utf-8");
  const body = stripFrontmatter(content).trim();
  const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${skill.baseDir}.\n\n${body}\n</skill>`;
  return args ? `${skillBlock}\n\n${args}` : skillBlock;
}
```

**Why relevant:** When the user types `/skill:my-skill some args`, Pi reads the skill file, strips frontmatter, and injects the full body as a `<skill>` block into the conversation **before** the user's args. This is the **deterministic path** — instead of advertising the skill and trusting the model to choose `read`, the user pins the skill down. Useful when you want to *force* a skill instead of *hint* at it. The `baseDir` annotation tells the model how to resolve relative paths inside the skill body.

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

**Why relevant:** A skill carries provenance metadata (user-scoped vs project-scoped vs ad-hoc path). Same security pattern as subagents: user-scoped skills are trusted by default; project-scoped skills (i.e., skills committed in a repo you cloned) are typically gated. The `SourceInfo` is also surfaced in the UI so users can audit "where did this skill come from?"

### Collision detection across multiple skill sources

```ts
// packages/coding-agent/src/core/skills.ts (excerpted)
const existing = skillMap.get(skill.name);
if (existing) {
  collisionDiagnostics.push({
    type: "collision",
    message: `name "${skill.name}" collision`,
    // ...
  });
} else {
  skillMap.set(skill.name, skill);
  realPathSet.add(realPath);
}
```

**Why relevant:** When skills are loaded from multiple paths (user dir + project dir + extension-provided paths), name collisions are detected during merge. The losing skill is shadowed by load-order precedence and the collision is surfaced as a diagnostic — not silently overwritten.

### Hidden-from-auto-invocation skills

```ts
// packages/coding-agent/src/core/skills.ts (excerpted)
disableModelInvocation: frontmatter["disable-model-invocation"] === true,
// ...
const visibleSkills = skills.filter((s) => !s.disableModelInvocation);
```

**Why relevant:** A skill can be **installed but invisible to the model** by setting `disable-model-invocation: true` in its frontmatter. The skill remains usable via `/skill:name` (the slash command bypasses the visibility filter and reads the file directly), but it's omitted from `<available_skills>` so it doesn't bias every prompt. Useful for skills that are expensive, sensitive, or that the user wants to invoke deliberately.

### Diagnostics — non-fatal validation errors flow to the user

```ts
// packages/coding-agent/src/core/skills.ts (function signature)
export interface LoadSkillsResult {
  skills: Skill[];
  diagnostics: ResourceDiagnostic[];
}
```

**Why relevant:** Loading is partial-failure-tolerant. Bad frontmatter is **skipped** (not silently included), the error is recorded as a `ResourceDiagnostic` returned alongside valid skills. The UI surfaces these so users can fix their skill files. The framework-layer counterpart uses `SkillDiagnostic` with stable codes (`file_info_failed`, `list_failed`, `read_failed`, `parse_failed`, `invalid_metadata`).

### Resource-loader integration — CLI + extension-provided paths

```ts
// packages/coding-agent/src/core/resource-loader.ts (excerpted)
const skillPaths = this.noSkills
  ? this.mergePaths(cliEnabledSkills, this.additionalSkillPaths)
  : this.mergePaths([...cliEnabledSkills, ...enabledSkills], this.additionalSkillPaths);

this.lastSkillPaths = skillPaths;
this.updateSkillsFromPaths(skillPaths, metadataByPath);
```

```ts
// packages/coding-agent/src/core/resource-loader.ts (extendResources)
extendResources(paths: ResourceExtensionPaths): void {
  const skillPaths = this.normalizeExtensionPaths(paths.skillPaths ?? []);
  // ...
  if (skillPaths.length > 0) {
    this.lastSkillPaths = this.mergePaths(
      this.lastSkillPaths,
      skillPaths.map((entry) => entry.path),
    );
    this.updateSkillsFromPaths(this.lastSkillPaths);
  }
}
```

**Why relevant:** Skills aren't just "scan two hardcoded directories." The resource loader composes paths from: CLI flags (`--enable-skill`), user/project enabled paths, extension-provided paths via `extendResources`, and `--no-skills` to skip the defaults entirely. Extensions can dynamically add skill paths at runtime — useful for plug-in systems that bundle their own skill packs.

## Tradeoffs and limitations

- **Spec-conformant but spec-bounded.** Pi tracks the Agent Skills spec faithfully. New frontmatter fields would require code changes. The `[key: string]: unknown` index signature on `SkillFrontmatter` permits extra fields at parse time but they're ignored at runtime.
- **Two implementations, duplicated.** The framework `harness/skills.ts` and the app `coding-agent/src/core/skills.ts` both implement loading and validation. Likely intentional (framework primitive vs app composition) but a drift hazard — a spec change requires updates in two places.
- **`/skill:name` reads the file fresh, not from the resource loader's cache.** Each invocation does a synchronous `readFileSync` — fine in practice (skill files are small), but means an in-place edit to the file is picked up by the next slash command without a session reload.
- **Skills are markdown, not code.** Bodies are natural-language instruction sets. Skills that need executable behavior typically reference scripts (`run-lint.sh`) that the agent invokes via its `bash` tool. The spec's intent — skills as *playbooks*, not *plugins*.
- **No skill composition primitives.** A skill body can reference others by name in prose ("after this, use the `convert-csv` skill") but no first-class dependency relationship.
- **No registry / network distribution.** Skills are loaded from disk only. No `pi skills install <url>` or marketplace.
- **No skill versioning.** A skill's `SKILL.md` is whatever's on disk. If a project ships a different version of a same-named skill, load order decides which wins — no semver resolution.
- **Auto-invocation depends on the model deciding to `read` the file.** Pi advertises the skill compactly but does not force execution unless the user calls `/skill:name`.
- **Collisions are detected but resolution is precedence-only.** The losing skill is shadowed silently from the model's POV (only the diagnostic surfaces this); the user has to notice the warning to know.

## "Not implemented" caveats

- ❌ Skills registry / distribution mechanism
- ❌ Skill versioning / dependency resolution
- ❌ First-class skill composition (skills depending on / extending other skills)
- ❌ Non-markdown skill formats (code-as-skill)
- ❌ Skill marketplace UI

What Pi does ship that the Skills pattern asks for:

- ✅ `SKILL.md` file convention with YAML frontmatter (name / description / disable-model-invocation)
- ✅ Spec-conformant validation (name regex, length limits)
- ✅ Recursive discovery with `.gitignore` honor + `node_modules` skip
- ✅ User-scope vs project-scope provenance tracking
- ✅ Partial-failure-tolerant loading with surfaced diagnostics
- ✅ Progressive disclosure: compact `<available_skills>` index in system prompt, full body loaded on demand
- ✅ Both **implicit** (model decides to `read`) and **explicit** (`/skill:name` slash command) invocation paths
- ✅ `disableModelInvocation` flag — installed but hidden from auto-discovery, still callable via slash command
- ✅ Collision detection across multiple skill sources with diagnostics
- ✅ Extension hook for dynamically adding skill paths (`extendResources`)
- ✅ Framework-layer skill primitive (`harness/skills.ts`) reusable by non-coding-agent apps
- ✅ Lenient fallback (parent dir name as `name` when frontmatter omits it) while keeping description required
