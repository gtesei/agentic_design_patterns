# Pi implementation notes — Skills

**Accessed on:** 2026-05-15  
**Scope:** `earendil-works/pi` (inspected local clone)

## 1) Implementation status
- **Implemented in core:** discovery, validation, collision handling, prompt formatting, and explicit `/skill:name` expansion.

## 2) Code evidence (with line refs)

### A. Discovery and loading pipeline
**Source:** `packages/coding-agent/src/core/skills.ts:169,174,229-230`

```ts
export function loadSkillsFromDir(options: LoadSkillsFromDirOptions): LoadSkillsResult {
  const { dir, source } = options;
  return loadSkillsFromDirInternal(dir, source, true);
}
```

`loadSkillsFromDirInternal(...)` includes recursive `SKILL.md` discovery plus ignore-file handling; it also skips `node_modules`.

Why this matters: this is concrete filesystem discovery behavior, not docs-only guidance.

### B. Validation policy is strict-enough but practical
**Source:** `packages/coding-agent/src/core/skills.ts:93,118,122`

```ts
function validateName(name: string): string[] { ... }
function validateDescription(description: string | undefined): string[] { ... }

if (!frontmatter.description || frontmatter.description.trim() === "") {
  return { skill: null, diagnostics };
}
```

Why this matters: invalid metadata emits diagnostics; missing description blocks skill loading.

### C. Prompt exposure for model-side matching
**Source:** `packages/coding-agent/src/core/skills.ts:336,348`

```ts
export function formatSkillsForPrompt(skills: Skill[]): string {
  const visibleSkills = skills.filter((s) => !s.disableModelInvocation);
  ...
  lines.push("<available_skills>");
  ...
}
```

Why this matters: available skills are serialized into system prompt context in a structured form.

### D. Explicit skill invocation via command expansion
**Source:** `packages/coding-agent/src/core/agent-session.ts:1147-1148,1160`

```ts
private _expandSkillCommand(text: string): string {
  if (!text.startsWith("/skill:")) return text;
  ...
  const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${skill.baseDir}.\n\n${body}\n</skill>`;
  return args ? `${skillBlock}\n\n${args}` : skillBlock;
}
```

Why this matters: `/skill:name` is transformed into grounded, source-path-aware instruction content.

## 3) Tradeoffs / limitations
- Pi is intentionally somewhat lenient vs strict Agent Skills interpretation (documented in README/docs).
- Auto-selection still depends on model behavior; explicit `/skill:name` is the deterministic path.
- Security remains a concern for third-party skills (instruction and execution risk).

## 4) Pattern mapping
Pi skills are a **real built-in orchestration primitive** with clear source-level implementation and runtime behavior.