# Pi implementation notes: skills

Accessed on: 2026-05-17

Verdict: Skills are a first-class pattern in Pi. They are implemented at both the lower-level harness layer (`packages/agent`) and the user-facing coding-agent layer (`packages/coding-agent`), with discovery, validation, collision handling, prompt integration, and explicit `/skill:name` invocation support.

## Relevant Pi code

- `packages/agent/src/harness/skills.ts`
- `packages/agent/src/harness/system-prompt.ts`
- `packages/coding-agent/src/core/skills.ts`
- `packages/coding-agent/src/core/resource-loader.ts`
- `packages/coding-agent/src/core/agent-session.ts`
- `packages/coding-agent/docs/skills.md`

## 1. Pi discovers skills from `SKILL.md` files and validates them

The coding-agent loader walks directories, stops at `SKILL.md` roots, respects ignore files, skips `node_modules`, and parses each skill's frontmatter:

Source: `packages/coding-agent/src/core/skills.ts`

```ts
for (const entry of entries) {
	if (entry.name !== "SKILL.md") {
		continue;
	}
	...
	const result = loadSkillFromFile(fullPath, source);
	if (result.skill) {
		skills.push(result.skill);
	}
	diagnostics.push(...result.diagnostics);
	return { skills, diagnostics };
}
...
if (entry.name === "node_modules") {
	continue;
}
```

Skill metadata is validated but still loaded leniently when possible:

Source: `packages/coding-agent/src/core/skills.ts`

```ts
const descErrors = validateDescription(frontmatter.description);
...
const name = frontmatter.name || parentDirName;
const nameErrors = validateName(name);
...
if (!frontmatter.description || frontmatter.description.trim() === "") {
	return { skill: null, diagnostics };
}
```

Why this matters: Pi treats skills as structured resources with loading rules and diagnostics, not as arbitrary markdown snippets sprinkled into prompts.

## 2. The lower-level harness has a general skill API

At the lower level, the agent harness exposes generic skill loading and explicit invocation formatting:

Source: `packages/agent/src/harness/skills.ts`

```ts
export async function loadSkills(
	env: ExecutionEnv,
	dirs: string | string[],
): Promise<{ skills: Skill[]; diagnostics: SkillDiagnostic[] }> {
	...
}

export function formatSkillInvocation(skill: Skill, additionalInstructions?: string): string {
	const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${dirnameEnvPath(skill.filePath)}.\n\n${skill.content}\n</skill>`;
	return additionalInstructions ? `${skillBlock}\n\n${additionalInstructions}` : skillBlock;
}
```

Why this matters: skills are not just a coding-agent UI feature. They exist as a reusable harness-level concept for other Pi-based applications.

## 3. Pi advertises skills compactly in the system prompt

Pi does not dump full skill bodies into every prompt. Instead, it tells the model what skills exist and where to read them:

Source: `packages/coding-agent/src/core/skills.ts`

```ts
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

The lower-level harness uses the same shape:

Source: `packages/agent/src/harness/system-prompt.ts`

```ts
const lines = [
	"The following skills provide specialized instructions for specific tasks.",
	"Read the full skill file when the task matches its description.",
	...
	"<available_skills>",
];
```

Why this matters: Pi's skill pattern is token-aware. Skills are model-visible enough to be discoverable, but the full instructions are loaded on demand.

## 4. Explicit `/skill:name` commands expand into a structured skill block

When the user invokes a skill directly, Pi reads the file and injects the full body into the conversation as a `<skill ...>` block:

Source: `packages/coding-agent/src/core/agent-session.ts`

```ts
private _expandSkillCommand(text: string): string {
	if (!text.startsWith("/skill:")) return text;
	...
	const skill = this.resourceLoader.getSkills().skills.find((s) => s.name === skillName);
	...
	const content = readFileSync(skill.filePath, "utf-8");
	const body = stripFrontmatter(content).trim();
	const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${skill.baseDir}.\n\n${body}\n</skill>`;
	return args ? `${skillBlock}\n\n${args}` : skillBlock;
}
```

Why this matters: Pi supports both implicit skill discovery and explicit skill application. The explicit path injects the full instructions deterministically instead of relying on the model to decide whether to `read` the file.

## 5. Skills are part of the resource-loading pipeline

The resource loader treats skills the same way it treats other configurable resources, including CLI paths, package resources, and extension-provided paths:

Source: `packages/coding-agent/src/core/resource-loader.ts`

```ts
const skillPaths = this.noSkills
	? this.mergePaths(cliEnabledSkills, this.additionalSkillPaths)
	: this.mergePaths([...cliEnabledSkills, ...enabledSkills], this.additionalSkillPaths);

this.lastSkillPaths = skillPaths;
this.updateSkillsFromPaths(skillPaths, metadataByPath);
```

Extension-provided resources can also add skills dynamically:

Source: `packages/coding-agent/src/core/resource-loader.ts`

```ts
extendResources(paths: ResourceExtensionPaths): void {
	const skillPaths = this.normalizeExtensionPaths(paths.skillPaths ?? []);
	...
	if (skillPaths.length > 0) {
		this.lastSkillPaths = this.mergePaths(
			this.lastSkillPaths,
			skillPaths.map((entry) => entry.path),
		);
		this.updateSkillsFromPaths(this.lastSkillPaths);
	}
}
```

Why this matters: Pi's skills system is not a hardcoded folder scan only. It is part of the platform's general resource model.

## 6. Pi handles collisions and hidden skills explicitly

Skill name collisions are detected during merge:

Source: `packages/coding-agent/src/core/skills.ts`

```ts
const existing = skillMap.get(skill.name);
if (existing) {
	collisionDiagnostics.push({
		type: "collision",
		message: `name "${skill.name}" collision`,
		...
	});
} else {
	skillMap.set(skill.name, skill);
	realPathSet.add(realPath);
}
```

And skills can be hidden from automatic model invocation while remaining available explicitly:

Source: `packages/coding-agent/src/core/skills.ts`

```ts
disableModelInvocation: frontmatter["disable-model-invocation"] === true,
...
const visibleSkills = skills.filter((s) => !s.disableModelInvocation);
```

Why this matters: Pi distinguishes between "installed" skills and "model-visible" skills. That is useful when a skill should be available by slash command but should not bias every prompt.

## Architectural tradeoffs and limitations

- Pi's skill system is strong on discoverability, packaging, and prompt efficiency.
- Automatic skill use still depends on the model deciding to read the right skill file. Pi advertises the skill and path, but does not force execution of the skill body unless the user calls `/skill:name`.
- Collision handling produces diagnostics, but the losing skill is simply shadowed by precedence/load order.
- The model-visible listing is compact by design, which saves tokens but means the model only sees summaries until it reads the full file.
- Skills are instruction artifacts, not executable workflows. If a task needs stronger guarantees than "follow this markdown procedure," Pi expects you to build an extension or tool instead.
