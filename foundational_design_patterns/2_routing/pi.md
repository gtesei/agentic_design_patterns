# Pi: Routing

Accessed on: 2026-05-17

## Verdict

Pi does implement routing, but mostly as deterministic input dispatch rather than an LLM-driven semantic router. The main routing surfaces I found are:

- slash-command routing to extension commands
- prompt-template routing by command name
- skill-command routing by `/skill:name`
- extension input hooks that can intercept or transform input before the normal turn

I did not find a first-class "classifier chooses one of several expert prompts" pattern in the core loop.

## How Pi routes input

The main dispatch path is in `AgentSession.prompt()`:

Source: `packages/coding-agent/src/core/agent-session.ts:973-1008`

```ts
if (expandPromptTemplates && text.startsWith("/")) {
	const handled = await this._tryExecuteExtensionCommand(text);
	if (handled) {
		preflightResult?.(true);
		return;
	}
}

let currentText = text;
let currentImages = options?.images;
if (this._extensionRunner.hasHandlers("input")) {
	const inputResult = await this._extensionRunner.emitInput(
		currentText,
		currentImages,
		options?.source ?? "interactive",
	);
	if (inputResult.action === "handled") {
		preflightResult?.(true);
		return;
	}
	if (inputResult.action === "transform") {
		currentText = inputResult.text;
		currentImages = inputResult.images ?? currentImages;
	}
}

let expandedText = currentText;
if (expandPromptTemplates) {
	expandedText = this._expandSkillCommand(expandedText);
	expandedText = expandPromptTemplate(expandedText, [...this.promptTemplates]);
}
```

Why this matters: the order is explicit. Pi first checks extension commands, then lets extensions intercept or rewrite input, then expands skill commands and prompt templates. That is a real routing layer, but it is rule-based.

Extension commands are routed by slash-command name:

Source: `packages/coding-agent/src/core/agent-session.ts:1116-1130`

```ts
private async _tryExecuteExtensionCommand(text: string): Promise<boolean> {
	const spaceIndex = text.indexOf(" ");
	const commandName = spaceIndex === -1 ? text.slice(1) : text.slice(1, spaceIndex);
	const args = spaceIndex === -1 ? "" : text.slice(spaceIndex + 1);

	const command = this._extensionRunner.getCommand(commandName);
	if (!command) return false;

	const ctx = this._extensionRunner.createCommandContext();

	try {
		await command.handler(args, ctx);
		return true;
```

Why this matters: command routing is name-based and deterministic. There is no model deciding which command path to take.

Skill routing is also name-based:

Source: `packages/coding-agent/src/core/agent-session.ts:1147-1162`

```ts
private _expandSkillCommand(text: string): string {
	if (!text.startsWith("/skill:")) return text;

	const spaceIndex = text.indexOf(" ");
	const skillName = spaceIndex === -1 ? text.slice(7) : text.slice(7, spaceIndex);
	const args = spaceIndex === -1 ? "" : text.slice(spaceIndex + 1).trim();

	const skill = this.resourceLoader.getSkills().skills.find((s) => s.name === skillName);
	if (!skill) return text;

	try {
		const content = readFileSync(skill.filePath, "utf-8");
		const body = stripFrontmatter(content).trim();
		const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${skill.baseDir}.\n\n${body}\n</skill>`;
		return args ? `${skillBlock}\n\n${args}` : skillBlock;
```

Why this matters: skills are not selected by a semantic router either. The user or caller names the route explicitly.

Pi also exposes the routed command inventory with source labels:

Source: `packages/coding-agent/src/modes/rpc/rpc-mode.ts:622-652`

```ts
case "get_commands": {
	const commands: RpcSlashCommand[] = [];

	for (const command of session.extensionRunner.getRegisteredCommands()) {
		commands.push({
			name: command.invocationName,
			description: command.description,
			source: "extension",
			sourceInfo: command.sourceInfo,
		});
	}

	for (const template of session.promptTemplates) {
		commands.push({
			name: template.name,
			description: template.description,
			source: "prompt",
			sourceInfo: template.sourceInfo,
		});
	}

	for (const skill of session.resourceLoader.getSkills().skills) {
		commands.push({
			name: `skill:${skill.name}`,
			description: skill.description,
			source: "skill",
			sourceInfo: skill.sourceInfo,
		});
	}
```

Why this matters: the codebase explicitly models routed command sources as `extension`, `prompt`, and `skill`.

## Tradeoffs and limitations

- This routing design is predictable and debuggable because it is explicit and name-based.
- It is not the same as a semantic router that reads user intent and chooses among expert chains or prompts.
- If the pattern is meant to mean classifier-based prompt routing, Pi only approximates it through extension hooks that can transform or intercept input.
