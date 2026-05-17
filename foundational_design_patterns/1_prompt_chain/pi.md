# Pi: Prompt Chain

Accessed on: 2026-05-17

## Verdict

Pi does not appear to implement prompt chaining as a first-class runtime abstraction in its core packages. The closest mechanisms are:

- prompt templates that expand `/name args` into a larger prompt body
- system-prompt composition that appends context files and skills into one prompt
- example workflow prompts that tell another tool to run a multi-step chain

That is useful, but it is not the same as a dedicated multi-stage prompt pipeline with typed stage outputs and explicit stage-to-stage transitions.

## How Pi gets close

Pi has a prompt-template loader plus a deterministic expander:

Source: `packages/coding-agent/src/core/prompt-templates.ts:279-295`

```ts
export function expandPromptTemplate(text: string, templates: PromptTemplate[]): string {
	if (!text.startsWith("/")) return text;

	const match = text.match(/^\/([^\s]+)(?:\s+([\s\S]*))?$/);
	if (!match) return text;

	const templateName = match[1];
	const argsString = match[2] ?? "";

	const template = templates.find((t) => t.name === templateName);
	if (template) {
		const args = parseCommandArgs(argsString);
		return substituteArgs(template.content, args);
	}

	return text;
}
```

Why this matters: this is the main prompt-level indirection mechanism I found in Pi. A slash command can expand into a larger prompt, but the result is still one prompt string, not a chained prompt graph.

That expansion happens before the normal agent turn is sent to the model:

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

Why this matters: Pi treats prompt expansion as pre-processing. It does not create a chain runtime with intermediate assistant outputs; it rewrites the input and then runs the normal loop.

Pi also composes a single system prompt from multiple sources:

Source: `packages/coding-agent/src/core/system-prompt.ts:153-169`

```ts
if (contextFiles.length > 0) {
	prompt += "\n\n# Project Context\n\n";
	prompt += "Project-specific instructions and guidelines:\n\n";
	for (const { path: filePath, content } of contextFiles) {
		prompt += `## ${filePath}\n\n${content}\n\n`;
	}
}

if (hasRead && skills.length > 0) {
	prompt += formatSkillsForPrompt(skills);
}

prompt += `\nCurrent date: ${date}`;
prompt += `\nCurrent working directory: ${promptCwd}`;
```

Why this matters: Pi is good at prompt composition, but this is still prompt assembly, not prompt chaining.

The nearest explicit "chain" I found is in an example workflow prompt that instructs the `subagent` tool to execute multiple steps:

Source: `packages/coding-agent/examples/extensions/subagent/prompts/implement.md:4-10`

```md
Use the subagent tool with the chain parameter to execute this workflow:

1. First, use the "scout" agent to find all code relevant to: $@
2. Then, use the "planner" agent to create an implementation plan for "$@" using the context from the previous step (use {previous} placeholder)
3. Finally, use the "worker" agent to implement the plan from the previous step (use {previous} placeholder)

Execute this as a chain, passing output between steps via {previous}.
```

Why this matters: Pi can express chained workflows, but here the chain lives in prompt text plus the `subagent` extension, not in a core prompt-chain API.

## Tradeoffs and limitations

- Pi's prompt-template system is simple and easy to reason about because it is just deterministic text expansion.
- The downside is that there is no first-class notion of prompt stages, typed intermediate outputs, per-stage retries, or prompt-chain observability in the core runtime.
- If this pattern is meant to mean "one prompt feeds another prompt in a dedicated chain abstraction," Pi does not meaningfully implement that in core.
