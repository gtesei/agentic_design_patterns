# Pi implementation notes: context management

Accessed on: 2026-05-17

Verdict: Pi has a real context-management stack. It assembles repo/user instruction files into the system prompt, advertises skills compactly, exposes a lower-level `transformContext` hook, and actively compresses or transfers context through compaction and branch summaries.

## Relevant Pi code

- `packages/coding-agent/src/core/resource-loader.ts`
- `packages/coding-agent/src/core/system-prompt.ts`
- `packages/coding-agent/src/core/agent-session.ts`
- `packages/coding-agent/src/core/compaction/compaction.ts`
- `packages/coding-agent/src/core/compaction/branch-summarization.ts`
- `packages/agent/src/agent-loop.ts`

## 1. Pi loads layered project context files

Pi walks from the current working directory to the filesystem root, collecting context files such as `AGENTS.md` and `CLAUDE.md`:

Source: `packages/coding-agent/src/core/resource-loader.ts`

```ts
export function loadProjectContextFiles(options: {
	cwd: string;
	agentDir: string;
}): Array<{ path: string; content: string }> {
	...
	const globalContext = loadContextFileFromDir(resolvedAgentDir);
	...
	let currentDir = resolvedCwd;
	const root = resolve("/");

	while (true) {
		const contextFile = loadContextFileFromDir(currentDir);
		if (contextFile && !seenPaths.has(contextFile.path)) {
			ancestorContextFiles.unshift(contextFile);
			seenPaths.add(contextFile.path);
		}
		if (currentDir === root) break;
		const parentDir = resolve(currentDir, "..");
		...
		currentDir = parentDir;
	}

	contextFiles.push(...ancestorContextFiles);
	return contextFiles;
}
```

Why this matters: Pi does not treat context as only chat history. It also treats repository instructions as part of the live working context, with ancestor-aware layering.

## 2. Pi builds context into the system prompt explicitly

The coding agent appends project context files, skills, date, and working directory into the prompt it sends to the model:

Source: `packages/coding-agent/src/core/system-prompt.ts`

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

Why this matters: Pi's context management is explicit and inspectable. The model sees concrete repo instructions and runtime state, not just a hidden middleware layer.

This prompt assembly is refreshed from the resource loader whenever the session rebuilds its prompt:

Source: `packages/coding-agent/src/core/agent-session.ts`

```ts
const loadedSkills = this._resourceLoader.getSkills().skills;
const loadedContextFiles = this._resourceLoader.getAgentsFiles().agentsFiles;

this._baseSystemPromptOptions = {
	cwd: this._cwd,
	skills: loadedSkills,
	contextFiles: loadedContextFiles,
	customPrompt: loaderSystemPrompt,
	appendSystemPrompt,
	selectedTools: validToolNames,
	toolSnippets,
	promptGuidelines,
};
return buildSystemPrompt(this._baseSystemPromptOptions);
```

Why this matters: context composition is part of normal session state management, not a one-time bootstrap step.

## 3. The lower-level agent loop supports pre-send context rewriting

Below the coding-agent layer, the generic agent loop exposes a `transformContext` hook:

Source: `packages/agent/src/agent-loop.ts`

```ts
let messages = context.messages;
if (config.transformContext) {
	messages = await config.transformContext(messages, signal);
}

const llmMessages = await config.convertToLlm(messages);
```

Why this matters: Pi supports context management as a programmable phase. Applications can prune, reorder, summarize, or otherwise transform the message list before provider conversion.

## 4. Pi automatically compacts context when it approaches the model window

The session checks the last assistant response against the model context window and compaction settings:

Source: `packages/coding-agent/src/core/agent-session.ts`

```ts
const settings = this.settingsManager.getCompactionSettings();
...
const contextWindow = this.model?.contextWindow ?? 0;
...
if (shouldCompact(contextTokens, contextWindow, settings)) {
	await this._runAutoCompaction("threshold", false);
}
```

Source: `packages/coding-agent/src/core/compaction/compaction.ts`

```ts
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};
```

Why this matters: Pi does not wait for context management to be handled manually. It has a built-in policy for reserving response space and summarizing older context.

The compaction cut point is computed by walking backward from the newest messages:

Source: `packages/coding-agent/src/core/compaction/compaction.ts`

```ts
for (let i = endIndex - 1; i >= startIndex; i--) {
	const entry = entries[i];
	if (entry.type !== "message") continue;

	const messageTokens = estimateTokens(entry.message);
	accumulatedTokens += messageTokens;

	if (accumulatedTokens >= keepRecentTokens) {
		for (let c = 0; c < cutPoints.length; c++) {
			if (cutPoints[c] >= i) {
				cutIndex = cutPoints[c];
				break;
			}
		}
		break;
	}
}
```

Why this matters: the compaction logic is not a vague "summarize sometimes" heuristic. It has an explicit boundary-finding algorithm and token budget policy.

## 5. Pi also manages context across branch switches

When navigating a session tree, Pi can summarize the branch being left behind and attach that summary to the destination path:

Source: `packages/coding-agent/src/core/agent-session.ts`

```ts
const { entries: entriesToSummarize, commonAncestorId } = collectEntriesForBranchSummary(
	this.sessionManager,
	oldLeafId,
	targetId,
);
...
const result = await generateBranchSummary(entriesToSummarize, {
	model,
	apiKey,
	headers,
	signal: this._branchSummaryAbortController.signal,
	customInstructions,
	replaceInstructions,
	reserveTokens: branchSummarySettings.reserveTokens,
});
...
const summaryId = this.sessionManager.branchWithSummary(
	newLeafId,
	summaryText,
	summaryDetails,
	fromExtension,
);
```

Source: `packages/coding-agent/src/core/compaction/branch-summarization.ts`

```ts
const oldPath = new Set(session.getBranch(oldLeafId).map((e) => e.id));
const targetPath = session.getBranch(targetId);
...
while (current && current !== commonAncestorId) {
	const entry = session.getEntry(current);
	if (!entry) break;
	entries.push(entry);
	current = entry.parentId;
}
entries.reverse();
```

Why this matters: Pi manages context not just over time, but over branching histories. It preserves abandoned branch work in condensed form instead of discarding it.

## Architectural tradeoffs and limitations

- Context files are appended wholesale. I did not find a selective relevance-ranking step before those file contents enter the prompt.
- Compaction uses exact usage numbers when available, but it can fall back to token estimation heuristics like `chars / 4`, so cut points are approximate in some cases.
- Both compaction and branch summaries depend on LLM summarization, so they can lose detail or introduce summary bias.
- `transformContext` is powerful, but it is a lower-level hook. The default coding-agent behavior still centers on prompt assembly plus compaction rather than arbitrary context graph logic.
- Branch summaries are strong for tree navigation, but they are not the same as a global retrieval system; they only preserve context along explicit session branches.
