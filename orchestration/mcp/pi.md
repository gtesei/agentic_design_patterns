# Pi — Model Context Protocol (MCP)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Not implemented, explicitly and intentionally.** Pi does not ship an MCP client, MCP server, or MCP transport in its core coding-agent packages.

The project documentation says this directly: MCP belongs in extensions or packages, not in the minimal core.

## Where it lives (or doesn't)

| Concern | Status in Pi |
|---|---|
| Built-in MCP client | ❌ none found |
| Built-in MCP server | ❌ none found |
| MCP transport integration | ❌ none found |
| Explicit "no MCP in core" policy | ✅ `packages/coding-agent/README.md`, `packages/coding-agent/docs/usage.md` |
| Local resource discovery hook | ✅ `resources_discover`, but this is **not** MCP |

## Key code excerpts

Source: `packages/coding-agent/README.md:470-472`

```md
Pi is aggressively extensible so it doesn't have to dictate your workflow.

**No MCP.** Build CLI tools with READMEs (see [Skills](#skills)), or build an extension that adds MCP support.
```

Why this matters: the absence of MCP is not accidental. It is a stated design decision.

Source: `packages/coding-agent/docs/usage.md:273-275`

```md
Pi keeps the core small and pushes workflow-specific behavior into extensions, skills, prompt templates, and packages.

It intentionally does not include built-in MCP, sub-agents, permission popups, plan mode, to-dos, or background bash.
```

Why this matters: the docs frame MCP the same way Pi frames several other out-of-core orchestration features.

Source: `packages/coding-agent/docs/extensions.md:339-352`

```ts
pi.on("resources_discover", async (event, _ctx) => {
	return {
		skillPaths: ["/path/to/skills"],
		promptPaths: ["/path/to/prompts"],
		themePaths: ["/path/to/themes"],
	};
});
```

Why this matters: this is the closest built-in mechanism to capability discovery, but it only discovers local Pi resources. It is not MCP.

Source: `packages/coding-agent/src/core/extensions/runner.ts:998-1035`

```ts
const skillPaths: Array<{ path: string; extensionPath: string }> = [];
const promptPaths: Array<{ path: string; extensionPath: string }> = [];
const themePaths: Array<{ path: string; extensionPath: string }> = [];
...
const event: ResourcesDiscoverEvent = { type: "resources_discover", cwd, reason };
const handlerResult = await handler(event, ctx);
...
if (result?.skillPaths?.length) {
	skillPaths.push(...result.skillPaths.map((path) => ({ path, extensionPath: ext.path })));
}
```

Why this matters: the runtime does implement discovery, but only for Pi-native resources contributed by extensions.

Source: `packages/coding-agent/examples/extensions/dynamic-resources/index.ts:7-14`

```ts
export default function (pi: ExtensionAPI) {
	pi.on("resources_discover", () => {
		return {
			skillPaths: [join(baseDir, "SKILL.md")],
			promptPaths: [join(baseDir, "dynamic.md")],
			themePaths: [join(baseDir, "dynamic.json")],
		};
	});
}
```

Why this matters: the example confirms the intended pattern: extensions contribute local skills/prompts/themes, not MCP endpoints.

## Tradeoffs and limitations

- Pi stays smaller and simpler by refusing to make MCP a required core abstraction.
- The downside is interoperability: stock Pi cannot talk to MCP servers without extra integration work.
- An integrator could write an MCP bridge extension that registers remote MCP tools via `pi.registerTool`, but that bridge is not part of this repository.

## Final word

Pi does **not** implement MCP. The closest thing in-tree is `resources_discover`, which is a local resource-discovery hook, not Model Context Protocol support.
