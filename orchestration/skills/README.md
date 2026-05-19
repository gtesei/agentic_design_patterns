# Skills Pattern

## Overview

The **Skills Pattern** packages agent capabilities as **on-disk, model-loadable instruction bundles** (`SKILL.md`). Instead of stuffing every capability into the system prompt or registering it as a fixed tool, skills are advertised compactly and loaded on demand — the agent sees a small index of what's available and pulls in the full body only for the skill it actually needs.

The pattern is grounded in the open [Agent Skills](https://agentskills.io) spec: a `SKILL.md` file with YAML frontmatter (`name`, `description`, optional `disable-model-invocation`) and a markdown body that prescribes a playbook the agent should follow when the description matches the task.

## How It Works

1. **Discovery**: At startup the agent scans skill directories (user-scope, project-scope, extension-provided) and parses each `SKILL.md` frontmatter.
2. **Advertisement (compact index)**: Only the metadata — `name` + `description` + `location` — is injected into the system prompt. Full bodies are never auto-loaded.
3. **Selection**: When a task matches a description, the agent either reads the body via its `read` tool (model-driven) or the user invokes it explicitly (deterministic).
4. **Execution**: The skill body is injected into the conversation and the agent follows its instructions, often calling out to scripts referenced from the skill directory.

### Progressive disclosure

```
System prompt:
  <available_skills>
    <skill><name>convert-csv</name><description>Convert CSVs to Parquet</description><location>…</location></skill>
    <skill><name>run-lint</name>     <description>Run repo-standard lint over a path</description><location>…</location></skill>
    ...
  </available_skills>

Agent decides to use convert-csv:
  read(/path/to/convert-csv/SKILL.md)
  → full body now in context → agent executes the playbook
```

A 50-skill project costs ~50 × ~150 tokens of metadata, **not** 50 × full bodies.

### Implicit vs explicit invocation

- **Implicit (model-decided)**: model reads the skill body when the description matches the task.
- **Explicit (`/skill:name args`)**: user pins the skill — frontmatter is stripped, body is injected verbatim before the args. Useful when you want to *force* a skill, not *hint* at one.

## Key Benefits

- **Token-efficient**: full bodies are loaded only when needed.
- **Composable**: capability lives on disk as ordinary markdown; ship it in a repo, a dotfile dir, or a plug-in package.
- **Auditable**: provenance (user vs project scope) travels with each skill, enabling trust boundaries against repo-controlled skills.
- **Playbook-shaped**: skills are *instruction sets*, not plugins — executable behavior is delegated to scripts the agent invokes through `bash`.
- **Spec-conformant**: stable contract (`name`, `description`) makes skills portable across compliant agents.

## When to Use This Pattern

**Rule of Thumb**: Use Skills when the agent needs a growing library of *opt-in* playbooks that would be wasteful or noisy to put in the always-on system prompt.

### Ideal Use Cases

- **Repo-local conventions**: `run-tests`, `release-branch`, `bump-version` playbooks tailored to one project.
- **User-level "muscle memory"**: personal skills like `summarize-as-bullets`, `format-as-rfc` that follow the user across projects.
- **Domain packs**: ship a skill bundle (e.g., a CSV-wrangling pack, a Terraform pack) as a distributable directory.
- **Sensitive operations**: hide them from auto-discovery with `disable-model-invocation: true` and require explicit `/skill:name` invocation.

### When NOT to Use

- A single always-available behavior that everyone always wants — put it in the system prompt or as a core tool.
- Behavior that must execute deterministically without LLM mediation — write a tool, not a skill.
- Workflows requiring complex state machines — skills are linear playbooks, not orchestration engines.

## Implementation Considerations

- **Required vs optional metadata**: `description` is required (the index falls apart without it); `name` falls back to the parent dir if omitted.
- **Name discipline**: enforce the spec regex (lowercase a-z, 0-9, hyphens; ≤ 64 chars; no leading/trailing/consecutive hyphens) at load time.
- **Collision policy**: when the same skill name appears under multiple sources, decide load-order precedence and surface the collision as a diagnostic — never silently overwrite.
- **Source provenance**: tag each skill with where it came from (user / project / extension) and gate project-scope skills behind explicit user opt-in.
- **Discovery hygiene**: honor `.gitignore`, skip `node_modules`; once a directory has `SKILL.md`, stop recursing into it (its subdirs are bundled assets).
- **Diagnostics, not exceptions**: bad frontmatter should skip the skill and emit a diagnostic, not crash loading.

## Example Architecture

```
Agent startup
   │
   ▼
[Resource loader]
   │   user paths    project paths    extension paths
   ▼          ▼              ▼
  Recursive scan → parse SKILL.md frontmatter → validate
   │
   ▼
[Skill index]  ──►  injected into system prompt as compact <available_skills>
   │
   ▼
Agent turn
   │
   ├── implicit: model calls read(<skill path>)
   └── explicit: user types /skill:name args → body injected before args
```

## Related Patterns

- **Tool Use**: skills *call* tools via `bash`; tools are the executable layer beneath skill playbooks.
- **Context Management**: skill metadata vs body is a deliberate context-budget tradeoff.
- **Routing**: explicit `/skill:name` is a deterministic-routing path that bypasses LLM classification.
- **Subagents**: a subagent system prompt is the closest cousin to a skill body — both are markdown instruction sets selected by name.

## Demos in this directory

- `src/skills_basic.py`: metadata discovery then full skill load on selection.
- `src/skills_advanced.py`: LLM-assisted selection from the metadata catalog.
- `skills/*.SKILL.md`: example capability packages.

Run:

```bash
uv sync
bash run.sh
```

## Corporate SSL proxy note

If you're behind a corporate SSL-inspecting proxy, run examples with:

```bash
AGENTIC_DISABLE_SSL=1 bash run.sh
```

---

*Skills turn ad-hoc system-prompt cruft into a discoverable, portable, opt-in library of agent playbooks.*
