# Pi — Computer Use (merged)

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** consolidates `pi_claude.md` (Claude direct read), `pi_pi.md` (Pi maintainer notes, 2026-05-15), and `pi_codex.md` (Codex agent read, 2026-05-17). Every excerpt re-verified against a fresh clone on 2026-05-17.

## Summary

**Not implemented in the browser/UI-automation sense the pattern means in 2026.** Pi does not ship a `computer_use` tool, a browser-control loop (Playwright / Puppeteer / Stagehand), Anthropic's computer-use API integration, OpenAI's CUA integration, or any screenshot→click→observe agent surface. The built-in tool inventory at `packages/coding-agent/src/core/tools/index.ts:83` is `"read" | "bash" | "edit" | "write" | "grep" | "find" | "ls"` — file system + shell only. No `examples/extensions/` entry implements browser or desktop control either.

What Pi **does** provide that is **multimodal-adjacent** (and sometimes conflated with computer use):

- **Image attachments via CLI** — `processFileArguments` in `cli/file-processor.ts` detects image MIME types and attaches them as `ImageContent` blocks on the user message.
- **Image content returned by the `read` tool** — `read("screenshot.png")` returns inline base64 image data so the model can *see* the file's contents.
- **Clipboard-image paste in interactive mode** — `handleClipboardImagePaste` writes the clipboard image to a temp file and inserts the path so the model can read it.
- **macOS screenshot filename normalization** — `path-utils.ts` handles `"Capture d'écran"` Unicode variants for the `read` tool.

All of the above are multimodal **input** ergonomics — the agent can reason about images that are *already on disk or already pasted*. None of it is interactive UI **control**. The pattern as it appears in agentic-design-patterns 2026 (agent operates a GUI via screenshot + coordinate-action loop) has no implementation in Pi.

There is a clear path to building computer use as a custom extension on top of `pi.registerTool` + Playwright (or `nut.js`/`robotjs` for desktop) — Pi's framework supports it; the repo just doesn't ship it.

## Where it lives (or doesn't)

| Capability | File:line | What it is |
|---|---|---|
| `bash` tool — execute arbitrary shell commands | `packages/coding-agent/src/core/tools/bash.ts` | Real shell access. The agent can `curl`, run scripts, invoke any installed CLI. The broad "computer use." |
| `read` tool — including inline image embedding | `packages/coding-agent/src/core/tools/read.ts:245-276` | The agent can read image files; embedded as `ImageContent`. **Multimodal input, not UI control.** |
| `edit` / `write` tools — filesystem mutations | `packages/coding-agent/src/core/tools/{edit,write}.ts` | The agent can modify files. |
| `find` / `grep` / `ls` — filesystem inspection | `packages/coding-agent/src/core/tools/{find,grep,ls}.ts` | Search and listing. |
| Built-in tool inventory (constants) | `packages/coding-agent/src/core/tools/index.ts:83-95` | The complete `ToolName` enum and `allToolNames` set — no `click` / `screenshot` / `browser` in sight. |
| CLI file processor — image attachment handling | `packages/coding-agent/src/cli/file-processor.ts:23-99` | Detects image MIME, attaches as `ImageContent` to user message. |
| Clipboard image paste | `packages/coding-agent/src/modes/interactive/interactive-mode.ts:2430-2446` | TUI convenience for pasting images. |
| macOS screenshot filename normalization | `packages/coding-agent/src/core/tools/path-utils.ts:11-21` | Path-handling helper for the `read` tool to open macOS screenshot files. |

None of this is the computer-use pattern. The pattern requires an *interaction loop* with a GUI: take screenshot → reason about pixels → emit click/type at coordinates → observe new screenshot → repeat. Pi has none of that loop.

## Key code excerpts

### Built-in tool inventory — no GUI primitives

```ts
// packages/coding-agent/src/core/tools/index.ts:83-95
export type ToolName = "read" | "bash" | "edit" | "write" | "grep" | "find" | "ls";
export const allToolNames: Set<ToolName> = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);

export interface ToolsOptions {
    read?: ReadToolOptions;
    bash?: BashToolOptions;
    write?: WriteToolOptions;
    edit?: EditToolOptions;
    grep?: GrepToolOptions;
    find?: FindToolOptions;
    ls?: LsToolOptions;
}
```

**Why relevant:** The complete inventory. Seven tools, all file/shell. There is no `click`, no `screenshot`, no `browser`, no `key_press` — and no place to register them in core. Any computer-use capability has to come from an extension.

### CLI file processor — image input ergonomics

```ts
// packages/coding-agent/src/cli/file-processor.ts:23-99 (excerpted)
export async function processFileArguments(fileArgs: string[], options?: ProcessFileOptions): Promise<ProcessedFiles> {
    const autoResizeImages = options?.autoResizeImages ?? true;
    let text = "";
    const images: ImageContent[] = [];
    // ...
    const mimeType = await detectSupportedImageMimeTypeFromFile(absolutePath);

    if (mimeType) {
        const content = await readFile(absolutePath);
        const base64Content = content.toString("base64");
        // ...
        images.push(attachment);
        // ...
    } else {
        const content = await readFile(absolutePath, "utf-8");
        text += `<file name="${absolutePath}">\n${content}\n</file>\n`;
    }
}
```

**Why relevant:** Pi can give a model screenshots or other images via CLI arguments (`pi "what is in this?" screenshot.png`). Adjacent to computer use, but still passive — the agent reasons about a picture, it doesn't *take* the picture or interact with what's in it.

### Read tool — image content in tool results

```ts
// packages/coding-agent/src/core/tools/read.ts:245-276 (excerpted)
const mimeType = ops.detectImageMimeType ? await ops.detectImageMimeType(absolutePath) : undefined;
let content: (TextContent | ImageContent)[];
// ...
if (mimeType) {
    const buffer = await ops.readFile(absolutePath);
    const base64 = buffer.toString("base64");
    // ...
    content = [
        { type: "text", text: textNote },
        { type: "image", data: resized.data, mimeType: resized.mimeType },
    ];
} else {
    // Read text content.
```

**Why relevant:** Pi can inspect screenshots or UI captures if they're already available as files. Same caveat — the agent can `read("./screenshot.png")` and see what's in it, but cannot generate a fresh screenshot, click on what it sees, or scroll to see more.

### Clipboard image paste in TUI

```ts
// packages/coding-agent/src/modes/interactive/interactive-mode.ts:2430-2446 (excerpted)
private async handleClipboardImagePaste(): Promise<void> {
    try {
        const image = await readClipboardImage();
        if (!image) return;
        // ...
        const filePath = path.join(tmpDir, fileName);
        fs.writeFileSync(filePath, Buffer.from(image.bytes));

        // Insert file path directly
        this.editor.insertTextAtCursor?.(filePath);
        this.ui.requestRender();
```

**Why relevant:** Pi includes ergonomics for vision input — paste a screenshot from clipboard into the prompt and Pi writes it to a temp file, inserting the path so the model can `read` it. Pleasant UX, still not action-taking on the computer itself.

### Why the macOS screenshot path-utils is *not* computer use

```ts
// packages/coding-agent/src/core/tools/path-utils.ts:11-21 (excerpted)
function tryMacOSScreenshotPath(filePath: string): string {
    // macOS uses U+2019 (right single quotation mark) in screenshot names like "Capture d'écran"
    // ...
}
```

**Why relevant:** Path-normalization helper for the **`read` tool** so users on macOS can drag-and-drop screenshot files into the prompt and have the agent read them. The agent receives the image as `ImageContent` and reasons about it. **Multimodal input ergonomics**, not Pi controlling a display.

## What's missing relative to the 2026 computer-use pattern

- ❌ Browser automation (no Playwright / Puppeteer / Stagehand integration)
- ❌ Anthropic Computer Use API tool definition
- ❌ OpenAI Computer-Using-Agent (CUA) integration
- ❌ Screenshot capture (the agent cannot *take* a screenshot, only *read* an existing image file)
- ❌ Click / type / scroll primitives (no coordinate-based action emission)
- ❌ Desktop accessibility-tree inspection (macOS AX, Windows UIA, Linux AT-SPI)
- ❌ Browser-context isolation per task (no sandboxed browser sessions)
- ❌ DOM-level interaction (no JS execution, no element selectors)

## What an integrator *could* build on Pi today

Because Pi's extension surface is rich, a determined integrator could build computer use **out of tree** by:

1. Writing an extension that calls `pi.registerTool({ name: "click", parameters: ... })`, `pi.registerTool({ name: "screenshot", parameters: ... })`, etc.
2. Backing those tools with Playwright/Puppeteer/CDP (browser) or `nut.js`/`robotjs` (desktop).
3. Returning screenshots as `ImageContent` in the tool result so the agent can reason about the resulting state on the next turn.

This is genuinely possible — Pi's framework primitives (custom tools with image-content results, `beforeToolCall` hooks for safety, parallel execution) are sufficient. But it would be the integrator's work; nothing in the repo today implements this.

A second path: wrap an existing browser-use library as a sidecar process and have Pi's `bash` tool invoke it. Then the "computer use" is happening in the sidecar, not Pi.

## Tradeoffs and limitations

- **Process-level access is unrestricted.** The `bash` tool can spawn anything the user can spawn. The framework gives no built-in container/sandbox — extensions like `permission-gate.ts` (regex-block dangerous commands) and `protected-paths.ts` (block writes to sensitive paths) are the available guardrails (see `10_hitl/pi.md`).
- **Vision is read-only.** The agent can see images but cannot generate or capture them through framework tools.
- **No structured-action vocabulary.** Even if a computer-use extension were built, Pi has no shared action vocabulary (click / type / scroll / drag) — every extension would invent its own schema. The pattern is still maturing in the broader 2026 ecosystem; no standard exists for Pi to align with.
- **Multimodal input ergonomics are good** (CLI attachment, `read`-tool inline images, clipboard paste, macOS screenshot path handling) — easy to confuse with computer use, but it's all passive image reading.

## "Not implemented" — final word

The 2026 computer-use pattern — *agent interacts with a GUI via screenshot + coordinate-action loop* — is **not implemented** in Pi. The reasons appear to be (a) Pi's focus on coding-agent workflows where shell/filesystem tools are sufficient, and (b) the lack of consensus on a UI-automation primitive in the broader TypeScript ecosystem.

If the chapter wants a Pi example, the honest framing is: *"Pi does not implement computer use. It provides shell + filesystem tools (which cover the 'agent operates programs and files' interpretation) plus solid multimodal-input ergonomics (CLI image attachment, `read`-tool inline images, clipboard paste, macOS screenshot path handling — but these are passive image reading, not interactive UI control). A browser-control or desktop-UI loop would require building an extension on top of Pi's custom-tool API plus an underlying automation library — there is no first-party or example implementation in the repo."*
