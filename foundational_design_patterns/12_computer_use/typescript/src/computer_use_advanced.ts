/**
 * Computer Use advanced — TypeScript port of src/computer_use_advanced.py.
 */

import { ChatOpenAI } from "@langchain/openai";

export async function runPlaywrightTask(): Promise<string> {
  try {
    const importer = new Function(
      "specifier",
      "return import(specifier);",
    ) as (specifier: string) => Promise<{
      chromium: {
        launch(options: { headless: boolean }): Promise<{
          newPage(): Promise<{
            goto(url: string, options: { waitUntil: string }): Promise<void>;
            title(): Promise<string>;
            locator(selector: string): { innerText(): Promise<string> };
          }>;
          close(): Promise<void>;
        }>;
      };
    }>;
    const playwright = await importer("playwright");
    const browser = await playwright.chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto("https://en.wikipedia.org/wiki/Artificial_intelligence", {
      waitUntil: "domcontentloaded",
    });
    const title = await page.title();
    const heading = await page.locator("#firstHeading").innerText();
    await browser.close();
    return `Visited page title='${title}', heading='${heading}'`;
  } catch {
    return "Playwright not installed; skipping real browser automation.";
  }
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("COMPUTER USE — ADVANCED");
  console.log("=".repeat(80));

  const safetyPolicy = {
    allowlist_domains: ["wikipedia.org"],
    blocked_actions: ["checkout", "payment", "account deletion"],
    human_approval_required_for: [
      "form submit",
      "external navigation",
      "file download",
    ],
  };

  console.log("\nSafety policy:");
  console.log(safetyPolicy);

  const outcome = await runPlaywrightTask();
  console.log("\nAutomation outcome:");
  console.log(outcome);

  if (process.env.OPENAI_API_KEY) {
    try {
      const llm = new ChatOpenAI({
        model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
        temperature: 0,
      });
      const response = await llm.invoke(
        `Summarize this browser automation run in 3 bullets:\n${outcome}`,
      );
      console.log("\nLLM summary:");
      console.log(response.content);
    } catch (exc: unknown) {
      console.log("\nLLM summary unavailable:");
      console.log(exc);
    }
  }
}

if (import.meta.main) {
  await main();
}
