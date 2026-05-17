/**
 * Computer Use basic — TypeScript port of src/computer_use_basic.py.
 */

export interface Step {
  screenshot: string;
  thought: string;
  action: string;
  observation: string;
}

function htmlToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

export async function fetchWikipediaSummary(topic: string): Promise<string> {
  const url = `https://en.wikipedia.org/wiki/${topic.replaceAll(" ", "_")}`;
  const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
  const html = await response.text();
  const paragraphMatch = html.match(/<p>([\s\S]*?)<\/p>/i);
  return paragraphMatch ? htmlToText(paragraphMatch[1]!) : "No summary paragraph found";
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("COMPUTER USE — BASIC LOOP (SIMULATED)");
  console.log("=".repeat(80));

  const steps: Step[] = [];
  const topic = "Large_language_model";

  const summary = await fetchWikipediaSummary(topic);
  steps.push({
    screenshot: "home_page.png",
    thought: "Need to gather infobox/summary for LLM topic.",
    action: "Open wikipedia page for Large language model.",
    observation: `${summary.slice(0, 180)}...`,
  });

  steps.forEach((step, index) => {
    console.log(`\nStep ${index + 1}`);
    console.log("screenshot:", step.screenshot);
    console.log("thought:", step.thought);
    console.log("action:", step.action);
    console.log("observation:", step.observation);
  });

  console.log(
    "\nSecurity note: keep computer-use agents sandboxed and require human approval for side effects.",
  );
}

if (import.meta.main) {
  await main();
}
