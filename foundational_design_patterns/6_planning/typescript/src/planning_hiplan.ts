/**
 * Planning Pattern: HiPlan-style hierarchical decomposition
 * — TypeScript port of src/planning_hiplan.py.
 *
 * Anchor scenario: onboarding automation for a new enterprise customer.
 *
 * This script demonstrates:
 *   - high-level milestones
 *   - sub-plan generation per milestone
 *   - milestone-by-milestone execution summaries
 */

import { ChatOpenAI } from "@langchain/openai";

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

// ---------------------------------------------------------------------------
// HiPlan milestone with generated subtasks.
// ---------------------------------------------------------------------------

export interface Milestone {
  name: string;
  goal: string;
  subtasks: string[];
}

// ---------------------------------------------------------------------------
// LLM (lazy-init so smoke tests can import without OPENAI_API_KEY).
// ---------------------------------------------------------------------------

let _llm: ChatOpenAI | undefined;
function llm(): ChatOpenAI {
  if (!_llm) {
    _llm = new ChatOpenAI({ model: MODEL, temperature: 0 });
  }
  return _llm;
}

function contentToString(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) =>
        typeof part === "string"
          ? part
          : typeof part === "object" && part !== null && "text" in part
            ? String((part as { text: unknown }).text)
            : "",
      )
      .join("");
  }
  return String(content);
}

// ---------------------------------------------------------------------------
// Milestone / subtask / execution generators
// ---------------------------------------------------------------------------

export async function generateMilestones(objective: string): Promise<Milestone[]> {
  const prompt = `
Create 5 milestones for the objective below.
Return each milestone as: name | goal

Objective:
${objective}
`;
  const response = await llm().invoke(prompt);
  const lines = contentToString(response.content).trim().split(/\r?\n/);

  const milestones: Milestone[] = [];
  for (const line of lines) {
    if (!line.includes("|")) continue;
    const sep = line.indexOf("|");
    const left = line.slice(0, sep).replace(/^[-•\s]+/, "").trim();
    const right = line.slice(sep + 1).trim();
    milestones.push({ name: left, goal: right, subtasks: [] });
  }

  return milestones.slice(0, 5);
}

export async function generateSubtasks(
  milestone: Milestone,
  objective: string,
): Promise<Milestone> {
  const prompt = `
You are planning execution details for one milestone.
Objective: ${objective}
Milestone: ${milestone.name}
Goal: ${milestone.goal}

Return 4 atomic subtasks, one per line.
`;
  const response = await llm().invoke(prompt);
  const lines = contentToString(response.content).trim().split(/\r?\n/);
  milestone.subtasks = lines
    .map((line) => line.replace(/^[-•\s]+/, "").trim())
    .filter((line) => line.length > 0)
    .slice(0, 4);
  return milestone;
}

export async function executeSubtasks(milestone: Milestone): Promise<string[]> {
  const logs: string[] = [];
  for (const subtask of milestone.subtasks) {
    const prompt = `
Simulate execution of this onboarding subtask and return one short status sentence.
Subtask: ${subtask}
`;
    const response = await llm().invoke(prompt);
    const status = contentToString(response.content).trim();
    logs.push(`${subtask} -> ${status}`);
  }
  return logs;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("PLANNING PATTERN — HIPLAN HIERARCHICAL DECOMPOSITION");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required for this example.");
    return;
  }

  const objective =
    "Onboard a new enterprise customer for our SaaS platform, including security review, " +
    "identity integration, data migration, training, and go-live support.";

  const milestones = await generateMilestones(objective);
  if (milestones.length === 0) {
    console.log("⚠️ Could not generate milestones.");
    return;
  }

  console.log("\n🏁 Milestones:");
  milestones.forEach((ms, idx) => {
    console.log(`  ${idx + 1}. ${ms.name} — ${ms.goal}`);
  });

  console.log("\n🔧 Generating subtasks and execution logs...");
  for (const ms of milestones) {
    await generateSubtasks(ms, objective);
    const logs = await executeSubtasks(ms);

    console.log(`\n📌 ${ms.name}`);
    for (const sub of ms.subtasks) {
      console.log(`   - ${sub}`);
    }
    console.log("   Execution:");
    for (const log of logs) {
      console.log(`     • ${log}`);
    }
  }
}

if (import.meta.main) {
  await main();
}
