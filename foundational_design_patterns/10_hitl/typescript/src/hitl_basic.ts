/**
 * Basic Human-in-the-Loop (HITL) Implementation
 * — TypeScript port of src/hitl_basic.py.
 */

import OpenAI from "openai";
import { createInterface, type Interface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_MODEL = "gpt-4o-mini";
const ENV_PATH = join(dirname(fileURLToPath(import.meta.url)), "../../../../.env");

function getDefaultModel(): string {
  return process.env.OPENAI_MODEL ?? DEFAULT_MODEL;
}

export class Colors {
  static HEADER = "\x1b[95m";
  static BLUE = "\x1b[94m";
  static CYAN = "\x1b[96m";
  static GREEN = "\x1b[92m";
  static YELLOW = "\x1b[93m";
  static RED = "\x1b[91m";
  static ENDC = "\x1b[0m";
  static BOLD = "\x1b[1m";
  static UNDERLINE = "\x1b[4m";
}

export function printHeader(text: string): void {
  console.log(`\n${Colors.HEADER}${Colors.BOLD}${"=".repeat(80)}${Colors.ENDC}`);
  console.log(`${Colors.HEADER}${Colors.BOLD}${text.padStart((80 + text.length) / 2).padEnd(80)}${Colors.ENDC}`);
  console.log(`${Colors.HEADER}${Colors.BOLD}${"=".repeat(80)}${Colors.ENDC}\n`);
}

export function printSection(title: string): void {
  console.log(`\n${Colors.CYAN}${Colors.BOLD}${title}${Colors.ENDC}`);
  console.log(`${Colors.CYAN}${"-".repeat(title.length)}${Colors.ENDC}`);
}

export function printContent(content: string): void {
  const lines = content.split("\n");
  const maxLength = Math.max(...lines.map((line) => line.length)) + 4;

  console.log(`\n${Colors.BLUE}+${"-".repeat(maxLength)}+${Colors.ENDC}`);
  for (const line of lines) {
    console.log(
      `${Colors.BLUE}|${Colors.ENDC} ${line.padEnd(maxLength - 2)} ${Colors.BLUE}|${Colors.ENDC}`,
    );
  }
  console.log(`${Colors.BLUE}+${"-".repeat(maxLength)}+${Colors.ENDC}\n`);
}

export function printSuccess(text: string): void {
  console.log(`${Colors.GREEN}${Colors.BOLD}✓ ${text}${Colors.ENDC}`);
}

export function printError(text: string): void {
  console.log(`${Colors.RED}${Colors.BOLD}✗ ${text}${Colors.ENDC}`);
}

export function printWarning(text: string): void {
  console.log(`${Colors.YELLOW}${Colors.BOLD}⚠ ${text}${Colors.ENDC}`);
}

export class ContentGenerator {
  client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  async generateContent(
    contentType: string,
    topic: string,
    additionalInstructions = "",
  ): Promise<string | null> {
    const prompts: Record<string, string> = {
      blog: `Write a short blog post (200-300 words) about: ${topic}. ${additionalInstructions}`,
      email: `Write a professional email about: ${topic}. ${additionalInstructions}`,
      social: `Write a social media post (2-3 sentences) about: ${topic}. ${additionalInstructions}`,
      tweet: `Write a tweet (280 characters max) about: ${topic}. ${additionalInstructions}`,
    };

    const prompt = prompts[contentType] ?? prompts.blog ?? "";

    try {
      const response = await this.client.chat.completions.create({
        model: getDefaultModel(),
        messages: [
          {
            role: "system",
            content:
              "You are a professional content writer. Create engaging, clear, and concise content.",
          },
          { role: "user", content: prompt },
        ],
        temperature: 0.7,
        max_tokens: 500,
      });

      return response.choices[0]?.message?.content?.trim() ?? null;
    } catch (error: unknown) {
      printError(
        `Failed to generate content: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      return null;
    }
  }
}

export type ReviewDecision = "approve" | "reject" | "modify";
export type AskFn = (prompt: string) => Promise<string>;

export interface DecisionEntry {
  attempt?: number;
  decision?: ReviewDecision;
  feedback?: string | null;
  content?: string;
  action?: string;
  content_type?: string;
}

export class HITLWorkflow {
  generator: ContentGenerator;
  decisionLog: DecisionEntry[];
  autoDecision: ReviewDecision | null;
  autoFeedback: string | null;
  ask: AskFn;

  constructor(
    generator: ContentGenerator,
    ask: AskFn,
    autoDecision?: ReviewDecision,
    autoFeedback?: string,
  ) {
    this.generator = generator;
    this.decisionLog = [];
    this.ask = ask;
    this.autoDecision = autoDecision ?? null;
    this.autoFeedback = autoFeedback ?? null;
  }

  async getHumanDecision(
    content: string,
    contentType: string,
  ): Promise<[ReviewDecision, string | null]> {
    printSection("🔍 HUMAN REVIEW CHECKPOINT");
    console.log(`\n${Colors.BOLD}Content Type:${Colors.ENDC} ${contentType.toUpperCase()}`);
    console.log(`${Colors.BOLD}Generated Content:${Colors.ENDC}`);
    printContent(content);

    console.log(`\n${Colors.YELLOW}${Colors.BOLD}Review Options:${Colors.ENDC}`);
    console.log(`  ${Colors.GREEN}[A]${Colors.ENDC} Approve - Publish the content as-is`);
    console.log(`  ${Colors.RED}[R]${Colors.ENDC} Reject - Discard the content`);
    console.log(`  ${Colors.YELLOW}[M]${Colors.ENDC} Modify - Request changes to the content`);

    if (this.autoDecision === "approve") {
      printWarning("Automatic review decision enabled: approve");
      return ["approve", this.autoFeedback];
    }
    if (this.autoDecision === "reject") {
      printWarning("Automatic review decision enabled: reject");
      return ["reject", this.autoFeedback ?? "Auto-rejected"];
    }
    if (this.autoDecision === "modify") {
      printWarning("Automatic review decision enabled: modify");
      return [
        "modify",
        this.autoFeedback ?? "Please tighten the draft and improve clarity.",
      ];
    }

    while (true) {
      const decision = (await this.ask(
        `\n${Colors.BOLD}Your decision [A/R/M]:${Colors.ENDC} `,
      ))
        .trim()
        .toUpperCase();

      if (decision === "A" || decision === "APPROVE") {
        return ["approve", null];
      }
      if (decision === "R" || decision === "REJECT") {
        const reason = await this.ask(
          `${Colors.BOLD}Reason for rejection (optional):${Colors.ENDC} `,
        );
        return ["reject", reason.trim() || "No reason provided"];
      }
      if (decision === "M" || decision === "MODIFY") {
        const feedback = (await this.ask(
          `${Colors.BOLD}What changes would you like?${Colors.ENDC} `,
        )).trim();
        if (feedback) {
          return ["modify", feedback];
        }
        printWarning("Please provide feedback for modifications.");
        continue;
      }
      printError("Invalid option. Please choose A, R, or M.");
    }
  }

  publishContent(content: string, contentType: string): void {
    printSection("📤 PUBLISHING CONTENT");
    console.log(`Publishing ${contentType} content...`);
    printSuccess(`${contentType.toUpperCase()} content published successfully!`);

    this.decisionLog.push({
      action: "published",
      content_type: contentType,
      content,
    });
  }

  async runWorkflow(contentType: string, topic: string): Promise<boolean> {
    printHeader("BASIC HITL CONTENT GENERATION WORKFLOW");

    printSection("📝 CONTENT GENERATION");
    console.log(`Content Type: ${contentType}`);
    console.log(`Topic: ${topic}`);
    console.log("\nGenerating content...");

    let additionalInstructions = "";
    const maxAttempts = 3;
    let attempt = 1;

    while (attempt <= maxAttempts) {
      const content = await this.generator.generateContent(
        contentType,
        topic,
        additionalInstructions,
      );

      if (!content) {
        printError("Failed to generate content. Aborting workflow.");
        return false;
      }

      printSuccess(`Content generated (Attempt ${attempt}/${maxAttempts})`);
      const [decision, feedback] = await this.getHumanDecision(content, contentType);

      this.decisionLog.push({
        attempt,
        decision,
        feedback,
        content,
      });

      if (decision === "approve") {
        this.publishContent(content, contentType);
        return true;
      }

      if (decision === "reject") {
        printSection("❌ CONTENT REJECTED");
        console.log(`Reason: ${feedback}`);
        printError("Workflow terminated by human reviewer.");
        return false;
      }

      printSection("🔄 REGENERATING CONTENT");
      console.log(`Feedback: ${feedback}`);
      additionalInstructions =
        `Previous feedback: ${feedback}. Please incorporate this feedback.`;
      attempt += 1;

      if (attempt <= maxAttempts) {
        console.log(`\nRegenerating content (Attempt ${attempt}/${maxAttempts})...`);
      } else {
        printWarning(`Maximum attempts (${maxAttempts}) reached.`);
        printError("Workflow terminated.");
        return false;
      }
    }

    return false;
  }

  showAuditTrail(): void {
    printSection("📋 AUDIT TRAIL");

    if (!this.decisionLog.length) {
      console.log("No decisions recorded.");
      return;
    }

    this.decisionLog.forEach((entry, index) => {
      console.log(`\n${Colors.BOLD}Entry ${index + 1}:${Colors.ENDC}`);
      Object.entries(entry).forEach(([key, value]) => {
        if (key === "content") {
          console.log(`  ${key}: [Content omitted for brevity]`);
        } else {
          console.log(`  ${key}: ${value}`);
        }
      });
    });
  }
}

export interface BasicArgs {
  scenario?: number;
  autoDecision?: ReviewDecision;
  autoFeedback?: string;
}

export function parseBasicArgs(argv: string[]): BasicArgs {
  const args: BasicArgs = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--scenario") {
      args.scenario = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--auto-decision") {
      const value = argv[index + 1] as ReviewDecision | undefined;
      if (value) {
        args.autoDecision = value;
      }
      index += 1;
      continue;
    }
    if (arg === "--auto-feedback") {
      args.autoFeedback = argv[index + 1];
      index += 1;
    }
  }

  return args;
}

async function askWithInterface(rl: Interface, prompt: string): Promise<string> {
  return await rl.question(prompt);
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const args = parseBasicArgs(argv);
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    printError("OPENAI_API_KEY not found in environment variables.");
    console.log(`Please ensure .env file exists at: ${ENV_PATH}`);
    return 1;
  }

  const rl = createInterface({ input, output });
  try {
    const generator = new ContentGenerator(apiKey);
    const workflow = new HITLWorkflow(
      generator,
      (prompt) => askWithInterface(rl, prompt),
      args.autoDecision,
      args.autoFeedback,
    );

    const scenarios: Array<[string, string]> = [
      [
        "blog",
        "The benefits of adopting a human-in-the-loop approach in AI systems",
      ],
      ["email", "Announcing a new AI safety feature to customers"],
      ["social", "Celebrating our company's commitment to ethical AI"],
    ];

    printHeader("WELCOME TO BASIC HITL DEMO");
    console.log(
      "\nThis demo shows a simple human-in-the-loop workflow for content generation.",
    );
    console.log(
      "You'll be asked to approve, reject, or request modifications to generated content.\n",
    );

    printSection("Available Scenarios");
    scenarios.forEach(([contentType, topic], index) => {
      console.log(
        `${index + 1}. ${Colors.BOLD}${contentType.toUpperCase()}${Colors.ENDC}: ${topic}`,
      );
    });

    let contentType: string;
    let topic: string;

    if (args.scenario == null) {
      while (true) {
        const choice = (await askWithInterface(
          rl,
          `\n${Colors.BOLD}Select a scenario (1-${scenarios.length}) or 'q' to quit:${Colors.ENDC} `,
        )).trim();

        if (choice.toLowerCase() === "q") {
          console.log("\nGoodbye!");
          return 0;
        }

        const choiceIndex = Number(choice) - 1;
        if (Number.isInteger(choiceIndex) && choiceIndex >= 0 && choiceIndex < scenarios.length) {
          [contentType, topic] = scenarios[choiceIndex]!;
          break;
        }

        printError(`Please enter a number between 1 and ${scenarios.length}.`);
      }
    } else {
      [contentType, topic] = scenarios[args.scenario - 1]!;
      printWarning(
        `Scenario override enabled. Auto-selecting scenario ${args.scenario}: ${contentType.toUpperCase()} - ${topic}`,
      );
    }

    const success = await workflow.runWorkflow(contentType, topic);
    workflow.showAuditTrail();

    printSection("📊 WORKFLOW SUMMARY");
    if (success) {
      printSuccess("Content was successfully published!");
    } else {
      printError("Content was not published.");
      if (args.scenario != null || args.autoDecision != null) {
        return 1;
      }
    }

    const decisionCount = workflow.decisionLog.filter((entry) => entry.decision).length;
    console.log(`\nTotal decisions made: ${decisionCount}`);
    return success ? 0 : 1;
  } finally {
    rl.close();
  }
}

if (import.meta.main) {
  process.exitCode = await main();
}
