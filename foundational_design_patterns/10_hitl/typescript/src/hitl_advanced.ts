/**
 * Advanced Human-in-the-Loop (HITL) Implementation with Risk-Based Checkpoints
 * — TypeScript port of src/hitl_advanced.py.
 */

import OpenAI from "openai";
import { createInterface, type Interface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { writeFile } from "node:fs/promises";
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

export enum RiskLevel {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
}

export enum CheckpointType {
  FINANCIAL = "financial",
  COMPLIANCE = "compliance",
  CONTENT_QUALITY = "content_quality",
  CUSTOMER_IMPACT = "customer_impact",
}

export interface RiskAssessment {
  risk_level: RiskLevel;
  checkpoint_type: CheckpointType;
  risk_factors: string[];
  score: number;
  requires_approval: boolean;
  justification: string;
}

export interface ApprovalDecision {
  timestamp: string;
  decision: "approve" | "reject" | "escalate";
  approver: string;
  risk_assessment: {
    risk_level: RiskLevel;
    checkpoint_type: CheckpointType;
    risk_factors: string[];
    score: number;
    requires_approval: boolean;
    justification: string;
  };
  feedback?: string | null;
  task_details?: Record<string, unknown>;
}

export type AskFn = (prompt: string) => Promise<string>;

export function printHeader(text: string): void {
  console.log(`\n${Colors.HEADER}${Colors.BOLD}${"=".repeat(80)}${Colors.ENDC}`);
  console.log(`${Colors.HEADER}${Colors.BOLD}${text.padStart((80 + text.length) / 2).padEnd(80)}${Colors.ENDC}`);
  console.log(`${Colors.HEADER}${Colors.BOLD}${"=".repeat(80)}${Colors.ENDC}\n`);
}

export function printSection(title: string): void {
  console.log(`\n${Colors.CYAN}${Colors.BOLD}${title}${Colors.ENDC}`);
  console.log(`${Colors.CYAN}${"-".repeat(title.length)}${Colors.ENDC}`);
}

export function printRiskLevel(riskLevel: RiskLevel): void {
  const colors = {
    [RiskLevel.LOW]: Colors.GREEN,
    [RiskLevel.MEDIUM]: Colors.YELLOW,
    [RiskLevel.HIGH]: Colors.RED,
  };
  const color = colors[riskLevel] ?? Colors.ENDC;
  console.log(`${color}${Colors.BOLD}Risk Level: ${riskLevel.toUpperCase()}${Colors.ENDC}`);
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

export function extractJsonBlock(raw: string): string {
  let cleaned = raw.trim();
  if (cleaned.includes("```json")) {
    cleaned = cleaned.split("```json")[1]!.split("```")[0]!.trim();
  } else if (cleaned.includes("```")) {
    cleaned = cleaned.split("```")[1]!.split("```")[0]!.trim();
  }
  return cleaned;
}

export function riskLevelFromScore(score: number): RiskLevel {
  if (score >= 70) {
    return RiskLevel.HIGH;
  }
  if (score >= 40) {
    return RiskLevel.MEDIUM;
  }
  return RiskLevel.LOW;
}

export class RiskAnalyzer {
  client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  async assessFinancialRisk(
    amount: number,
    description: string,
  ): Promise<RiskAssessment> {
    const riskFactors: string[] = [];
    let score = 0;

    if (amount > 10_000) {
      riskFactors.push(`High transaction amount: $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
      score += 60;
    } else if (amount > 1_000) {
      riskFactors.push(`Moderate transaction amount: $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
      score += 30;
    } else {
      riskFactors.push(`Low transaction amount: $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
      score += 10;
    }

    try {
      const response = await this.client.chat.completions.create({
        model: getDefaultModel(),
        messages: [
          {
            role: "system",
            content:
              "You are a financial risk analyst. Analyze the transaction description and provide a risk score from 0-40 and list any concerning factors.",
          },
          {
            role: "user",
            content:
              `Transaction: ${description}\nAmount: $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}\n` +
              'Provide risk score (0-40) and factors in JSON format: {"score": <number>, "factors": [<list of strings>]}',
          },
        ],
        temperature: 0.3,
        max_tokens: 200,
      });

      const resultText = extractJsonBlock(
        response.choices[0]?.message?.content?.trim() ?? "",
      );
      const result = JSON.parse(resultText) as {
        score?: number;
        factors?: string[];
      };
      score += result.score ?? 0;
      riskFactors.push(...(result.factors ?? []));
    } catch (error: unknown) {
      printWarning(
        `AI risk analysis failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      score += 20;
      riskFactors.push("Unable to perform detailed risk analysis");
    }

    const riskLevel = riskLevelFromScore(score);
    return {
      risk_level: riskLevel,
      checkpoint_type: CheckpointType.FINANCIAL,
      risk_factors: riskFactors,
      score: Math.min(score, 100),
      requires_approval:
        riskLevel === RiskLevel.MEDIUM || riskLevel === RiskLevel.HIGH,
      justification:
        "Financial transaction risk assessment based on amount and description analysis",
    };
  }

  async assessComplianceRisk(
    content: string,
    regulations: string[],
  ): Promise<RiskAssessment> {
    const riskFactors: string[] = [];
    let score = 0;
    const regulationsStr = regulations.join(", ");

    try {
      const response = await this.client.chat.completions.create({
        model: getDefaultModel(),
        messages: [
          {
            role: "system",
            content:
              "You are a compliance officer. Analyze content for potential regulatory violations.",
          },
          {
            role: "user",
            content:
              `Content: ${content}\n\nRegulations to check: ${regulationsStr}\n\n` +
              'Provide risk assessment in JSON: {"score": <0-100>, "violations": [<list>], "concerns": [<list>]}',
          },
        ],
        temperature: 0.2,
        max_tokens: 300,
      });

      const resultText = extractJsonBlock(
        response.choices[0]?.message?.content?.trim() ?? "",
      );
      const result = JSON.parse(resultText) as {
        score?: number;
        violations?: string[];
        concerns?: string[];
      };
      score = result.score ?? 50;
      riskFactors.push(...(result.violations ?? []));
      riskFactors.push(...(result.concerns ?? []));
    } catch (error: unknown) {
      printWarning(
        `Compliance analysis failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      score = 50;
      riskFactors.push("Unable to perform detailed compliance analysis");
    }

    const riskLevel = riskLevelFromScore(score);
    return {
      risk_level: riskLevel,
      checkpoint_type: CheckpointType.COMPLIANCE,
      risk_factors: riskFactors,
      score,
      requires_approval:
        riskLevel === RiskLevel.MEDIUM || riskLevel === RiskLevel.HIGH,
      justification: `Compliance risk assessment for ${regulationsStr}`,
    };
  }

  assessCustomerImpactRisk(
    action: string,
    affectedCustomers: number,
  ): RiskAssessment {
    const riskFactors: string[] = [];
    let score = 0;

    if (affectedCustomers > 1000) {
      riskFactors.push(`High customer impact: ${affectedCustomers.toLocaleString()} customers`);
      score += 60;
    } else if (affectedCustomers > 100) {
      riskFactors.push(`Moderate customer impact: ${affectedCustomers.toLocaleString()} customers`);
      score += 35;
    } else {
      riskFactors.push(`Low customer impact: ${affectedCustomers.toLocaleString()} customers`);
      score += 15;
    }

    const highRiskKeywords = ["cancel", "delete", "terminate", "suspend", "charge", "penalty"];
    const mediumRiskKeywords = ["change", "update", "modify", "notify"];
    const actionLower = action.toLowerCase();

    if (highRiskKeywords.some((keyword) => actionLower.includes(keyword))) {
      riskFactors.push("Action contains high-impact keywords");
      score += 30;
    } else if (mediumRiskKeywords.some((keyword) => actionLower.includes(keyword))) {
      riskFactors.push("Action contains moderate-impact keywords");
      score += 15;
    }

    const riskLevel = riskLevelFromScore(score);
    return {
      risk_level: riskLevel,
      checkpoint_type: CheckpointType.CUSTOMER_IMPACT,
      risk_factors: riskFactors,
      score,
      requires_approval:
        riskLevel === RiskLevel.MEDIUM || riskLevel === RiskLevel.HIGH,
      justification:
        `Customer impact assessment for action affecting ${affectedCustomers.toLocaleString()} customers`,
    };
  }
}

export class AdvancedHITLWorkflow {
  riskAnalyzer: RiskAnalyzer;
  auditLog: ApprovalDecision[];
  ask: AskFn;

  constructor(riskAnalyzer: RiskAnalyzer, ask: AskFn) {
    this.riskAnalyzer = riskAnalyzer;
    this.ask = ask;
    this.auditLog = [];
  }

  displayRiskAssessment(assessment: RiskAssessment): void {
    printSection("🔍 RISK ASSESSMENT");

    printRiskLevel(assessment.risk_level);
    console.log(`${Colors.BOLD}Checkpoint Type:${Colors.ENDC} ${assessment.checkpoint_type}`);
    console.log(`${Colors.BOLD}Risk Score:${Colors.ENDC} ${assessment.score.toFixed(1)}/100`);
    console.log(
      `${Colors.BOLD}Requires Approval:${Colors.ENDC} ${
        assessment.requires_approval ? "Yes" : "No"
      }`,
    );

    console.log(`\n${Colors.BOLD}Risk Factors:${Colors.ENDC}`);
    assessment.risk_factors.forEach((factor) => console.log(`  • ${factor}`));

    console.log(`\n${Colors.BOLD}Justification:${Colors.ENDC}`);
    console.log(`  ${assessment.justification}`);
  }

  async getApproval(
    assessment: RiskAssessment,
    taskDetails: Record<string, unknown>,
  ): Promise<ApprovalDecision> {
    printSection("⚠️  APPROVAL REQUIRED");

    console.log(`\n${Colors.BOLD}Task Details:${Colors.ENDC}`);
    Object.entries(taskDetails).forEach(([key, value]) => {
      console.log(`  ${key}: ${value}`);
    });

    console.log(`\n${Colors.YELLOW}${Colors.BOLD}Approval Options:${Colors.ENDC}`);
    console.log(`  ${Colors.GREEN}[A]${Colors.ENDC} Approve - Proceed with the action`);
    console.log(`  ${Colors.RED}[R]${Colors.ENDC} Reject - Block the action`);
    console.log(`  ${Colors.YELLOW}[E]${Colors.ENDC} Escalate - Send to higher authority`);

    while (true) {
      const decision = (await this.ask(
        `\n${Colors.BOLD}Your decision [A/R/E]:${Colors.ENDC} `,
      ))
        .trim()
        .toUpperCase();

      if (decision === "A" || decision === "APPROVE") {
        const feedback = await this.ask(
          `${Colors.BOLD}Approval notes (optional):${Colors.ENDC} `,
        );
        return {
          timestamp: new Date().toISOString(),
          decision: "approve",
          approver: "human_reviewer",
          risk_assessment: { ...assessment },
          feedback: feedback.trim() || "Approved without additional notes",
          task_details: taskDetails,
        };
      }

      if (decision === "R" || decision === "REJECT") {
        const reason = await this.ask(`${Colors.BOLD}Rejection reason:${Colors.ENDC} `);
        return {
          timestamp: new Date().toISOString(),
          decision: "reject",
          approver: "human_reviewer",
          risk_assessment: { ...assessment },
          feedback: reason.trim() || "Rejected without specific reason",
          task_details: taskDetails,
        };
      }

      if (decision === "E" || decision === "ESCALATE") {
        const notes = await this.ask(`${Colors.BOLD}Escalation notes:${Colors.ENDC} `);
        return {
          timestamp: new Date().toISOString(),
          decision: "escalate",
          approver: "human_reviewer",
          risk_assessment: { ...assessment },
          feedback: notes.trim() || "Escalated for higher-level review",
          task_details: taskDetails,
        };
      }

      printError("Invalid option. Please choose A, R, or E.");
    }
  }

  async processWithCheckpoint(
    assessment: RiskAssessment,
    taskDetails: Record<string, unknown>,
    autoApproveLowRisk = true,
  ): Promise<boolean> {
    this.displayRiskAssessment(assessment);

    if (assessment.risk_level === RiskLevel.LOW && autoApproveLowRisk) {
      printSection("✓ AUTO-APPROVED");
      printSuccess("Low-risk item automatically approved");

      this.auditLog.push({
        timestamp: new Date().toISOString(),
        decision: "approve",
        approver: "system_auto",
        risk_assessment: { ...assessment },
        feedback: "Automatically approved - low risk",
        task_details: taskDetails,
      });
      return true;
    }

    if (assessment.requires_approval) {
      const decision = await this.getApproval(assessment, taskDetails);
      this.auditLog.push(decision);

      if (decision.decision === "approve") {
        printSuccess("Action approved by human reviewer");
        return true;
      }
      if (decision.decision === "reject") {
        printError("Action rejected by human reviewer");
        return false;
      }

      printWarning("Action escalated to higher authority");
      return false;
    }

    return true;
  }

  showAuditTrail(): void {
    printSection("📋 COMPREHENSIVE AUDIT TRAIL");

    if (!this.auditLog.length) {
      console.log("No decisions recorded.");
      return;
    }

    this.auditLog.forEach((decision, index) => {
      console.log(`\n${Colors.BOLD}${"─".repeat(80)}${Colors.ENDC}`);
      console.log(`${Colors.BOLD}Decision #${index + 1}${Colors.ENDC}`);
      console.log(`Timestamp: ${decision.timestamp}`);
      console.log(`Decision: ${decision.decision.toUpperCase()}`);
      console.log(`Approver: ${decision.approver}`);
      console.log(`Risk Level: ${decision.risk_assessment.risk_level}`);
      console.log(`Risk Score: ${decision.risk_assessment.score.toFixed(1)}/100`);
      console.log(`Checkpoint: ${decision.risk_assessment.checkpoint_type}`);

      if (decision.feedback) {
        console.log(`Feedback: ${decision.feedback}`);
      }

      if (decision.task_details) {
        console.log(`${Colors.BOLD}Task Details:${Colors.ENDC}`);
        Object.entries(decision.task_details).forEach(([key, value]) => {
          console.log(`  ${key}: ${value}`);
        });
      }
    });
  }

  async exportAuditLog(filepath: string): Promise<void> {
    try {
      await writeFile(filepath, JSON.stringify(this.auditLog, null, 2));
      printSuccess(`Audit log exported to: ${filepath}`);
    } catch (error: unknown) {
      printError(
        `Failed to export audit log: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }
}

async function askWithInterface(rl: Interface, prompt: string): Promise<string> {
  return await rl.question(prompt);
}

export async function runFinancialScenario(
  workflow: AdvancedHITLWorkflow,
  analyzer: RiskAnalyzer,
): Promise<void> {
  printHeader("SCENARIO: EXPENSE APPROVAL WORKFLOW");

  const expenses: Array<[number, string]> = [
    [500, "Office supplies and equipment"],
    [2500, "Conference attendance and travel"],
    [15000, "New server infrastructure purchase"],
  ];

  for (const [amount, description] of expenses) {
    printSection(`Processing Expense: $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
    console.log(`Description: ${description}`);

    const assessment = await analyzer.assessFinancialRisk(amount, description);
    const taskDetails = {
      Amount: `$${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      Description: description,
      Department: "Engineering",
      "Requested By": "John Doe",
    };

    const approved = await workflow.processWithCheckpoint(assessment, taskDetails);
    if (approved) {
      printSuccess(`Expense of $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} has been processed`);
    } else {
      printError(`Expense of $${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} was not approved`);
    }

    await workflow.ask(`\n${Colors.BOLD}Press Enter to continue...${Colors.ENDC}`);
  }
}

export async function runCustomerImpactScenario(
  workflow: AdvancedHITLWorkflow,
  analyzer: RiskAnalyzer,
): Promise<void> {
  printHeader("SCENARIO: CUSTOMER NOTIFICATION WORKFLOW");

  const actions: Array<[string, number]> = [
    ["Send product update notification", 50],
    ["Notify about pricing change", 500],
    ["Send service termination notice", 2000],
  ];

  for (const [action, customerCount] of actions) {
    printSection(`Processing Action: ${action}`);
    console.log(`Affected Customers: ${customerCount.toLocaleString()}`);

    const assessment = analyzer.assessCustomerImpactRisk(action, customerCount);
    const taskDetails = {
      Action: action,
      "Affected Customers": customerCount.toLocaleString(),
      "Scheduled For": "Next 24 hours",
      Category: "Customer Communication",
    };

    const approved = await workflow.processWithCheckpoint(assessment, taskDetails);
    if (approved) {
      printSuccess(`Customer action approved: ${action}`);
    } else {
      printError(`Customer action blocked: ${action}`);
    }

    await workflow.ask(`\n${Colors.BOLD}Press Enter to continue...${Colors.ENDC}`);
  }
}

export async function main(): Promise<number> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    printError("OPENAI_API_KEY not found in environment variables.");
    console.log(`Please ensure .env file exists at: ${ENV_PATH}`);
    return 1;
  }

  const rl = createInterface({ input, output });
  try {
    const analyzer = new RiskAnalyzer(apiKey);
    const workflow = new AdvancedHITLWorkflow(
      analyzer,
      (prompt) => askWithInterface(rl, prompt),
    );

    printHeader("ADVANCED HITL WITH RISK-BASED CHECKPOINTS");
    console.log("\nThis demo shows risk-based approval workflows.");
    console.log("Low-risk items are auto-approved, medium/high-risk require human review.\n");

    const scenarios: Record<string, [string, () => Promise<void>]> = {
      "1": ["Financial Expense Approval", () => runFinancialScenario(workflow, analyzer)],
      "2": ["Customer Impact Assessment", () => runCustomerImpactScenario(workflow, analyzer)],
    };

    printSection("Available Scenarios");
    Object.entries(scenarios).forEach(([key, [name]]) => {
      console.log(`${key}. ${name}`);
    });

    while (true) {
      const choice = (await askWithInterface(
        rl,
        `\n${Colors.BOLD}Select scenario (1-2) or 'q' to quit:${Colors.ENDC} `,
      )).trim();

      if (choice.toLowerCase() === "q") {
        console.log("\nGoodbye!");
        return 0;
      }

      if (choice in scenarios) {
        await scenarios[choice]![1]();
        break;
      }

      printError("Invalid choice. Please select 1 or 2.");
    }

    workflow.showAuditTrail();
    const auditFile = join(dirname(fileURLToPath(import.meta.url)), "audit_log.json");
    await workflow.exportAuditLog(auditFile);

    printSection("📊 WORKFLOW SUMMARY");
    const totalDecisions = workflow.auditLog.length;
    const approved = workflow.auditLog.filter((d) => d.decision === "approve").length;
    const rejected = workflow.auditLog.filter((d) => d.decision === "reject").length;
    const escalated = workflow.auditLog.filter((d) => d.decision === "escalate").length;
    const autoApproved = workflow.auditLog.filter((d) => d.approver === "system_auto").length;

    console.log(`Total decisions: ${totalDecisions}`);
    console.log(`  ${Colors.GREEN}Approved:${Colors.ENDC} ${approved} (${autoApproved} auto-approved)`);
    console.log(`  ${Colors.RED}Rejected:${Colors.ENDC} ${rejected}`);
    console.log(`  ${Colors.YELLOW}Escalated:${Colors.ENDC} ${escalated}`);
    return 0;
  } finally {
    rl.close();
  }
}

if (import.meta.main) {
  process.exitCode = await main();
}
