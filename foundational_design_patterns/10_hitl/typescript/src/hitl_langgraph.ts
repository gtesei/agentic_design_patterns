/**
 * Human-in-the-Loop (HITL) with LangGraph Integration
 * — TypeScript port of src/hitl_langgraph.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { Annotation, END, StateGraph } from "@langchain/langgraph";
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

export function printState(stateName: string): void {
  console.log(`\n${Colors.BLUE}${Colors.BOLD}[STATE: ${stateName}]${Colors.ENDC}`);
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

export interface ConversationEntry {
  timestamp: string;
  node: string;
  content?: string;
  revision?: number;
  decision?: string;
  feedback?: string;
  status?: string;
  final_output?: string;
}

export const WorkflowState = Annotation.Root({
  task: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  generated_content: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  human_feedback: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  approval_status: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "pending",
  }),
  revision_count: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 0,
  }),
  conversation_history: Annotation<ConversationEntry[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  final_output: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
});

type WorkflowStateType = typeof WorkflowState.State;
export type AskFn = (prompt: string) => Promise<string>;

const llm = new ChatOpenAI({
  model: getDefaultModel(),
  temperature: 0.7,
});

export async function generateContentNode(
  state: WorkflowStateType,
): Promise<Partial<WorkflowStateType>> {
  printState("GENERATE_CONTENT");
  console.log(`Generating content for: ${state.task}`);

  let prompt: string;
  if (state.human_feedback) {
    prompt = `Task: ${state.task}

Previous attempt: ${state.generated_content}

Human feedback: ${state.human_feedback}

Please revise the content based on the feedback above.`;
  } else {
    prompt = `Task: ${state.task}

Please generate high-quality content for this task. Be creative and professional.`;
  }

  const response = await llm.invoke(prompt);
  const generated = String(response.content ?? "");

  printSuccess("Content generated successfully");
  console.log(`\n${Colors.BOLD}Generated Content:${Colors.ENDC}`);
  console.log(`${Colors.BLUE}${"-".repeat(80)}${Colors.ENDC}`);
  console.log(generated);
  console.log(`${Colors.BLUE}${"-".repeat(80)}${Colors.ENDC}`);

  return {
    generated_content: generated,
    approval_status: "pending",
    conversation_history: [
      {
        timestamp: new Date().toISOString(),
        node: "generate_content",
        content: generated,
        revision: state.revision_count ?? 0,
      },
    ],
  };
}

export function createReviewNode(ask: AskFn) {
  return async function reviewNode(
    state: WorkflowStateType,
  ): Promise<Partial<WorkflowStateType>> {
    printState("HUMAN_REVIEW");
    printSection("🔍 HUMAN REVIEW CHECKPOINT");

    console.log(`\n${Colors.BOLD}Task:${Colors.ENDC} ${state.task}`);
    console.log(`\n${Colors.BOLD}Generated Content:${Colors.ENDC}`);
    console.log(`${Colors.BLUE}${"-".repeat(80)}${Colors.ENDC}`);
    console.log(state.generated_content);
    console.log(`${Colors.BLUE}${"-".repeat(80)}${Colors.ENDC}`);

    console.log(`\n${Colors.BOLD}Revision Count:${Colors.ENDC} ${state.revision_count ?? 0}`);

    console.log(`\n${Colors.YELLOW}${Colors.BOLD}Review Options:${Colors.ENDC}`);
    console.log(`  ${Colors.GREEN}[A]${Colors.ENDC} Approve - Accept the content`);
    console.log(`  ${Colors.RED}[R]${Colors.ENDC} Reject - Discard and end workflow`);
    console.log(`  ${Colors.YELLOW}[M]${Colors.ENDC} Modify - Request changes`);

    let approvalStatus = state.approval_status;
    let humanFeedback = state.human_feedback;
    let revisionCount = state.revision_count ?? 0;

    while (true) {
      const decision = (await ask(
        `\n${Colors.BOLD}Your decision [A/R/M]:${Colors.ENDC} `,
      ))
        .trim()
        .toUpperCase();

      if (decision === "A" || decision === "APPROVE") {
        approvalStatus = "approved";
        humanFeedback = "";
        printSuccess("Content approved!");
        break;
      }
      if (decision === "R" || decision === "REJECT") {
        approvalStatus = "rejected";
        const reason = await ask(`${Colors.BOLD}Rejection reason:${Colors.ENDC} `);
        humanFeedback = reason.trim() || "No reason provided";
        printError("Content rejected");
        break;
      }
      if (decision === "M" || decision === "MODIFY") {
        approvalStatus = "needs_revision";
        const feedback = (await ask(
          `${Colors.BOLD}What changes would you like?${Colors.ENDC} `,
        )).trim();
        if (feedback) {
          humanFeedback = feedback;
          revisionCount += 1;
          printWarning(`Requesting revision (Attempt ${revisionCount})`);
          break;
        }
        printError("Please provide feedback for modifications");
        continue;
      }
      printError("Invalid option. Please choose A, R, or M");
    }

    return {
      approval_status: approvalStatus,
      human_feedback: humanFeedback,
      revision_count: revisionCount,
      conversation_history: [
        {
          timestamp: new Date().toISOString(),
          node: "human_review",
          decision: approvalStatus,
          feedback: humanFeedback,
        },
      ],
    };
  };
}

export function revisionCheckNode(
  state: WorkflowStateType,
): Partial<WorkflowStateType> {
  printState("REVISION_CHECK");

  const maxRevisions = 3;
  const currentCount = state.revision_count ?? 0;

  if (currentCount >= maxRevisions) {
    printWarning(`Maximum revision limit (${maxRevisions}) reached`);
    return {
      approval_status: "rejected",
      human_feedback: `Maximum revisions (${maxRevisions}) exceeded`,
    };
  }

  return {};
}

export function finalizeNode(
  state: WorkflowStateType,
): Partial<WorkflowStateType> {
  printState("FINALIZE");

  if (state.approval_status === "approved") {
    printSection("✅ FINALIZING APPROVED CONTENT");
    printSuccess("Content has been finalized and is ready for use");
    return {
      final_output: state.generated_content,
      conversation_history: [
        {
          timestamp: new Date().toISOString(),
          node: "finalize",
          status: state.approval_status,
          final_output: state.generated_content,
        },
      ],
    };
  }

  printSection("❌ WORKFLOW TERMINATED");
  printError("No content was finalized");
  return {
    final_output: "",
    conversation_history: [
      {
        timestamp: new Date().toISOString(),
        node: "finalize",
        status: state.approval_status,
        final_output: "",
      },
    ],
  };
}

export function routingLogic(
  state: WorkflowStateType,
): "finalize" | "revision_check" | "generate_content" {
  const status = state.approval_status ?? "pending";
  if (status === "approved" || status === "rejected") {
    return "finalize";
  }
  if (status === "needs_revision") {
    return "revision_check";
  }
  return "generate_content";
}

export function createHitlWorkflow(ask: AskFn) {
  return new StateGraph(WorkflowState)
    .addNode("generate_content", generateContentNode)
    .addNode("review", createReviewNode(ask))
    .addNode("revision_check", revisionCheckNode)
    .addNode("finalize", finalizeNode)
    .addEdge("__start__", "generate_content")
    .addEdge("generate_content", "review")
    .addConditionalEdges("review", routingLogic, {
      finalize: "finalize",
      revision_check: "revision_check",
      generate_content: "generate_content",
    })
    .addConditionalEdges(
      "revision_check",
      (state) =>
        state.approval_status === "needs_revision"
          ? "generate_content"
          : "finalize",
      {
        generate_content: "generate_content",
        finalize: "finalize",
      },
    )
    .addEdge("finalize", END)
    .compile();
}

export function displayConversationHistory(history: ConversationEntry[]): void {
  printSection("📜 CONVERSATION HISTORY");

  history.forEach((entry, index) => {
    console.log(`\n${Colors.BOLD}Entry ${index + 1}:${Colors.ENDC}`);
    console.log(`  Timestamp: ${entry.timestamp}`);
    console.log(`  Node: ${entry.node}`);

    if (entry.content) {
      console.log(`  Content: [Generated content - revision ${entry.revision ?? 0}]`);
    }
    if (entry.decision) {
      console.log(`  Decision: ${entry.decision}`);
    }
    if (entry.feedback) {
      console.log(`  Feedback: ${entry.feedback}`);
    }
    if (entry.status) {
      console.log(`  Status: ${entry.status}`);
    }
  });
}

export async function runWorkflowExample(
  task: string,
  ask: AskFn,
): Promise<void> {
  printHeader("LANGGRAPH HITL WORKFLOW");
  console.log(`\n${Colors.BOLD}Task:${Colors.ENDC} ${task}\n`);

  const workflow = createHitlWorkflow(ask);
  const initialState = {
    task,
    generated_content: "",
    human_feedback: "",
    approval_status: "pending",
    revision_count: 0,
    conversation_history: [],
    final_output: "",
  };

  printSection("🚀 STARTING WORKFLOW");

  try {
    const finalState = await workflow.invoke(initialState, {
      configurable: { thread_id: "hitl_demo_1" },
    });

    printSection("📊 WORKFLOW RESULTS");
    console.log(`\n${Colors.BOLD}Final Status:${Colors.ENDC} ${finalState.approval_status}`);
    console.log(`${Colors.BOLD}Total Revisions:${Colors.ENDC} ${finalState.revision_count ?? 0}`);

    if (finalState.final_output) {
      console.log(`\n${Colors.BOLD}Final Output:${Colors.ENDC}`);
      console.log(`${Colors.GREEN}${"-".repeat(80)}${Colors.ENDC}`);
      console.log(finalState.final_output);
      console.log(`${Colors.GREEN}${"-".repeat(80)}${Colors.ENDC}`);
    } else {
      printError("\nNo final output produced");
    }

    displayConversationHistory(finalState.conversation_history);

    printSection("📈 SUMMARY STATISTICS");
    const totalEvents = finalState.conversation_history.length;
    const generationEvents = finalState.conversation_history.filter(
      (entry) => entry.node === "generate_content",
    ).length;
    const reviewEvents = finalState.conversation_history.filter(
      (entry) => entry.node === "human_review",
    ).length;

    console.log(`Total events: ${totalEvents}`);
    console.log(`Content generations: ${generationEvents}`);
    console.log(`Human reviews: ${reviewEvents}`);
    console.log(`Final status: ${finalState.approval_status}`);
  } catch (error: unknown) {
    printError(
      `Workflow error: ${error instanceof Error ? error.message : String(error)}`,
    );
    console.error(error);
  }
}

async function askWithInterface(rl: Interface, prompt: string): Promise<string> {
  return await rl.question(prompt);
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
    printHeader("WELCOME TO LANGGRAPH HITL DEMO");
    console.log("\nThis demo shows human-in-the-loop workflow using LangGraph.");
    console.log(
      "The workflow includes state management, conditional routing, and human checkpoints.\n",
    );

    const tasks = [
      "Write a professional email announcing a new product feature",
      "Create a social media post about company culture",
      "Draft a customer support response for a billing inquiry",
      "Generate a blog post introduction about AI ethics",
    ];

    printSection("Available Tasks");
    tasks.forEach((task, index) => {
      console.log(`${index + 1}. ${task}`);
    });

    console.log(`\n${Colors.BOLD}Or enter a custom task${Colors.ENDC}`);

    let task = "";
    while (true) {
      const choice = (await askWithInterface(
        rl,
        `\n${Colors.BOLD}Select task (1-${tasks.length}) or enter custom task:${Colors.ENDC} `,
      )).trim();

      if (/^\d+$/.test(choice) && Number(choice) >= 1 && Number(choice) <= tasks.length) {
        task = tasks[Number(choice) - 1]!;
        break;
      }
      if (choice) {
        task = choice;
        break;
      }
      printError("Please enter a valid choice");
    }

    await runWorkflowExample(task, (prompt) => askWithInterface(rl, prompt));

    printSection("✨ DEMO COMPLETE");
    console.log("\nKey LangGraph Features Demonstrated:");
    console.log("  • State management across workflow nodes");
    console.log("  • Human-in-the-loop checkpoints");
    console.log("  • Conditional routing based on decisions");
    console.log("  • Conversation history tracking");
    console.log("  • Iterative refinement with revision limits");
    return 0;
  } finally {
    rl.close();
  }
}

if (import.meta.main) {
  process.exitCode = await main();
}
