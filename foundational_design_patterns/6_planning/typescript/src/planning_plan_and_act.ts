/**
 * Planning Pattern: Plan-and-Act with explicit replanning (LangGraph)
 * — TypeScript port of src/planning_plan_and_act.py.
 *
 * Anchor scenario: incident response automation.
 *
 * Flow:
 *   - planner node creates ordered actions
 *   - executor node performs one action at a time
 *   - reviewer node decides whether replanning is required
 */

import { ChatOpenAI } from "@langchain/openai";
import { END, StateGraph, Annotation } from "@langchain/langgraph";

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

// ---------------------------------------------------------------------------
// State container for plan-and-act execution.
// ---------------------------------------------------------------------------

export const PlanActState = Annotation.Root({
  incident: Annotation<string>(),
  context: Annotation<string>(),
  plan: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  completed: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  current_action: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),
  notes: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  requires_replan: Annotation<boolean>({
    reducer: (_prev, next) => next,
    default: () => false,
  }),
});

type State = typeof PlanActState.State;

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
// Nodes
// ---------------------------------------------------------------------------

/** Create or refresh the execution plan. */
export async function plannerNode(state: State): Promise<Partial<State>> {
  const prompt = `
You are an incident commander. Build a short action plan (4-6 steps) for this incident.

Incident:
${state.incident}

Context:
${state.context}

Already completed:
${JSON.stringify(state.completed)}

Return one step per line, no numbering.
`;
  const response = await llm().invoke(prompt);
  const lines = contentToString(response.content).trim().split(/\r?\n/);
  const planRaw = lines
    .map((line) => line.replace(/^[-•\s]+/, "").trim())
    .filter((line) => line.length > 0);

  const plan = planRaw.filter((step) => !state.completed.includes(step));
  const notes = state.notes.concat(`planner: generated ${plan.length} remaining steps`);

  return { plan, notes, requires_replan: false };
}

/** Execute one action from the plan. */
export async function executorNode(state: State): Promise<Partial<State>> {
  if (state.plan.length === 0) {
    return { current_action: "" };
  }

  const action = state.plan[0]!;
  const remaining = state.plan.slice(1);

  const execPrompt = `
You are executing one incident response action.

Action: ${action}
Incident: ${state.incident}

Return exactly two lines:
STATUS: done|blocked
NOTE: <short operational note>
`;
  const result = await llm().invoke(execPrompt);
  const lines = contentToString(result.content)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  let status = "done";
  let note = "no note";
  for (const line of lines) {
    if (line.toLowerCase().startsWith("status:")) {
      status = line.slice(line.indexOf(":") + 1).trim().toLowerCase();
    }
    if (line.toLowerCase().startsWith("note:")) {
      note = line.slice(line.indexOf(":") + 1).trim();
    }
  }

  if (status === "done") {
    return {
      plan: remaining,
      current_action: action,
      completed: state.completed.concat(action),
      notes: state.notes.concat(`executor: done -> ${action} | ${note}`),
      requires_replan: false,
    };
  }
  return {
    plan: remaining,
    current_action: action,
    notes: state.notes.concat(`executor: blocked -> ${action} | ${note}`),
    requires_replan: true,
  };
}

/** Decide if another step can run or if replanning is needed. */
export function reviewerNode(state: State): Partial<State> {
  let note: string;
  if (state.requires_replan) {
    note = "reviewer: requesting replanning";
  } else if (state.plan.length > 0) {
    note = "reviewer: continue with remaining plan";
  } else {
    note = "reviewer: plan complete";
  }
  return { notes: state.notes.concat(note) };
}

export function routeAfterReview(state: State): "planner" | "executor" | typeof END {
  if (state.requires_replan) return "planner";
  if (state.plan.length > 0) return "executor";
  return END;
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export function buildGraph() {
  return new StateGraph(PlanActState)
    .addNode("planner", plannerNode)
    .addNode("executor", executorNode)
    .addNode("reviewer", reviewerNode)
    .addEdge("__start__", "planner")
    .addEdge("planner", "executor")
    .addEdge("executor", "reviewer")
    .addConditionalEdges("reviewer", routeAfterReview, {
      planner: "planner",
      executor: "executor",
      [END]: END,
    })
    .compile();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("PLANNING PATTERN — PLAN-AND-ACT (LANGGRAPH)");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required for this example.");
    return;
  }

  const initial = {
    incident: "Payment API error rate jumped to 35% after deployment.",
    context: "Service: billing-gateway, Region: eu-west, Recent change: release v2.8.4",
  };

  const app = buildGraph();
  const final = await app.invoke(initial);

  const completed = final.completed ?? [];
  const notes = final.notes ?? [];

  console.log("\n✅ Completed steps:");
  completed.forEach((step, idx) => {
    console.log(`  ${idx + 1}. ${step}`);
  });

  console.log("\n🧾 Execution notes:");
  for (const note of notes) {
    console.log(`  - ${note}`);
  }
}

if (import.meta.main) {
  await main();
}
