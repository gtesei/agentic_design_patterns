/**
 * Multi-Agent Collaboration: Orchestrator-Worker topology
 * — TypeScript port of src/orchestrator_worker.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { Annotation, END, StateGraph } from "@langchain/langgraph";

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

export const OrchestratorState = Annotation.Root({
  goal: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  work_items: Annotation<string[]>({
    reducer: (_left, right) => right,
    default: () => [],
  }),
  results: Annotation<string[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  final_report: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
});

type OrchestratorStateType = typeof OrchestratorState.State;

const llm = new ChatOpenAI({
  model: MODEL,
  temperature: 0,
});

export async function supervisorPlan(
  state: OrchestratorStateType,
): Promise<Partial<OrchestratorStateType>> {
  const prompt = `
Break this support-ops research goal into exactly 3 worker tasks:
${state.goal}

Return one task per line.
`;

  const response = await llm.invoke(prompt);
  const tasks = String(response.content ?? "")
    .split("\n")
    .map((line) => line.trim().replace(/^[-•]\s*/, ""))
    .filter(Boolean)
    .slice(0, 3);

  return { work_items: tasks };
}

export async function workerResearch(
  state: OrchestratorStateType,
): Promise<Partial<OrchestratorStateType>> {
  if (!state.work_items.length) {
    return {};
  }

  const [task, ...remaining] = state.work_items;
  const response = await llm.invoke(
    `Research this subtask for support operations and return 2 concise bullets:\n${task}`,
  );

  return {
    work_items: remaining,
    results: [`TASK: ${task}\n${String(response.content ?? "")}`],
  };
}

export async function synthesize(
  state: OrchestratorStateType,
): Promise<Partial<OrchestratorStateType>> {
  const prompt = `
Synthesize a final report from these worker outputs.

Goal: ${state.goal}
Worker outputs:
${state.results.join("\n")}

Return:
- executive summary
- immediate actions (3 bullets)
- escalation recommendation
`;

  const response = await llm.invoke(prompt);
  return { final_report: String(response.content ?? "") };
}

export function routeWorker(state: OrchestratorStateType): string {
  return state.work_items.length ? "worker" : "synthesize";
}

export async function trySupervisorPackage(): Promise<boolean> {
  try {
    const importer = new Function(
      "specifier",
      "return import(specifier);",
    ) as (specifier: string) => Promise<unknown>;
    await importer("langgraph-supervisor");
    console.log(
      "✅ langgraph-supervisor detected. (This demo uses lightweight fallback graph for portability.)",
    );
    return true;
  } catch {
    console.log(
      "ℹ️ langgraph-supervisor not installed; using built-in fallback graph.",
    );
    return false;
  }
}

export function buildGraph() {
  return new StateGraph(OrchestratorState)
    .addNode("plan", supervisorPlan)
    .addNode("worker", workerResearch)
    .addNode("synthesize", synthesize)
    .addEdge("__start__", "plan")
    .addConditionalEdges("plan", routeWorker, {
      worker: "worker",
      synthesize: "synthesize",
    })
    .addConditionalEdges("worker", routeWorker, {
      worker: "worker",
      synthesize: "synthesize",
    })
    .addEdge("synthesize", END)
    .compile();
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("MULTI-AGENT — ORCHESTRATOR/WORKER");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required.");
    return;
  }

  await trySupervisorPackage();

  const app = buildGraph();
  const goal =
    "Investigate enterprise customer complaints about payment approval latency and produce " +
    "a triage recommendation for support operations.";
  const output = await app.invoke({
    goal,
    work_items: [],
    results: [],
    final_report: "",
  });

  console.log("\n🧾 Final report:");
  console.log(output.final_report);
}

if (import.meta.main) {
  await main();
}
