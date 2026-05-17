/**
 * RAG Pattern: Agentic loop — TypeScript port of src/rag_agentic.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { Annotation, END, StateGraph } from "@langchain/langgraph";
import { HybridRetriever, buildSupportChunks } from "./rag_basic.ts";

const llm = new ChatOpenAI({
  model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
  temperature: 0,
});

export const AgenticRAGState = Annotation.Root({
  question: Annotation<string>(),
  rewritten_query: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  retrieved_context: Annotation<string[]>({
    reducer: (_left, right) => right,
    default: () => [],
  }),
  context_grade: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "insufficient",
  }),
  fallback_notes: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  draft_answer: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  final_answer: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
});

type State = typeof AgenticRAGState.State;

let retrieverPromise: Promise<HybridRetriever> | undefined;
function getRetriever(): Promise<HybridRetriever> {
  if (!retrieverPromise) {
    retrieverPromise = HybridRetriever.create(buildSupportChunks());
  }
  return retrieverPromise;
}

export async function rewriteQuery(state: State): Promise<Partial<State>> {
  const prompt =
    "Rewrite this support query to maximize retrieval quality. Keep key entities and intent.\n\n" +
    `Query: ${state.question}`;
  const response = await llm.invoke(prompt);
  return { rewritten_query: String(response.content).trim() };
}

export async function retrieve(state: State): Promise<Partial<State>> {
  const retriever = await getRetriever();
  const retrieved = await retriever.retrieve(
    state.rewritten_query || state.question,
    5,
  );
  return {
    retrieved_context: retrieved.map(([chunk]) => `[${chunk.source}] ${chunk.text}`),
  };
}

export async function gradeContext(state: State): Promise<Partial<State>> {
  const context = state.retrieved_context.join("\n");
  const prompt = `You are grading retrieval quality.
Question: ${state.question}
Context:
${context}

Return one word only: sufficient OR insufficient.`;
  const response = await llm.invoke(prompt);
  const verdict = String(response.content).trim().toLowerCase();
  return {
    context_grade:
      verdict.includes("sufficient") && !verdict.includes("insufficient")
        ? "sufficient"
        : "insufficient",
  };
}

export async function maybeFallback(state: State): Promise<Partial<State>> {
  if (state.context_grade === "sufficient") {
    return { fallback_notes: "fallback not required" };
  }
  return {
    fallback_notes:
      "Context was insufficient. In production this branch would trigger web search, ticket history retrieval, or another corpus.",
  };
}

export async function generate(state: State): Promise<Partial<State>> {
  const context = state.retrieved_context.join("\n\n");
  const prompt = `Answer the question using provided context and fallback notes.

Question: ${state.question}
Fallback notes: ${state.fallback_notes}
Context:
${context}

Constraints:
- cite sources in [source] format
- if uncertain, explicitly say what is missing`;
  const response = await llm.invoke(prompt);
  return { draft_answer: String(response.content) };
}

export async function selfCheck(state: State): Promise<Partial<State>> {
  const prompt = `Perform a quick self-check of the answer below.
If there are unsupported claims, revise to be more cautious.

Answer:
${state.draft_answer}`;
  const response = await llm.invoke(prompt);
  return { final_answer: String(response.content) };
}

export function afterGrade(state: State): "fallback" | "generate" {
  return state.context_grade === "insufficient" ? "fallback" : "generate";
}

export function buildGraph() {
  return new StateGraph(AgenticRAGState)
    .addNode("rewrite", rewriteQuery)
    .addNode("retrieve", retrieve)
    .addNode("grade", gradeContext)
    .addNode("fallback", maybeFallback)
    .addNode("generate", generate)
    .addNode("self_check", selfCheck)
    .addEdge("__start__", "rewrite")
    .addEdge("rewrite", "retrieve")
    .addEdge("retrieve", "grade")
    .addConditionalEdges("grade", afterGrade, {
      fallback: "fallback",
      generate: "generate",
    })
    .addEdge("fallback", "generate")
    .addEdge("generate", "self_check")
    .addEdge("self_check", END)
    .compile();
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("RAG AGENTIC — CRAG/SELF-RAG STYLE LOOP");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required.");
    return;
  }

  const app = buildGraph();
  const question =
    "We see payment API latency for enterprise customers after a release. What should support do first?";
  const result = await app.invoke({ question });

  console.log("\n🔄 Rewritten query:");
  console.log(result.rewritten_query);

  console.log("\n📚 Retrieved context:");
  for (const line of result.retrieved_context) {
    console.log("-", line);
  }

  console.log("\n🧪 Context grade:", result.context_grade);
  console.log("🛟 Fallback:", result.fallback_notes);

  console.log("\n✅ Final answer:");
  console.log(result.final_answer);
}

if (import.meta.main) {
  await main();
}
