/**
 * Parallelization Pattern — TypeScript port of src/parallelization.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableMap, RunnablePassthrough } from "@langchain/core/runnables";

const LLM_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";
const LLM_TEMPERATURE = 0.7;

const llm = new ChatOpenAI({
  model: LLM_MODEL,
  temperature: LLM_TEMPERATURE,
});

export function createSummarizeChain() {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Summarize the following topic concisely in 2-3 sentences:"],
    ["user", "{topic}"],
  ]);
  return prompt.pipe(llm).pipe(new StringOutputParser());
}

export function createQuestionsChain() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Generate three thought-provoking questions about the following topic. Format each question on a new line with a number.",
    ],
    ["user", "{topic}"],
  ]);
  return prompt.pipe(llm).pipe(new StringOutputParser());
}

export function createTermsChain() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Identify 5-10 key terms or concepts related to the following topic. Return them as a comma-separated list.",
    ],
    ["user", "{topic}"],
  ]);
  return prompt.pipe(llm).pipe(new StringOutputParser());
}

export function createSynthesisChain() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are synthesizing information about a topic. Based on the following:

Summary: {summary}

Related Questions:
{questions}

Key Terms: {key_terms}

Create a comprehensive, well-structured response that:
1. Integrates the summary with the key terms naturally
2. Addresses or contextualizes the questions raised
3. Provides additional insights or connections
4. Maintains clarity and coherence`,
    ],
    ["user", "Original topic: {topic}"],
  ]);
  return prompt.pipe(llm).pipe(new StringOutputParser());
}

export function buildParallelChain() {
  const summarizeChain = createSummarizeChain();
  const questionsChain = createQuestionsChain();
  const termsChain = createTermsChain();

  const mapChain = RunnableMap.from<string>({
    summary: async (topic: string) =>
      await summarizeChain.invoke({ topic }),
    questions: async (topic: string) =>
      await questionsChain.invoke({ topic }),
    key_terms: async (topic: string) =>
      await termsChain.invoke({ topic }),
    topic: new RunnablePassthrough(),
  });

  const synthesisChain = createSynthesisChain();
  return mapChain.pipe(synthesisChain).pipe(new StringOutputParser());
}

export type ProcessResult = {
  topic: string;
  response: string | null;
  status: "success" | "error";
  error?: string;
};

export async function processTopicAsync(
  topic: string,
  chain = buildParallelChain(),
): Promise<ProcessResult> {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`Processing Topic: '${topic}'`);
  console.log(`${"=".repeat(70)}\n`);

  try {
    const response = await chain.invoke(topic);
    console.log("\n--- Final Synthesized Response ---");
    console.log(response);
    console.log(`\n${"=".repeat(70)}\n`);
    return { topic, response, status: "success" };
  } catch (e: unknown) {
    const errorMsg = `Chain execution failed: ${
      e instanceof Error ? e.message : String(e)
    }`;
    console.log(`\n❌ ERROR: ${errorMsg}\n`);
    return {
      topic,
      response: null,
      status: "error",
      error: errorMsg,
    };
  }
}

export async function processMultipleTopics(
  topics: string[],
): Promise<ProcessResult[]> {
  const chain = buildParallelChain();
  const tasks = topics.map((topic) => processTopicAsync(topic, chain));
  return await Promise.all(tasks);
}

export function processTopicSync(topic: string): Promise<ProcessResult> {
  return processTopicAsync(topic);
}

async function main(): Promise<void> {
  console.log("\n🚀 Running Single Topic Example");
  const singleTopic = "The history of space exploration";
  await processTopicSync(singleTopic);

  // console.log("\n🚀 Running Multiple Topics Example");
  // const multipleTopics = [
  //   "The history of space exploration",
  //   "Quantum computing fundamentals",
  //   "Climate change mitigation strategies",
  // ];
  // const results = await processMultipleTopics(multipleTopics);
  // const successful = results.filter((r) => r.status === "success").length;
  // console.log(`\n✅ Successfully processed ${successful}/${results.length} topics`);
}

if (import.meta.main) {
  await main();
}
