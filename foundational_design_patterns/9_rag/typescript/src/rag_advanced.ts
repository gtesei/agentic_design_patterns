/**
 * RAG Pattern: Hosted file search + local hybrid fallback — TypeScript port of src/rag_advanced.py.
 */

import OpenAI from "openai";
import { ChatOpenAI } from "@langchain/openai";
import { HybridRetriever, answerQuery, buildSupportChunks } from "./rag_basic.ts";

export async function tryHostedFileSearch(query: string): Promise<string | null> {
  const vectorStoreId = process.env.OPENAI_VECTOR_STORE_ID;
  if (!vectorStoreId) return null;

  try {
    const client = new OpenAI();
    const response = await client.responses.create({
      model: process.env.OPENAI_ADVANCED_MODEL ?? process.env.OPENAI_MODEL ?? "gpt-5.2",
      tools: [{ type: "file_search", vector_store_ids: [vectorStoreId] }],
      input: `Use file_search on the support corpus and answer this question with citations: ${query}`,
    });
    return response.output_text;
  } catch (exc: unknown) {
    console.log(`⚠️ Hosted file_search unavailable: ${exc}`);
    return null;
  }
}

export async function localHybridAnswer(query: string): Promise<string> {
  const retriever = await HybridRetriever.create(buildSupportChunks());
  const retrieved = await retriever.retrieve(query, 5);
  return await answerQuery(query, retrieved);
}

export async function compareModels(query: string): Promise<[string, string]> {
  const retriever = await HybridRetriever.create(buildSupportChunks());
  const retrieved = await retriever.retrieve(query, 5);
  const context = retrieved.map(([chunk]) => `[${chunk.source}] ${chunk.text}`).join("\n\n");

  const lowCost = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const highCap = new ChatOpenAI({
    model: process.env.OPENAI_ADVANCED_MODEL ?? process.env.OPENAI_MODEL ?? "gpt-5.2",
    temperature: 0,
  });

  const prompt = `Answer with 3 bullets and cite source tags.\n\nContext:\n${context}\n\nQuestion: ${query}`;
  const low = await lowCost.invoke(prompt);
  const high = await highCap.invoke(prompt);
  return [String(low.content), String(high.content)];
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("RAG ADVANCED — HOSTED FILE_SEARCH + LOCAL HYBRID FALLBACK");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required.");
    return;
  }

  const query =
    "What are recommended actions for a payment incident affecting enterprise customers?";
  const hosted = await tryHostedFileSearch(query);
  if (hosted) {
    console.log("\n🌐 Hosted file_search answer:");
    console.log(hosted);
  } else {
    console.log("\n📚 Hosted file_search not configured. Using local hybrid fallback...");
    console.log(await localHybridAnswer(query));
  }

  console.log("\n🧪 Model comparison on the same retrieved context:");
  const [low, high] = await compareModels(query);
  console.log("\n- gpt-4o-mini output:\n", low);
  console.log("\n- advanced model output:\n", high);
}

if (import.meta.main) {
  await main();
}
