/**
 * RAG Pattern: Evaluation and failure modes — TypeScript port of src/rag_eval_and_failure_modes.py.
 */

import { HybridRetriever, buildSupportChunks } from "./rag_basic.ts";

export interface EvalCase {
  name: string;
  question: string;
  expected_keywords: string[];
}

export function recallAtK(
  retrievedText: string,
  expectedKeywords: string[],
): number {
  const hits = expectedKeywords.filter((kw) =>
    retrievedText.toLowerCase().includes(kw.toLowerCase()),
  ).length;
  return hits / Math.max(1, expectedKeywords.length);
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("RAG EVAL — FAILURE MODES");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required (retriever uses OpenAI embeddings).");
    return;
  }

  const retriever = await HybridRetriever.create(buildSupportChunks());
  const cases: EvalCase[] = [
    {
      name: "normal_case",
      question:
        "How do we handle payment API latency for enterprise customers?",
      expected_keywords: ["enterprise", "incident", "latency", "escalation"],
    },
    {
      name: "lexical_mismatch",
      question:
        "What should we do about checkout sluggishness in premium accounts?",
      expected_keywords: ["enterprise", "payment", "latency"],
    },
    {
      name: "missing_coverage",
      question: "Which Kafka partition key should we use for ledger sharding?",
      expected_keywords: ["kafka", "partition", "ledger"],
    },
  ];

  for (const evalCase of cases) {
    const retrieved = await retriever.retrieve(evalCase.question, 5);
    const merged = retrieved.map(([chunk]) => chunk.text).join("\n");
    const metric = recallAtK(merged, evalCase.expected_keywords);

    console.log(`\n🧪 Case: ${evalCase.name}`);
    console.log(`Question: ${evalCase.question}`);
    console.log(`Recall@5 (keyword proxy): ${metric.toFixed(2)}`);

    if (metric < 0.5) {
      console.log("⚠️ Likely failure mode detected.");
      console.log(
        "   Mitigations: query rewrite, corpus expansion, reranker tuning, web fallback.",
      );
    }

    console.log("Top sources:");
    retrieved.forEach(([chunk, score], index) => {
      console.log(`  ${index + 1}. ${chunk.source} (score=${score.toFixed(4)})`);
    });
  }
}

if (import.meta.main) {
  await main();
}
