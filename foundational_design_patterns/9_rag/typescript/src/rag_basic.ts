/**
 * RAG Pattern: Hybrid retrieval baseline — TypeScript port of src/rag_basic.py.
 */

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

export interface Chunk {
  id: string;
  text: string;
  source: string;
}

export class HybridRetriever {
  chunks: Chunk[];
  embeddings: OpenAIEmbeddings;
  chunkVectors: number[][];
  tokenized: string[][];
  idf: Record<string, number>;

  private constructor(
    chunks: Chunk[],
    embeddings: OpenAIEmbeddings,
    chunkVectors: number[][],
    tokenized: string[][],
    idf: Record<string, number>,
  ) {
    this.chunks = chunks;
    this.embeddings = embeddings;
    this.chunkVectors = chunkVectors;
    this.tokenized = tokenized;
    this.idf = idf;
  }

  static async create(chunks: Chunk[]): Promise<HybridRetriever> {
    const embeddings = new OpenAIEmbeddings();
    const chunkVectors = await embeddings.embedDocuments(chunks.map((c) => c.text));
    const tokenized = chunks.map((c) => HybridRetriever.tokenize(c.text));
    const idf = HybridRetriever.buildIdf(tokenized);
    return new HybridRetriever(chunks, embeddings, chunkVectors, tokenized, idf);
  }

  static tokenize(text: string): string[] {
    return text.toLowerCase().match(/[a-zA-Z0-9_]+/g) ?? [];
  }

  static buildIdf(docs: string[][]): Record<string, number> {
    const df: Record<string, number> = {};
    const n = docs.length;
    for (const doc of docs) {
      for (const tok of new Set(doc)) {
        df[tok] = (df[tok] ?? 0) + 1;
      }
    }
    return Object.fromEntries(
      Object.entries(df).map(([tok, freq]) => [tok, Math.log((n + 1) / (freq + 1)) + 1]),
    );
  }

  static dot(a: Iterable<number>, b: Iterable<number>): number {
    let total = 0;
    const arrB = [...b];
    let i = 0;
    for (const x of a) {
      total += x * (arrB[i] ?? 0);
      i += 1;
    }
    return total;
  }

  bm25ishScore(queryTokens: string[], docTokens: string[]): number {
    let score = 0;
    const tf: Record<string, number> = {};
    for (const tok of docTokens) {
      tf[tok] = (tf[tok] ?? 0) + 1;
    }
    for (const tok of queryTokens) {
      if (tf[tok]) {
        score += (1 + Math.log(tf[tok]!)) * (this.idf[tok] ?? 1);
      }
    }
    return score;
  }

  async denseRank(query: string): Promise<Array<[number, number]>> {
    const qv = await this.embeddings.embedQuery(query);
    const scored = this.chunkVectors.map((dv, idx) => [idx, HybridRetriever.dot(qv, dv)] as [number, number]);
    return scored.sort((a, b) => b[1] - a[1]);
  }

  lexicalRank(query: string): Array<[number, number]> {
    const qTokens = HybridRetriever.tokenize(query);
    const scored = this.tokenized.map((dt, idx) => [idx, this.bm25ishScore(qTokens, dt)] as [number, number]);
    return scored.sort((a, b) => b[1] - a[1]);
  }

  async retrieve(query: string, topK = 5): Promise<Array<[Chunk, number]>> {
    const lexical = this.lexicalRank(query);
    const dense = await this.denseRank(query);

    const rrfScores: Record<number, number> = {};
    const k = 60.0;
    lexical.forEach(([idx], rankIdx) => {
      const rank = rankIdx + 1;
      rrfScores[idx] = (rrfScores[idx] ?? 0) + 1 / (k + rank);
    });
    dense.forEach(([idx], rankIdx) => {
      const rank = rankIdx + 1;
      rrfScores[idx] = (rrfScores[idx] ?? 0) + 1 / (k + rank);
    });

    const fused = Object.entries(rrfScores)
      .map(([idx, score]) => [Number(idx), score] as [number, number])
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK * 2);

    const queryTerms = new Set(HybridRetriever.tokenize(query));
    const reranked: Array<[Chunk, number]> = [];

    for (const [idx, baseScore] of fused) {
      const chunk = this.chunks[idx]!;
      const overlap = [...queryTerms].filter((term) => this.tokenized[idx]!.includes(term)).length;
      reranked.push([chunk, baseScore + 0.01 * overlap]);
    }

    reranked.sort((a, b) => b[1] - a[1]);
    return reranked.slice(0, topK);
  }
}

export function buildSupportChunks(): Chunk[] {
  const docs: Array<[string, string]> = [
    [
      "billing_guide",
      "Payment latency troubleshooting: check queue depth, retry backlog, and database write IOPS before rolling back.",
    ],
    [
      "incident_runbook",
      "For incident severity P1, page on-call immediately, assign incident commander, and publish customer status every 15 minutes.",
    ],
    [
      "sla_policy",
      "Enterprise tier customers require first response within 2 hours and escalation to senior support for payment pipeline incidents.",
    ],
    [
      "release_notes",
      "Release v2.8.4 introduced a new payment reconciliation worker with configurable concurrency and timeout settings.",
    ],
    [
      "db_observability",
      "If payment approval API latency exceeds 1.5 seconds p95, inspect lock contention and slow query logs.",
    ],
    [
      "ops_playbook",
      "During rollout failures, pause canary, compare error budgets, and decide rollback using incident commander approval.",
    ],
  ];

  return docs.map(([source, text], index) => ({
    id: `doc-${index + 1}`,
    source,
    text,
  }));
}

export async function answerQuery(
  query: string,
  retrieved: Array<[Chunk, number]>,
): Promise<string> {
  const llm = new ChatOpenAI({
    model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
    temperature: 0,
  });
  const context = retrieved.map(([chunk]) => `[${chunk.source}] ${chunk.text}`).join("\n\n");
  const prompt = `
You are a support documentation assistant.
Use only the context below. If information is missing, say so.

Context:
${context}

Question: ${query}
`;
  const response = await llm.invoke(prompt);
  return String(response.content);
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("RAG BASIC — HYBRID RETRIEVAL (LEXICAL + DENSE + RRF)");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required for this demo.");
    return;
  }

  const retriever = await HybridRetriever.create(buildSupportChunks());
  const query =
    "How should we handle payment API latency for an enterprise customer during incident response?";
  const retrieved = await retriever.retrieve(query, 5);

  console.log("\n🔍 Top-5 retrieved chunks:");
  retrieved.forEach(([chunk, score], index) => {
    console.log(`${index + 1}. score=${score.toFixed(4)} source=${chunk.source}`);
    console.log(`   ${chunk.text}`);
  });

  const answer = await answerQuery(query, retrieved);
  console.log("\n📝 Final answer:");
  console.log(answer);
}

if (import.meta.main) {
  await main();
}
