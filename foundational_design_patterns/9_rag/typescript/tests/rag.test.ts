import { describe, expect, test } from "bun:test";
import {
  buildSupportChunks,
  HybridRetriever,
} from "../src/rag_basic.ts";
import { recallAtK } from "../src/rag_eval_and_failure_modes.ts";
import { afterGrade, buildGraph } from "../src/rag_agentic.ts";

describe("rag (TS) — pure helpers", () => {
  test("buildSupportChunks returns the fixed support corpus", () => {
    const chunks = buildSupportChunks();
    expect(chunks.length).toBe(6);
    expect(chunks[0]?.source).toBe("billing_guide");
  });

  test("recallAtK matches keyword proxy logic", () => {
    expect(recallAtK("enterprise payment latency", ["enterprise", "latency"])).toBe(1);
  });

  test("agentic route after grade works", () => {
    expect(
      afterGrade({
        question: "x",
        rewritten_query: "",
        retrieved_context: [],
        context_grade: "insufficient",
        fallback_notes: "",
        draft_answer: "",
        final_answer: "",
      }),
    ).toBe("fallback");
  });

  test("agentic graph compiles", () => {
    const graph = buildGraph();
    expect(typeof graph.invoke).toBe("function");
  });

  test("hybrid retriever factory exists", () => {
    expect(typeof HybridRetriever.create).toBe("function");
  });
});
