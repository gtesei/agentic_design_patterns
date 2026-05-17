import { describe, expect, test } from "bun:test";
import { producerChain, reflectionChain } from "../src/reflection.ts";
import {
  BlogCritiqueSchema,
  createBlogWriterGraph,
  shouldContinue,
} from "../src/reflection_stateful_loop.ts";

describe("reflection (TS) — basic chain shape", () => {
  test("producer chain is invokable", async () => {
    expect(typeof producerChain.invoke).toBe("function");
    expect(typeof reflectionChain.invoke).toBe("function");
  });
});

describe("reflection (TS) — stateful graph shape", () => {
  test("blog critique schema validates expected shape", () => {
    const parsed = BlogCritiqueSchema.parse({
      overall_quality: 8,
      flow_score: 8,
      flow_issues: [],
      tone_score: 8,
      tone_issues: [],
      clarity_score: 8,
      clarity_issues: [],
      content_score: 8,
      content_issues: [],
      engagement_score: 8,
      engagement_issues: [],
      strengths: [],
      priority_improvements: [],
      suggested_additions: [],
      suggested_removals: [],
      is_publication_ready: true,
    });
    expect(parsed.is_publication_ready).toBe(true);
  });

  test("graph compiles", () => {
    const graph = createBlogWriterGraph();
    expect(typeof graph.invoke).toBe("function");
  });

  test("shouldContinue ends when approved", () => {
    expect(
      shouldContinue({
        topic: "x",
        target_audience: "y",
        tone: "z",
        word_count_target: 100,
        draft: "",
        critique: "",
        iteration: 1,
        issues_found: [],
        quality_score: 8.5,
        is_approved: true,
        max_iterations: 3,
        improvement_history: [],
      }),
    ).toBe("__end__");
  });
});
