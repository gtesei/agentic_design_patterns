import { describe, expect, test } from "bun:test";
import {
  buildParallelChain,
  createQuestionsChain,
  createSummarizeChain,
  createTermsChain,
  processTopicAsync,
} from "../src/parallelization.ts";

describe("parallelization (TS) — module shape", () => {
  test("exports individual chains", () => {
    expect(typeof createSummarizeChain().invoke).toBe("function");
    expect(typeof createQuestionsChain().invoke).toBe("function");
    expect(typeof createTermsChain().invoke).toBe("function");
  });

  test("buildParallelChain is invokable", () => {
    expect(typeof buildParallelChain().invoke).toBe("function");
  });

  test("processTopicAsync accepts a stub chain", () => {
    const stubChain = {
      invoke: async (topic: string) => `processed: ${topic}`,
    };
    expect(processTopicAsync("a", stubChain as never)).toBeInstanceOf(Promise);
  });
});
