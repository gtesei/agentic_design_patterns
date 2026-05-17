// Smoke tests — import-only, no LLM calls.
// The real demo lives in src/chain_prompt.ts and requires OPENAI_API_KEY.

import { describe, expect, test } from "bun:test";
import {
  createExtractionPrompt,
  createTransformationPrompt,
  buildExtractionChain,
  buildFullChain,
  processTextToJson,
  processMultipleTexts,
  EXAMPLE_TEXTS,
} from "../src/chain_prompt.ts";

describe("prompt_chain (TS) — module shape", () => {
  test("exports the two prompt builders", () => {
    expect(typeof createExtractionPrompt).toBe("function");
    expect(typeof createTransformationPrompt).toBe("function");
  });

  test("exports the chain builders", () => {
    expect(typeof buildExtractionChain).toBe("function");
    expect(typeof buildFullChain).toBe("function");
  });

  test("exports the execution helpers", () => {
    expect(typeof processTextToJson).toBe("function");
    expect(typeof processMultipleTexts).toBe("function");
  });

  test("exports the demo dataset", () => {
    expect(Array.isArray(EXAMPLE_TEXTS)).toBe(true);
    expect(EXAMPLE_TEXTS.length).toBe(3);
  });
});

describe("prompt_chain (TS) — chain construction (offline)", () => {
  test("buildExtractionChain returns a runnable with invoke()", () => {
    const chain = buildExtractionChain();
    expect(typeof chain.invoke).toBe("function");
  });

  test("buildFullChain returns a runnable with invoke()", () => {
    const chain = buildFullChain();
    expect(typeof chain.invoke).toBe("function");
  });
});
