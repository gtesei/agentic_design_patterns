import { describe, expect, test } from "bun:test";
import {
  agent,
  calculator,
  getWordCount,
  search,
} from "../src/react_agent.ts";

describe("react (TS) — module shape", () => {
  test("basic tools are exported", () => {
    expect(search.name).toBe("search");
    expect(calculator.name).toBe("calculator");
    expect(getWordCount.name).toBe("get_word_count");
  });

  test("basic agent is invokable", () => {
    expect(typeof agent.invoke).toBe("function");
  });
});

describe("react advanced (TS) — graph shape", () => {
  test("buildGraph returns compiled graph", async () => {
    const mod = await import("../src/react_agent_advanced.ts");
    const graph = mod.buildGraph();
    expect(typeof graph.invoke).toBe("function");
  });

  test("shouldContinue ends when iteration limit reached", async () => {
    const mod = await import("../src/react_agent_advanced.ts");
    expect(
      mod.shouldContinue({
        messages: [],
        iteration: 10,
        max_iterations: 10,
      }),
    ).toBe("__end__");
  });
});
