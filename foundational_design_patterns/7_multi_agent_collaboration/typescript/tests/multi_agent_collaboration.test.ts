import { describe, expect, test } from "bun:test";

import {
  cleanJsonBlock,
  decideAgent,
} from "../src/research_report_agent.ts";
import { routeWorker, trySupervisorPackage } from "../src/orchestrator_worker.ts";
import { llmError } from "../src/utils.ts";

describe("multi-agent collaboration (TS) — pure helpers", () => {
  test("cleanJsonBlock unwraps fenced JSON", () => {
    expect(cleanJsonBlock('```json\n{"agent":"writer_agent"}\n```')).toBe(
      '{"agent":"writer_agent"}',
    );
  });

  test("routeWorker chooses worker while tasks remain", () => {
    expect(
      routeWorker({
        goal: "goal",
        work_items: ["task"],
        results: [],
        final_report: "",
      }),
    ).toBe("worker");
  });

  test("llmError returns LLM-friendly error payload", () => {
    expect(llmError("boom")).toEqual([{ error: "boom" }]);
  });
});

describe("multi-agent collaboration (TS) — module shape", () => {
  test("trySupervisorPackage resolves to boolean", async () => {
    expect(typeof (await trySupervisorPackage())).toBe("boolean");
  });

  test("decideAgent is exported as a function", () => {
    expect(typeof decideAgent).toBe("function");
  });
});
