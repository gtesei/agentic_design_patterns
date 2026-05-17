// Smoke tests — verify the graph compiles and routing logic is correct.
// No LLM calls (planner/executor nodes are not invoked).

import { describe, expect, test } from "bun:test";
import {
  PlanActState,
  buildGraph,
  reviewerNode,
  routeAfterReview,
} from "../src/planning_plan_and_act.ts";

const baseState = {
  incident: "Payment API error rate jumped to 35% after deployment.",
  context: "Service: billing-gateway, Region: eu-west",
  plan: [] as string[],
  completed: [] as string[],
  current_action: "",
  notes: [] as string[],
  requires_replan: false,
};

describe("planning (TS) — module shape", () => {
  test("PlanActState annotation root is defined", () => {
    expect(PlanActState).toBeDefined();
  });

  test("buildGraph returns a compiled graph with invoke()", () => {
    const graph = buildGraph();
    expect(typeof graph.invoke).toBe("function");
    expect(typeof graph.stream).toBe("function");
  });

  test("graph exposes planner, executor, reviewer nodes", () => {
    const graph = buildGraph();
    const nodeIds = Object.keys(graph.getGraph().nodes);
    expect(nodeIds).toContain("planner");
    expect(nodeIds).toContain("executor");
    expect(nodeIds).toContain("reviewer");
  });
});

describe("planning (TS) — reviewer + routing (pure functions)", () => {
  test("reviewer notes 'requesting replanning' when requires_replan", () => {
    const out = reviewerNode({ ...baseState, requires_replan: true });
    expect(out.notes?.at(-1)).toContain("requesting replanning");
  });

  test("reviewer notes 'continue' when plan has remaining steps", () => {
    const out = reviewerNode({ ...baseState, plan: ["check logs"] });
    expect(out.notes?.at(-1)).toContain("continue");
  });

  test("reviewer notes 'plan complete' when plan is empty", () => {
    const out = reviewerNode(baseState);
    expect(out.notes?.at(-1)).toContain("plan complete");
  });

  test("routeAfterReview returns 'planner' on replan request", () => {
    expect(routeAfterReview({ ...baseState, requires_replan: true })).toBe(
      "planner",
    );
  });

  test("routeAfterReview returns 'executor' when steps remain", () => {
    expect(routeAfterReview({ ...baseState, plan: ["check logs"] })).toBe(
      "executor",
    );
  });

  test("routeAfterReview returns END when plan is empty", () => {
    // END is the langgraph END sentinel string "__end__".
    expect(routeAfterReview(baseState)).toBe("__end__");
  });
});
