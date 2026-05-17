import { describe, expect, test } from "bun:test";

import {
  parseBasicArgs,
  printContent,
} from "../src/hitl_basic.ts";
import {
  AdvancedHITLWorkflow,
  CheckpointType,
  RiskLevel,
  riskLevelFromScore,
  extractJsonBlock,
} from "../src/hitl_advanced.ts";
import { revisionCheckNode, routingLogic } from "../src/hitl_langgraph.ts";

describe("hitl basic (TS) — helpers", () => {
  test("parseBasicArgs reads scenario and auto decision flags", () => {
    expect(
      parseBasicArgs([
        "--scenario",
        "2",
        "--auto-decision",
        "approve",
        "--auto-feedback",
        "Ship it",
      ]),
    ).toEqual({
      scenario: 2,
      autoDecision: "approve",
      autoFeedback: "Ship it",
    });
  });

  test("printContent is exported", () => {
    expect(typeof printContent).toBe("function");
  });
});

describe("hitl advanced (TS) — risk helpers", () => {
  test("riskLevelFromScore maps thresholds like the Python demo", () => {
    expect(riskLevelFromScore(15)).toBe(RiskLevel.LOW);
    expect(riskLevelFromScore(45)).toBe(RiskLevel.MEDIUM);
    expect(riskLevelFromScore(75)).toBe(RiskLevel.HIGH);
  });

  test("extractJsonBlock unwraps fenced JSON", () => {
    expect(extractJsonBlock('```json\n{"score": 10}\n```')).toBe('{"score": 10}');
  });

  test("low-risk items auto-approve without prompting", async () => {
    const workflow = new AdvancedHITLWorkflow(
      {} as never,
      async () => {
        throw new Error("prompt should not be called for low-risk auto approval");
      },
    );

    const approved = await workflow.processWithCheckpoint(
      {
        risk_level: RiskLevel.LOW,
        checkpoint_type: CheckpointType.FINANCIAL,
        risk_factors: ["Low transaction amount: $10.00"],
        score: 10,
        requires_approval: false,
        justification: "demo",
      },
      { Amount: "$10.00" },
    );

    expect(approved).toBe(true);
    expect(workflow.auditLog[0]?.approver).toBe("system_auto");
  });
});

describe("hitl langgraph (TS) — routing", () => {
  test("routingLogic sends approved content to finalize", () => {
    expect(
      routingLogic({
        task: "task",
        generated_content: "",
        human_feedback: "",
        approval_status: "approved",
        revision_count: 0,
        conversation_history: [],
        final_output: "",
      }),
    ).toBe("finalize");
  });

  test("revisionCheckNode rejects when max revisions reached", () => {
    expect(
      revisionCheckNode({
        task: "task",
        generated_content: "",
        human_feedback: "",
        approval_status: "needs_revision",
        revision_count: 3,
        conversation_history: [],
        final_output: "",
      }),
    ).toEqual({
      approval_status: "rejected",
      human_feedback: "Maximum revisions (3) exceeded",
    });
  });
});
