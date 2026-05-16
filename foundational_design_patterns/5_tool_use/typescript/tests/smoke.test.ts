import { expect, test } from "bun:test";
import { triageDecision } from "../src/tool_use_basic";
import { synthesizePlan } from "../src/tool_use_advanced";

test("tool_use_basic exports triageDecision", () => {
  const decision = triageDecision(
    { customerId: "C1", city: "Rome", issue: "payment down" },
    { temperature_c: 20, wind_speed_kmh: 10, precipitation_mm: 0 }
  );
  expect(decision.toLowerCase()).toContain("severity");
});

test("tool_use_advanced exports synthesizePlan", () => {
  const out = synthesizePlan(
    { repository: "repo", customerId: "C1", city: "Rome", task: "latency" },
    { risk: "high", incidents30d: 3 },
    "weather ok"
  );
  expect(out.immediate_actions.length).toBeGreaterThan(0);
});
