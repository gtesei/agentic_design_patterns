import { expect, test } from "bun:test";
import { reactLoop } from "../src/react_basic";
import { multiStepReact } from "../src/react_advanced";

test("react basic returns trace", () => {
  const out = reactLoop("payment incident");
  expect(out).toContain("Thought:");
  expect(out).toContain("Action:");
});

test("react advanced returns multi-step output", () => {
  const out = multiStepReact("payment incident");
  expect(out).toContain("Step 1");
});
