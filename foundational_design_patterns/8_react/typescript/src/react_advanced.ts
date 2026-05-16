import { reactLoop } from "./react_basic";

export function multiStepReact(question: string): string {
  const plan = [
    "Gather evidence",
    "Cross-check SLA implications",
    "Produce action checklist",
  ];

  const loops = plan.map((step, i) => `--- Step ${i + 1}: ${step} ---\n${reactLoop(question)}`);
  return loops.join("\n\n");
}

if (import.meta.main) {
  console.log("\n=== ReAct (TypeScript) — Advanced ===");
  console.log(multiStepReact("enterprise payment latency after release"));
}
