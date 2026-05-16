export type ToolResult = { observation: string };

export function searchDocs(query: string): ToolResult {
  const docs: Record<string, string> = {
    payment: "Runbook: check queue depth, DB locks, and recent deploy toggles.",
    sla: "Enterprise incidents require rapid response and escalation.",
  };
  const key = /payment/i.test(query) ? "payment" : "sla";
  return { observation: docs[key] };
}

export function reactLoop(question: string): string {
  const thought1 = "Need relevant operational guidance before acting.";
  const action1 = `searchDocs(${question})`;
  const obs1 = searchDocs(question).observation;

  const thought2 = "Now synthesize concise triage recommendation.";
  return [
    `Thought: ${thought1}`,
    `Action: ${action1}`,
    `Observation: ${obs1}`,
    `Thought: ${thought2}`,
    "Final: Escalate severity, inspect bottlenecks, communicate SLA updates.",
  ].join("\n");
}

if (import.meta.main) {
  console.log("\n=== ReAct (TypeScript) — Basic ===");
  console.log(reactLoop("payment incident for enterprise account"));
}
