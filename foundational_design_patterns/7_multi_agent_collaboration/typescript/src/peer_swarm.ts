/**
 * Multi-Agent Collaboration: Peer/Swarm topology
 * — TypeScript port of src/peer_swarm.py.
 */

import { ChatOpenAI } from "@langchain/openai";

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

export interface PeerState {
  problem: string;
  analyst_notes: string[];
  final_decision: string;
}

const llm = new ChatOpenAI({
  model: MODEL,
  temperature: 0,
});

export async function peerAnalysis(
  role: string,
  problem: string,
): Promise<string> {
  const response = await llm.invoke(
    `You are ${role}. Analyze this problem and propose one actionable recommendation:\n${problem}`,
  );
  return String(response.content ?? "");
}

export async function peerCritique(notes: string[]): Promise<string> {
  const prompt = `
You are a peer reviewer. Critique these peer recommendations and resolve conflicts.

${notes.join("\n")}

Return: conflict_summary + merged recommendation.
`;

  const response = await llm.invoke(prompt);
  return String(response.content ?? "");
}

export async function trySwarmPackage(): Promise<boolean> {
  try {
    const importer = new Function(
      "specifier",
      "return import(specifier);",
    ) as (specifier: string) => Promise<unknown>;
    await importer("langgraph-swarm");
    console.log(
      "✅ langgraph-swarm detected. (This demo uses portable fallback loop.)",
    );
    return true;
  } catch {
    console.log("ℹ️ langgraph-swarm not installed; using fallback peer loop.");
    return false;
  }
}

export async function runPeerSwarm(problem: string): Promise<PeerState> {
  const state: PeerState = {
    problem,
    analyst_notes: [],
    final_decision: "",
  };
  const roles = [
    "SRE analyst",
    "Support lead",
    "Product reliability manager",
  ];

  for (const role of roles) {
    state.analyst_notes.push(
      `[${role}]\n${await peerAnalysis(role, problem)}`,
    );
  }

  state.final_decision = await peerCritique(state.analyst_notes);
  return state;
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("MULTI-AGENT — PEER/SWARM");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required.");
    return;
  }

  await trySwarmPackage();

  const problem =
    "Multiple enterprise accounts report payment approval delays after release v2.8.4.";
  const state = await runPeerSwarm(problem);

  console.log("\n🤝 Peer notes:");
  for (const note of state.analyst_notes) {
    console.log(note);
    console.log("-".repeat(60));
  }

  console.log("\n✅ Final peer decision:");
  console.log(state.final_decision);
}

if (import.meta.main) {
  await main();
}
