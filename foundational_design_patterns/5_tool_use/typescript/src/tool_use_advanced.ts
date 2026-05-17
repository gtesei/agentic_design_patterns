/**
 * Tool Use Pattern: Advanced strict-schema orchestration
 * — TypeScript port of src/tool_use_advanced.py.
 *
 * Anchor scenario: coding-agent file operations + support context enrichment.
 *
 * Highlights:
 *   - Zod-constrained tool arguments (Pydantic equivalent)
 *   - Parallel calls for independent tools
 *   - Deterministic fallback handling when a tool fails
 */

import { createAgent } from "langchain";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const DEFAULT_MODEL = "gpt-4o-mini";
const ADVANCED_MODEL_NAME = "gpt-5.2";

function getDefaultModel(): string {
  return process.env.OPENAI_MODEL ?? DEFAULT_MODEL;
}

function getAdvancedModel(): string {
  return (
    process.env.OPENAI_ADVANCED_MODEL ??
    process.env.OPENAI_MODEL ??
    ADVANCED_MODEL_NAME
  );
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/** Input payload for the advanced demo. */
export interface OpsRequest {
  repository: string;
  customer_id: string;
  task: string;
}

const FILE_INDEX: Record<string, string[]> = {
  agentic_design_patterns: [
    "README.md",
    "scripts/run_demos_smoke.sh",
    "repo_support.py",
    "ssl_fix.py",
  ],
  sample_backend: ["main.py", "routes.py", "requirements.txt"],
};

const CUSTOMER_RISK: Record<
  string,
  { incident_count_30d: number; risk_level: string }
> = {
  "CUST-1001": { incident_count_30d: 4, risk_level: "high" },
  "CUST-1002": { incident_count_30d: 1, risk_level: "medium" },
  "CUST-1003": { incident_count_30d: 0, risk_level: "low" },
};

// ---------------------------------------------------------------------------
// Strict schemas
// ---------------------------------------------------------------------------

const RepoInput = z.object({
  repository: z.string().describe("Repository slug"),
});

const PatternInput = z.object({
  query: z.string().min(5).describe("Search phrase for code files"),
});

const RiskInput = z.object({
  customer_id: z.string().describe("Customer identifier"),
});

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export const list_repo_files = tool(
  async ({ repository }) => {
    const files = FILE_INDEX[repository];
    if (files === undefined) {
      return { error: `unknown repository '${repository}'` };
    }
    return { repository, files };
  },
  {
    name: "list_repo_files",
    description: "List key files known for a repository.",
    schema: RepoInput,
  },
);

export const search_repo_for_pattern = tool(
  async ({ query }) => {
    const mapping: Record<string, string[]> = {
      ssl: ["ssl_fix.py", "repo_support.py"],
      demo: ["scripts/run_demos_smoke.sh"],
      model: ["repo_support.py"],
    };
    const lowered = query.toLowerCase();
    for (const [key, hits] of Object.entries(mapping)) {
      if (lowered.includes(key)) {
        return { query, hits };
      }
    }
    return { query, hits: [] };
  },
  {
    name: "search_repo_for_pattern",
    description: "Fake semantic search over repository files for deterministic demos.",
    schema: PatternInput,
  },
);

export const get_customer_risk_profile = tool(
  async ({ customer_id }) => {
    const profile = CUSTOMER_RISK[customer_id];
    if (profile === undefined) {
      return { error: `unknown customer_id '${customer_id}'` };
    }
    return { customer_id, ...profile };
  },
  {
    name: "get_customer_risk_profile",
    description: "Fetch a synthetic risk profile used by support operations.",
    schema: RiskInput,
  },
);

// ---------------------------------------------------------------------------
// Parallel context
// ---------------------------------------------------------------------------

/** Fetch independent context in parallel. */
export async function parallelContext(req: OpsRequest): Promise<{
  repo: unknown;
  search: unknown;
  risk: unknown;
}> {
  const [repo, search, risk] = await Promise.all([
    list_repo_files.invoke({ repository: req.repository }),
    search_repo_for_pattern.invoke({ query: req.task }),
    get_customer_risk_profile.invoke({ customer_id: req.customer_id }),
  ]);
  return { repo, search, risk };
}

// ---------------------------------------------------------------------------
// Agent composition
// ---------------------------------------------------------------------------

/** Use createAgent to synthesize an action plan from tool outputs. */
export async function composeOpsPlan(
  req: OpsRequest,
  context: { repo: unknown; search: unknown; risk: unknown },
): Promise<string> {
  const agent = createAgent({
    model: getAdvancedModel(),
    tools: [list_repo_files, search_repo_for_pattern, get_customer_risk_profile],
    systemPrompt:
      "You are a reliability engineer. " +
      "Generate a short support-ops playbook with concrete next steps. " +
      "If context data is missing, explicitly mark assumptions.",
  });

  const prompt =
    `Request:\n${JSON.stringify(req, null, 2)}\n\n` +
    `Parallel context:\n${JSON.stringify(context, null, 2)}\n\n` +
    "Return JSON with keys: severity, immediate_actions, owner, escalation.";

  const result = await agent.invoke({
    messages: [{ role: "user", content: prompt }],
  });
  const messages =
    (result as { messages?: Array<{ content: unknown }> }).messages ?? [];
  if (messages.length === 0) return "{}";
  const last = messages[messages.length - 1]!.content;
  return typeof last === "string" ? last : JSON.stringify(last);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("TOOL USE ADVANCED — STRICT SCHEMA ORCHESTRATION");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required for this example.");
    return;
  }

  // Silence unused-warning for getDefaultModel — advanced demo uses getAdvancedModel.
  void getDefaultModel;

  const req: OpsRequest = {
    repository: "agentic_design_patterns",
    customer_id: "CUST-1001",
    task: "Find SSL and demo smoke related code paths and suggest mitigation.",
  };

  const context = await parallelContext(req);
  console.log("\n🔎 Parallel context:");
  console.log(JSON.stringify(context, null, 2));

  console.log("\n🤖 Building action plan...");
  const plan = await composeOpsPlan(req, context);

  console.log("\n📋 Action plan output:");
  console.log(plan);
}

if (import.meta.main) {
  await main();
}
