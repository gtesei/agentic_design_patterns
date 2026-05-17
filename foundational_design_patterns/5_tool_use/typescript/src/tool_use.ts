/**
 * Tool Use Pattern: Strict-Schema + Real API (Open-Meteo) — TS port of src/tool_use.py.
 *
 * Anchor scenario: support-ops CRM triage.
 *
 * This example demonstrates modern tool use in three parts:
 *   1) Strict schema tools using Zod (Pydantic equivalent)
 *   2) Parallel tool execution for independent lookups
 *   3) Agent runtime with langchain.agents.createAgent
 */

import { createAgent } from "langchain";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/** Simple support case payload used for demo runs. */
export interface CustomerCase {
  customer_id: string;
  city: string;
  issue: string;
}

interface CrmProfile {
  tier: string;
  open_tickets: number;
  contract_sla_hours: number;
}

const MOCK_CRM: Record<string, CrmProfile> = {
  "CUST-1001": { tier: "enterprise", open_tickets: 2, contract_sla_hours: 2 },
  "CUST-1002": { tier: "pro", open_tickets: 1, contract_sla_hours: 8 },
  "CUST-1003": { tier: "starter", open_tickets: 0, contract_sla_hours: 24 },
};

// ---------------------------------------------------------------------------
// Strict schemas (Pydantic equivalents)
// ---------------------------------------------------------------------------

const CRMInput = z.object({
  customer_id: z
    .string()
    .describe("Customer identifier, e.g. CUST-1001"),
});

const GeoInput = z.object({
  city: z.string().min(2).describe("City name"),
});

const WeatherInput = z.object({
  latitude: z.number().gte(-90).lte(90),
  longitude: z.number().gte(-180).lte(180),
});

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export const get_customer_profile = tool(
  async ({ customer_id }) => {
    const profile = MOCK_CRM[customer_id];
    if (!profile) {
      return { error: `customer_id '${customer_id}' not found` };
    }
    return { customer_id, ...profile };
  },
  {
    name: "get_customer_profile",
    description: "Fetch support profile data from CRM.",
    schema: CRMInput,
  },
);

export const geocode_city = tool(
  async ({ city }) => {
    const url =
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}` +
      `&count=1&language=en&format=json`;
    const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
    if (!response.ok) {
      throw new Error(`Geocoding HTTP ${response.status}`);
    }
    const payload = (await response.json()) as {
      results?: Array<{
        latitude: number;
        longitude: number;
        country?: string;
      }>;
    };
    const results = payload.results ?? [];
    if (results.length === 0) {
      return { error: `no location match for '${city}'` };
    }

    const best = results[0]!;
    return {
      city,
      latitude: best.latitude,
      longitude: best.longitude,
      country: best.country ?? null,
    };
  },
  {
    name: "geocode_city",
    description: "Resolve a city name to coordinates using Open-Meteo geocoding API.",
    schema: GeoInput,
  },
);

export const get_current_weather = tool(
  async ({ latitude, longitude }) => {
    const url =
      `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}` +
      `&current=temperature_2m,precipitation,wind_speed_10m`;
    const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
    if (!response.ok) {
      throw new Error(`Weather HTTP ${response.status}`);
    }
    const payload = (await response.json()) as {
      current?: {
        temperature_2m?: number;
        precipitation?: number;
        wind_speed_10m?: number;
        time?: string;
      };
    };
    const current = payload.current ?? {};

    return {
      temperature_c: current.temperature_2m ?? null,
      precipitation_mm: current.precipitation ?? null,
      wind_speed_kmh: current.wind_speed_10m ?? null,
      observation_time: current.time ?? null,
    };
  },
  {
    name: "get_current_weather",
    description: "Fetch current weather from Open-Meteo forecast API.",
    schema: WeatherInput,
  },
);

// ---------------------------------------------------------------------------
// Parallel enrichment
// ---------------------------------------------------------------------------

export interface EnrichmentPayload {
  crm: unknown;
  geo: unknown;
  weather: unknown;
}

/** Run independent data enrichments in parallel. */
export async function runParallelEnrichment(
  caseInput: CustomerCase,
): Promise<EnrichmentPayload> {
  console.log("\n🔀 Running parallel enrichment (CRM + geocoding)...");

  const [crm, geo] = await Promise.all([
    get_customer_profile.invoke({ customer_id: caseInput.customer_id }),
    geocode_city.invoke({ city: caseInput.city }),
  ]);

  let weather: unknown;
  if (typeof geo === "object" && geo !== null && "error" in geo) {
    weather = { error: "weather skipped due to geocoding failure" };
  } else {
    const geoObj = geo as { latitude: number; longitude: number };
    weather = await get_current_weather.invoke({
      latitude: geoObj.latitude,
      longitude: geoObj.longitude,
    });
  }

  const enrichment: EnrichmentPayload = { crm, geo, weather };
  console.log("✅ Parallel enrichment complete");
  return enrichment;
}

// ---------------------------------------------------------------------------
// Agentic response
// ---------------------------------------------------------------------------

/** Use createAgent runtime to produce a support action plan. */
export async function runAgenticResponse(
  caseInput: CustomerCase,
  enrichment: EnrichmentPayload,
): Promise<string> {
  const agent = createAgent({
    model: MODEL,
    tools: [get_customer_profile, geocode_city, get_current_weather],
    systemPrompt:
      "You are a support-ops assistant. " +
      "Return concise triage actions with clear SLA and escalation recommendation.",
  });

  const userPrompt = `
Case:
- customer_id: ${caseInput.customer_id}
- city: ${caseInput.city}
- issue: ${caseInput.issue}

Pre-fetched enrichment:
${JSON.stringify(enrichment, null, 2)}

Write a triage note with:
1) severity (low/medium/high)
2) next actions (max 3 bullets)
3) whether to escalate now
`;

  const result = await agent.invoke({
    messages: [{ role: "user", content: userPrompt }],
  });
  const messages = (result as { messages?: Array<{ content: unknown }> }).messages ?? [];
  if (messages.length === 0) {
    return "No response from agent.";
  }
  const last = messages[messages.length - 1]!.content;
  return typeof last === "string" ? last : JSON.stringify(last);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("TOOL USE PATTERN — STRICT SCHEMA + PARALLEL + REAL API");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required for the agent runtime demo.");
    return;
  }

  const caseInput: CustomerCase = {
    customer_id: "CUST-1001",
    city: "Milan",
    issue: "Payment dashboard latency spikes affecting invoice approvals.",
  };

  const enrichment = await runParallelEnrichment(caseInput);
  console.log("\n📦 Enrichment payload:");
  console.log(enrichment);

  console.log("\n🤖 Generating agent triage note...");
  const triageNote = await runAgenticResponse(caseInput, enrichment);

  console.log("\n📝 Final triage note:");
  console.log(triageNote);
}

if (import.meta.main) {
  await main();
}
