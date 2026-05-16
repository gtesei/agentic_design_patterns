import { z } from "zod";
import { geocodeCity, getWeather } from "./tool_use_basic";

const RequestSchema = z.object({
  repository: z.string(),
  customerId: z.string(),
  city: z.string(),
  task: z.string(),
});

export async function fetchRiskProfile(customerId: string) {
  const map: Record<string, { risk: "low" | "medium" | "high"; incidents30d: number }> = {
    "CUST-1001": { risk: "high", incidents30d: 4 },
    "CUST-1002": { risk: "medium", incidents30d: 1 },
  };
  return map[customerId] ?? { risk: "low", incidents30d: 0 };
}

export function synthesizePlan(input: z.infer<typeof RequestSchema>, risk: { risk: string; incidents30d: number }, weatherSummary: string) {
  return {
    severity: risk.risk === "high" ? "high" : "medium",
    immediate_actions: [
      "Create incident thread and assign incident commander",
      `Inspect ${input.repository} hot paths related to: ${input.task}`,
      "Validate SLA comms to affected customers",
    ],
    context: { risk, weatherSummary },
  };
}

async function main() {
  console.log("\n=== Tool Use (TypeScript) — Advanced ===");
  const input = RequestSchema.parse({
    repository: "agentic_design_patterns",
    customerId: "CUST-1001",
    city: "Milan",
    task: "payment latency in support dashboards",
  });

  const [risk, geo] = await Promise.all([fetchRiskProfile(input.customerId), geocodeCity(input.city)]);

  let weatherSummary = "weather unavailable";
  if (geo) {
    try {
      const weather = await getWeather(geo.latitude, geo.longitude);
      weatherSummary = `temp=${weather.temperature_c}C wind=${weather.wind_speed_kmh}km/h`;
    } catch {
      weatherSummary = "weather fetch failed";
    }
  }

  const plan = synthesizePlan(input, risk, weatherSummary);
  console.log(JSON.stringify(plan, null, 2));
}

if (import.meta.main) {
  await main();
}
