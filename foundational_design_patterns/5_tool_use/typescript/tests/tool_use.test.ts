// Smoke tests — exercise tool implementations directly (CRM is offline; geo/weather
// hit Open-Meteo and are skipped if the network blocks). No LLM calls.

import { describe, expect, test } from "bun:test";
import {
  get_customer_profile,
  geocode_city,
  get_current_weather,
  runParallelEnrichment,
  runAgenticResponse,
} from "../src/tool_use.ts";

describe("tool_use (TS) — module shape", () => {
  test("exports the three tools", () => {
    expect(get_customer_profile.name).toBe("get_customer_profile");
    expect(geocode_city.name).toBe("geocode_city");
    expect(get_current_weather.name).toBe("get_current_weather");
  });

  test("exports the enrichment + agent entry points", () => {
    expect(typeof runParallelEnrichment).toBe("function");
    expect(typeof runAgenticResponse).toBe("function");
  });
});

describe("tool_use (TS) — get_customer_profile (offline)", () => {
  test("returns profile for known customer_id", async () => {
    const out = await get_customer_profile.invoke({ customer_id: "CUST-1001" });
    expect(out).toMatchObject({
      customer_id: "CUST-1001",
      tier: "enterprise",
      open_tickets: 2,
      contract_sla_hours: 2,
    });
  });

  test("returns error for unknown customer_id", async () => {
    const out = await get_customer_profile.invoke({ customer_id: "CUST-9999" });
    expect(out).toMatchObject({
      error: expect.stringContaining("not found"),
    });
  });

  test("rejects invalid schema (missing field)", async () => {
    // Zod-validated tool: empty input must throw / reject at runtime.
    await expect(
      get_customer_profile.invoke({} as { customer_id: string }),
    ).rejects.toThrow();
  });
});
