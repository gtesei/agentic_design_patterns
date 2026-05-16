import { z } from "zod";

const TicketSchema = z.object({
  customerId: z.string().min(1),
  city: z.string().min(2),
  issue: z.string().min(5),
});

const WeatherSchema = z.object({
  temperature_c: z.number().nullable().optional(),
  wind_speed_kmh: z.number().nullable().optional(),
  precipitation_mm: z.number().nullable().optional(),
});

export type Ticket = z.infer<typeof TicketSchema>;

export async function geocodeCity(city: string): Promise<{ latitude: number; longitude: number } | null> {
  const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`;
  const response = await fetch(url);
  const json = await response.json();
  const first = json?.results?.[0];
  if (!first) return null;
  return { latitude: first.latitude, longitude: first.longitude };
}

export async function getWeather(latitude: number, longitude: number) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,precipitation,wind_speed_10m`;
  const response = await fetch(url);
  const json = await response.json();
  return WeatherSchema.parse({
    temperature_c: json?.current?.temperature_2m ?? null,
    wind_speed_kmh: json?.current?.wind_speed_10m ?? null,
    precipitation_mm: json?.current?.precipitation ?? null,
  });
}

export function triageDecision(ticket: Ticket, weather: z.infer<typeof WeatherSchema> | null): string {
  const severe = /down|incident|latency|payment/i.test(ticket.issue);
  const weatherHint = weather?.precipitation_mm && weather.precipitation_mm > 20 ? "Potential weather-related infra risk. " : "";
  if (severe) return `${weatherHint}Severity=high. Escalate to incident channel and assign owner now.`;
  return `${weatherHint}Severity=medium. Handle in support queue with SLA tracking.`;
}

async function main() {
  console.log("\n=== Tool Use (TypeScript) — Basic ===");
  const parsed = TicketSchema.parse({
    customerId: "CUST-1001",
    city: "Milan",
    issue: "Payment dashboard latency spikes affecting invoice approvals",
  });

  try {
    const geo = await geocodeCity(parsed.city);
    const weather = geo ? await getWeather(geo.latitude, geo.longitude) : null;
    console.log("Ticket:", parsed);
    console.log("Weather:", weather);
    console.log("Decision:", triageDecision(parsed, weather));
  } catch (error) {
    console.log("Network/API unavailable, using fallback triage.");
    console.log("Decision:", triageDecision(parsed, null));
  }
}

if (import.meta.main) {
  await main();
}
