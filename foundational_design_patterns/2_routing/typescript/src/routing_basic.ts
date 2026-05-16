export type Route = "support" | "billing" | "engineering";

export function routeQuery(query: string): Route {
  if (/invoice|refund|payment|billing/i.test(query)) return "billing";
  if (/bug|error|latency|api|integration/i.test(query)) return "engineering";
  return "support";
}

export function handleRoute(route: Route, query: string): string {
  const handlers: Record<Route, string> = {
    support: "Support handler: acknowledge issue and collect context.",
    billing: "Billing handler: verify account and payment events.",
    engineering: "Engineering handler: open incident and inspect systems.",
  };
  return `Route=${route} | ${handlers[route]} | Query=${query}`;
}

if (import.meta.main) {
  const query = "Payment dashboard latency after recent release";
  const route = routeQuery(query);
  console.log("\n=== Routing (TypeScript) — Basic ===");
  console.log(handleRoute(route, query));
}
