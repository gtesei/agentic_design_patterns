import { handleRoute, routeQuery, type Route } from "./routing_basic";

function lexicalScore(query: string, keywords: string[]): number {
  const q = query.toLowerCase();
  return keywords.reduce((acc, k) => acc + (q.includes(k) ? 1 : 0), 0);
}

export function cascadeRoute(query: string): Route {
  // Stage 1: rule routing
  const rule = routeQuery(query);
  if (rule !== "support") return rule;

  // Stage 2: lightweight lexical scoring fallback
  const scores: Record<Route, number> = {
    support: lexicalScore(query, ["help", "access", "login"]),
    billing: lexicalScore(query, ["invoice", "refund", "charge", "payment"]),
    engineering: lexicalScore(query, ["latency", "bug", "api", "incident"]),
  };
  return (Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0] as Route) || "support";
}

if (import.meta.main) {
  const queries = [
    "I need a refund for an accidental charge",
    "Our API latency jumped to 2s p95",
    "I cannot access my account settings",
  ];
  console.log("\n=== Routing (TypeScript) — Advanced Cascade ===");
  for (const q of queries) {
    const route = cascadeRoute(q);
    console.log(handleRoute(route, q));
  }
}
