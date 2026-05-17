# Tool Use — TypeScript

**Direct port of `../src/tool_use.py`.** Same scenario (support-ops CRM triage), same three tools, same flow: parallel pre-enrichment → agent runtime.

## What it does

Three Zod-typed tools (Pydantic equivalents in Python):

- `get_customer_profile(customer_id)` — mocked CRM lookup
- `geocode_city(city)` — real call to Open-Meteo geocoding API
- `get_current_weather(latitude, longitude)` — real call to Open-Meteo forecast API

Then `runParallelEnrichment` fans out CRM + geocoding concurrently with `Promise.all` (Python uses `ThreadPoolExecutor`), and `runAgenticResponse` calls `createAgent` (LangChain v1) with all three tools and produces a triage note.

## Run it

```bash
bash run.sh
```

## Smoke test

```bash
bun test
```

Tests cover the offline CRM tool and Zod schema enforcement. Geocoding/weather tests are skipped because they require live network.

## Notes on parity

| Python | TypeScript |
|---|---|
| `@tool(args_schema=Pydantic)` | `tool(fn, { schema: zodSchema })` |
| `ThreadPoolExecutor` + `pool.submit` | `Promise.all([...])` |
| `requests.get(url, timeout=20)` | `fetch(url, { signal: AbortSignal.timeout(20_000) })` |
| `create_agent(model=..., tools=..., system_prompt=...)` | `createAgent({ model, tools, systemPrompt })` |
