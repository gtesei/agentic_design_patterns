# Parallelization — TypeScript

**Direct port of `../src/parallelization.py`.** Same three-way parallel fan-out, same synthesis step, same example topic, and the same optional multi-topic batch flow.

## Run it

```bash
bash run.sh
```

## Smoke test

```bash
bun test
```

## Notes on parity

| Python | TypeScript |
|---|---|
| `RunnableParallel({...})` | `RunnableMap.from({...})` parallel mapping |
| `await chain.ainvoke(topic)` | `await chain.invoke(topic)` |
| `asyncio.gather(...)` | `Promise.all(...)` |
