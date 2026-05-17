# Routing — TypeScript

**Direct port of `../src/routing.py`.** Same travel-support routing scenario, same coordinator prompt, same handler split, and the same `booker` / `info` / `unclear` branching logic.

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
| `RunnableBranch(...)` | `new RunnableBranch(...)` |
| `RunnablePassthrough.assign(...)` | `RunnablePassthrough.assign(...)` |
| dict LCEL composition | `RunnableMap.from(...)` composition |

If the Python source changes, update this port in the same PR.
