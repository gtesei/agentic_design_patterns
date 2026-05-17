# Prompt Chaining — TypeScript

**Direct port of `../src/chain_prompt.py`.** Same scenario (hardware specifications extraction), same two-step structure, same example data. Uses LangChain.js LCEL (`prompt.pipe(llm).pipe(parser)` and `RunnableMap`) to mirror the Python pipe operator.

The lesson is unchanged: decomposition through sequential LLM calls where each step has a single responsibility and the second sees only the structured output of the first.

## Run it

```bash
bash run.sh
```

`run.sh` sources `OPENAI_API_KEY` from the repo-root `.env` and bootstraps the bun workspace on first run.

## Smoke test

```bash
bun test
```

Smoke tests verify module exports and that the chain builders return runnables — no LLM call.

## Notes on parity

| Python | TypeScript |
|---|---|
| `prompt \| llm \| StrOutputParser()` | `prompt.pipe(llm).pipe(new StringOutputParser())` |
| `{"specifications": extraction_chain} \| transformation_prompt \| llm \| parser` | `RunnableMap.from({ specifications: extractionChain }).pipe(transformationPrompt).pipe(llm).pipe(parser)` |
| `ChatPromptTemplate.from_messages([(role, text), ...])` | `ChatPromptTemplate.fromMessages([[role, text], ...])` |

If you change the Python source, update this file to match. When refactoring to anchor scenarios from `SCENARIOS.md`, do Python and TS in the same PR.
