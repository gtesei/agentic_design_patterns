# Evaluation Template (P5 Scaffold)

Use this template in each pattern directory.

## Required assets

1. `tests/test_smoke.py`
   - Imports target modules from `src/`
   - Asserts top-level symbols only
   - Must not invoke live LLM calls

2. `README.md` section: **What good looks like**
   - Expected output shape
   - Common failure cases
   - Link to sample trace

3. `tests/sample_trace.md`
   - Happy-path structured trace excerpt

## Suggested command

```bash
uv sync
uv run pytest -q tests/test_smoke.py
```
