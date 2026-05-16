# Structured Outputs Pattern

Shows why schema-constrained output is a core reliability primitive.

- `src/structured_outputs_basic.py`: naive prompt parsing vs `with_structured_output(PydanticModel)`
- `src/structured_outputs_advanced.py`: Instructor retries + malformed input failure mode

Run:

```bash
uv sync
bash run.sh
```

If you're behind a corporate SSL-inspecting proxy:

```bash
AGENTIC_DISABLE_SSL=1 bash run.sh
```
