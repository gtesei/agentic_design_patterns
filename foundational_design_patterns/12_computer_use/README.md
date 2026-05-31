# Computer Use Pattern

Browser/UI automation framing with explicit safety policy.

## Architecture

```mermaid
---
title: Computer Use — Screenshot · Think · Act · Observe
---
%%{init: {'look':'handDrawn','theme':'base','themeVariables':{'background':'#f5ecd9','primaryColor':'#ede0bd','primaryBorderColor':'#6b4423','primaryTextColor':'#3e2723','lineColor':'#6b4423','clusterBkg':'#efe5cd','clusterBorder':'#c5b393','fontFamily':'Caveat, Patrick Hand, cursive'}}}%%
flowchart LR
    Goal([find LLM info<br/>on Wikipedia])
    Done([result])

    subgraph loop ["control loop"]
        Snap[/screenshot/]
        Think[LLM reasoning]
        Act[click · type · fetch]
        Obs[/new page state/]
        Snap --> Think --> Act --> Obs --> Snap
    end

    Goal --> Snap
    Think -. "task complete" .-> Done
```

- `src/computer_use_basic.py`: screenshot-think-act-observe simulation on Wikipedia
- `src/computer_use_advanced.py`: optional Playwright execution + safety controls

Run:

```bash
uv sync
bash run.sh
```

If you're behind a corporate SSL-inspecting proxy:

```bash
AGENTIC_DISABLE_SSL=1 bash run.sh
```
