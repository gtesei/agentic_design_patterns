"""Computer Use basic: screenshot-think-act-observe loop (safe simulation)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example

configure_example(__file__)

import requests
from bs4 import BeautifulSoup


@dataclass
class Step:
    screenshot: str
    thought: str
    action: str
    observation: str


def fetch_wikipedia_summary(topic: str) -> str:
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    p = soup.select_one("#mw-content-text p")
    return p.get_text(strip=True) if p else "No summary paragraph found"


def main() -> None:
    print("\n" + "=" * 80)
    print("COMPUTER USE — BASIC LOOP (SIMULATED)")
    print("=" * 80)

    steps = []
    topic = "Large_language_model"

    summary = fetch_wikipedia_summary(topic)
    steps.append(
        Step(
            screenshot="home_page.png",
            thought="Need to gather infobox/summary for LLM topic.",
            action="Open wikipedia page for Large language model.",
            observation=summary[:180] + "...",
        )
    )

    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}")
        print("screenshot:", step.screenshot)
        print("thought:", step.thought)
        print("action:", step.action)
        print("observation:", step.observation)

    print("\nSecurity note: keep computer-use agents sandboxed and require human approval for side effects.")


if __name__ == "__main__":
    main()
