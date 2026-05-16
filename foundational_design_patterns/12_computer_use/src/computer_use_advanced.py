"""Computer Use advanced: optional Playwright execution + safety policy."""

from __future__ import annotations

import os
import sys

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI


def run_playwright_task() -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return "Playwright not installed; skipping real browser automation."

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://en.wikipedia.org/wiki/Artificial_intelligence", wait_until="domcontentloaded")
        title = page.title()
        heading = page.locator("#firstHeading").inner_text()
        browser.close()
    return f"Visited page title='{title}', heading='{heading}'"


def main() -> None:
    print("\n" + "=" * 80)
    print("COMPUTER USE — ADVANCED")
    print("=" * 80)

    safety_policy = {
        "allowlist_domains": ["wikipedia.org"],
        "blocked_actions": ["checkout", "payment", "account deletion"],
        "human_approval_required_for": ["form submit", "external navigation", "file download"],
    }
    print("\nSafety policy:")
    print(safety_policy)

    outcome = run_playwright_task()
    print("\nAutomation outcome:")
    print(outcome)

    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model=get_default_model(), temperature=0)
            summary = llm.invoke(f"Summarize this browser automation run in 3 bullets:\n{outcome}").content
            print("\nLLM summary:")
            print(summary)
        except Exception as exc:
            print("\nLLM summary unavailable:")
            print(exc)


if __name__ == "__main__":
    main()
