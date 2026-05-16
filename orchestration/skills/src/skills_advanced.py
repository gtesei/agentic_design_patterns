"""Skills advanced: LLM-assisted skill selection from metadata."""

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

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def load_metadata() -> list[dict]:
    items = []
    for path in sorted(SKILLS_DIR.glob("*.SKILL.md")):
        text = path.read_text(encoding="utf-8")
        description = next((l.split(":", 1)[1].strip() for l in text.splitlines() if "description:" in l), "")
        items.append({"skill": path.stem, "description": description, "path": str(path)})
    return items


def main() -> None:
    print("\n" + "=" * 80)
    print("SKILLS — ADVANCED LLM SKILL ROUTING")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    metadata = load_metadata()
    llm = ChatOpenAI(model=get_default_model(), temperature=0)

    task = "Review this Python snippet and suggest ruff-friendly lint fixes."
    try:
        decision = llm.invoke(f"Choose one skill for task: {task}\nAvailable: {metadata}\nReturn skill name only.").content.strip()
    except Exception as exc:
        print(f"LLM selection failed: {exc}")
        print("Falling back to deterministic selection...")
        decision = "python_linter"

    match = next((m for m in metadata if m["skill"] in decision), None)
    if not match:
        print("Could not confidently select skill. Metadata only:")
        print(metadata)
        return

    print(f"Selected skill: {match['skill']}")
    print("\nLoaded skill body:")
    print(Path(match["path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
