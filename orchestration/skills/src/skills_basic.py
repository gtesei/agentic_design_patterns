"""Skills basic: progressive disclosure (metadata first, full body on demand)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example

configure_example(__file__)


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def load_skill_metadata(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    name = lines[0].replace("# SKILL:", "").strip() if lines else path.stem
    description = next((l.split(":", 1)[1].strip() for l in lines if "description:" in l), "")
    triggers = next((l.split(":", 1)[1].strip() for l in lines if "triggers:" in l), "[]")
    return {"name": name, "description": description, "triggers": triggers, "path": str(path)}


def choose_skill(query: str, metadata: list[dict]) -> dict | None:
    lowered = query.lower()
    for item in metadata:
        if any(token in lowered for token in ["csv", "table", "markdown"]) and "csv_to_markdown" in item["name"]:
            return item
        if any(token in lowered for token in ["lint", "ruff", "python"]) and "python_linter" in item["name"]:
            return item
    return None


def main() -> None:
    print("\n" + "=" * 80)
    print("SKILLS — BASIC PROGRESSIVE DISCLOSURE")
    print("=" * 80)

    skill_files = sorted(SKILLS_DIR.glob("*.SKILL.md"))
    metadata = [load_skill_metadata(p) for p in skill_files]

    print("\nDiscovered skill metadata (lightweight):")
    for item in metadata:
        print(item)

    query = "Can you convert this CSV into markdown table format?"
    selected = choose_skill(query, metadata)
    print(f"\nQuery: {query}")

    if not selected:
        print("No skill selected.")
        return

    print(f"Selected skill: {selected['name']}")
    full_body = Path(selected["path"]).read_text(encoding="utf-8")
    print("\nLoaded full SKILL body only after selection:\n")
    print(full_body)


if __name__ == "__main__":
    main()
