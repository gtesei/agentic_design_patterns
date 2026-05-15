"""
Tool Use Pattern: Advanced strict-schema orchestration

Anchor scenario: coding-agent file operations + support context enrichment.

Highlights:
- Pydantic-constrained tool arguments
- Parallel calls for independent tools
- Deterministic fallback handling when a tool fails
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_advanced_model

configure_example(__file__)

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@dataclass
class OpsRequest:
    """Input payload for the advanced demo."""

    repository: str
    customer_id: str
    task: str


FILE_INDEX: dict[str, list[str]] = {
    "agentic_design_patterns": ["README.md", "scripts/run_demos_smoke.sh", "repo_support.py", "ssl_fix.py"],
    "sample_backend": ["main.py", "routes.py", "requirements.txt"],
}

CUSTOMER_RISK: dict[str, dict[str, Any]] = {
    "CUST-1001": {"incident_count_30d": 4, "risk_level": "high"},
    "CUST-1002": {"incident_count_30d": 1, "risk_level": "medium"},
    "CUST-1003": {"incident_count_30d": 0, "risk_level": "low"},
}


class RepoInput(BaseModel):
    repository: str = Field(..., description="Repository slug")


class PatternInput(BaseModel):
    query: str = Field(..., min_length=5, description="Search phrase for code files")


class RiskInput(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")


@tool(args_schema=RepoInput)
def list_repo_files(repository: str) -> dict[str, Any]:
    """List key files known for a repository."""
    files = FILE_INDEX.get(repository)
    if files is None:
        return {"error": f"unknown repository {repository!r}"}
    return {"repository": repository, "files": files}


@tool(args_schema=PatternInput)
def search_repo_for_pattern(query: str) -> dict[str, Any]:
    """Fake semantic search over repository files for deterministic demos."""
    mapping = {
        "ssl": ["ssl_fix.py", "repo_support.py"],
        "demo": ["scripts/run_demos_smoke.sh"],
        "model": ["repo_support.py"],
    }
    lowered = query.lower()
    for key, hits in mapping.items():
        if key in lowered:
            return {"query": query, "hits": hits}
    return {"query": query, "hits": []}


@tool(args_schema=RiskInput)
def get_customer_risk_profile(customer_id: str) -> dict[str, Any]:
    """Fetch a synthetic risk profile used by support operations."""
    profile = CUSTOMER_RISK.get(customer_id)
    if profile is None:
        return {"error": f"unknown customer_id {customer_id!r}"}
    return {"customer_id": customer_id, **profile}


def parallel_context(req: OpsRequest) -> dict[str, Any]:
    """Fetch independent context in parallel."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            "repo": pool.submit(list_repo_files.invoke, {"repository": req.repository}),
            "search": pool.submit(search_repo_for_pattern.invoke, {"query": req.task}),
            "risk": pool.submit(get_customer_risk_profile.invoke, {"customer_id": req.customer_id}),
        }
    return {name: future.result() for name, future in futures.items()}


def compose_ops_plan(req: OpsRequest, context: dict[str, Any]) -> str:
    """Use create_agent to synthesize an action plan from tool outputs."""
    agent = create_agent(
        model=get_advanced_model(),
        tools=[list_repo_files, search_repo_for_pattern, get_customer_risk_profile],
        system_prompt=(
            "You are a reliability engineer. "
            "Generate a short support-ops playbook with concrete next steps. "
            "If context data is missing, explicitly mark assumptions."
        ),
    )

    prompt = (
        f"Request:\n{req}\n\n"
        f"Parallel context:\n{json.dumps(context, indent=2)}\n\n"
        "Return JSON with keys: severity, immediate_actions, owner, escalation."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])
    return messages[-1].content if messages else "{}"


def main() -> None:
    print("\n" + "=" * 80)
    print("TOOL USE ADVANCED — STRICT SCHEMA ORCHESTRATION")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required for this example.")
        return

    req = OpsRequest(
        repository="agentic_design_patterns",
        customer_id="CUST-1001",
        task="Find SSL and demo smoke related code paths and suggest mitigation.",
    )

    context = parallel_context(req)
    print("\n🔎 Parallel context:")
    print(json.dumps(context, indent=2))

    print("\n🤖 Building action plan...")
    plan = compose_ops_plan(req, context)

    print("\n📋 Action plan output:")
    print(plan)


if __name__ == "__main__":
    main()
