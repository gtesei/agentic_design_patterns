"""
Tool Use Pattern: Strict-Schema + Real API (Open-Meteo)

Anchor scenario: support-ops CRM triage.

This example demonstrates modern tool use in three parts:
1) Strict schema tools using Pydantic
2) Parallel tool execution for independent lookups
3) Agent runtime with langchain.agents.create_agent
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

import requests
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@dataclass
class CustomerCase:
    """Simple support case payload used for demo runs."""

    customer_id: str
    city: str
    issue: str


MOCK_CRM: dict[str, dict[str, Any]] = {
    "CUST-1001": {"tier": "enterprise", "open_tickets": 2, "contract_sla_hours": 2},
    "CUST-1002": {"tier": "pro", "open_tickets": 1, "contract_sla_hours": 8},
    "CUST-1003": {"tier": "starter", "open_tickets": 0, "contract_sla_hours": 24},
}


class CRMInput(BaseModel):
    """Strict schema for CRM lookups."""

    customer_id: str = Field(..., description="Customer identifier, e.g. CUST-1001")


class GeoInput(BaseModel):
    """Strict schema for geocoding lookups."""

    city: str = Field(..., min_length=2, description="City name")


class WeatherInput(BaseModel):
    """Strict schema for weather lookups."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


@tool(args_schema=CRMInput)
def get_customer_profile(customer_id: str) -> dict[str, Any]:
    """Fetch support profile data from CRM."""
    profile = MOCK_CRM.get(customer_id)
    if not profile:
        return {"error": f"customer_id {customer_id!r} not found"}
    return {"customer_id": customer_id, **profile}


@tool(args_schema=GeoInput)
def geocode_city(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates using Open-Meteo geocoding API."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(city)}&count=1&language=en&format=json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        return {"error": f"no location match for {city!r}"}

    best = results[0]
    return {
        "city": city,
        "latitude": best["latitude"],
        "longitude": best["longitude"],
        "country": best.get("country"),
    }


@tool(args_schema=WeatherInput)
def get_current_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Fetch current weather from Open-Meteo forecast API."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current=temperature_2m,precipitation,wind_speed_10m"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})

    return {
        "temperature_c": current.get("temperature_2m"),
        "precipitation_mm": current.get("precipitation"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "observation_time": current.get("time"),
    }


def run_parallel_enrichment(case: CustomerCase) -> dict[str, Any]:
    """Run independent data enrichments in parallel."""
    print("\n🔀 Running parallel enrichment (CRM + geocoding)...")

    with ThreadPoolExecutor(max_workers=2) as pool:
        crm_future = pool.submit(get_customer_profile.invoke, {"customer_id": case.customer_id})
        geo_future = pool.submit(geocode_city.invoke, {"city": case.city})
        crm = crm_future.result()
        geo = geo_future.result()

    weather: dict[str, Any]
    if "error" in geo:
        weather = {"error": "weather skipped due to geocoding failure"}
    else:
        weather = get_current_weather.invoke({"latitude": geo["latitude"], "longitude": geo["longitude"]})

    enrichment = {"crm": crm, "geo": geo, "weather": weather}
    print("✅ Parallel enrichment complete")
    return enrichment


def run_agentic_response(case: CustomerCase, enrichment: dict[str, Any]) -> str:
    """Use create_agent runtime to produce a support action plan."""
    agent = create_agent(
        model=get_default_model(),
        tools=[get_customer_profile, geocode_city, get_current_weather],
        system_prompt=(
            "You are a support-ops assistant. "
            "Return concise triage actions with clear SLA and escalation recommendation."
        ),
    )

    user_prompt = f"""
Case:
- customer_id: {case.customer_id}
- city: {case.city}
- issue: {case.issue}

Pre-fetched enrichment:
{enrichment}

Write a triage note with:
1) severity (low/medium/high)
2) next actions (max 3 bullets)
3) whether to escalate now
"""

    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    messages = result.get("messages", [])
    if not messages:
        return "No response from agent."
    return messages[-1].content


def main() -> None:
    print("\n" + "=" * 80)
    print("TOOL USE PATTERN — STRICT SCHEMA + PARALLEL + REAL API")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required for the agent runtime demo.")
        return

    case = CustomerCase(
        customer_id="CUST-1001",
        city="Milan",
        issue="Payment dashboard latency spikes affecting invoice approvals.",
    )

    enrichment = run_parallel_enrichment(case)
    print("\n📦 Enrichment payload:")
    print(enrichment)

    print("\n🤖 Generating agent triage note...")
    triage_note = run_agentic_response(case, enrichment)

    print("\n📝 Final triage note:")
    print(triage_note)


if __name__ == "__main__":
    main()
