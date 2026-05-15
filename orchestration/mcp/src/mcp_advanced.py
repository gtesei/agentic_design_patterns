"""
MCP Pattern: Advanced sections in one script

Sections:
1) MCP basics recap (stdio)
2) Remote servers (streamable-http note + config)
3) Security & roots (filesystem allowlist)
4) MCP-with-agents orchestration
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI


ALLOWED_ROOTS = {
    "/workspace/docs": "Support docs root",
    "/workspace/runbooks": "Incident runbooks root",
}


def _build_secure_server():
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server("adp-mcp-advanced")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="read_root_file",
                description="Read a file from approved roots only",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="compute_priority",
                description="Compute simple incident priority based on impact and urgency",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "impact": {"type": "integer", "minimum": 1, "maximum": 10},
                        "urgency": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["impact", "urgency"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None):
        args = arguments or {}

        if name == "read_root_file":
            requested = str(args.get("path", ""))
            if not any(requested.startswith(root) for root in ALLOWED_ROOTS):
                return {
                    "error": "path outside allowed roots",
                    "allowed_roots": list(ALLOWED_ROOTS.keys()),
                }
            return {
                "path": requested,
                "content": "[mock-content] Incident runbook snippet: page on-call, isolate blast radius, post status.",
            }

        if name == "compute_priority":
            impact = int(args["impact"])
            urgency = int(args["urgency"])
            score = 0.6 * impact + 0.4 * urgency
            label = "P1" if score >= 8.5 else "P2" if score >= 6.5 else "P3"
            return {"score": score, "priority": label}

        raise ValueError(f"Unknown tool: {name}")

    return server


async def run_secure_server() -> None:
    from mcp.server.stdio import stdio_server

    server = _build_secure_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_agent_orchestration_demo() -> None:
    """Client calls MCP tools then asks an LLM to synthesize decision text."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command="uv", args=["run", "python", "src/mcp_advanced.py", "--mode", "server"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            priority = await session.call_tool("compute_priority", {"impact": 9, "urgency": 8})
            runbook = await session.call_tool("read_root_file", {"path": "/workspace/runbooks/payment_incident.md"})

    llm = ChatOpenAI(model=get_default_model(), temperature=0)
    synthesis = llm.invoke(
        f"""
You are incident commander support.
Use MCP outputs below and produce a concise action plan.

priority_tool_output={priority}
runbook_tool_output={runbook}
"""
    ).content

    print("\n🧩 MCP-with-agents synthesis:")
    print(synthesis)


def print_governance_note() -> None:
    print("\n📘 Governance note:")
    print("MCP is presented here as an open protocol standard (AIF / Linux Foundation governance context),")
    print("not as a proprietary single-vendor interface.")


def ensure_sdk_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except Exception:
        print("❌ MCP SDK not installed. Install with: uv add mcp")
        return False


def parse_mode() -> str:
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return "demo"


def main() -> None:
    print("\n" + "=" * 80)
    print("MCP ADVANCED — SECURITY ROOTS + MCP-WITH-AGENTS")
    print("=" * 80)

    if not ensure_sdk_available():
        return

    mode = parse_mode()
    if mode == "server":
        asyncio.run(run_secure_server())
        return

    print("\n🌐 Remote transport note: the same server can be exposed through streamable HTTP/SSE in production.")
    print("🔐 Allowed roots:", ALLOWED_ROOTS)

    print_governance_note()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not set; skipping MCP-with-agents synthesis.")
        return

    asyncio.run(run_agent_orchestration_demo())


if __name__ == "__main__":
    main()
