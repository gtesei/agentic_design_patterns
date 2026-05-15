"""
MCP Pattern: Official protocol basics (stdio transport)

This file demonstrates a real MCP server + client handshake using the official
MCP Python SDK instead of a custom mock registry.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example

configure_example(__file__)


def _build_server():
    """Create MCP server handlers with official SDK APIs."""
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server("adp-mcp-basic")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="add",
                description="Add two integers",
                inputSchema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="read_policy",
                description="Return one support policy snippet",
                inputSchema={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None):
        args = arguments or {}

        if name == "add":
            a = int(args["a"])
            b = int(args["b"])
            return {"result": a + b}

        if name == "read_policy":
            topic = str(args.get("topic", "")).lower()
            if "enterprise" in topic:
                text = "Enterprise incidents require first response within 2 hours."
            else:
                text = "Default support SLA is next-business-day for non-critical issues."
            return {"topic": topic, "policy": text}

        raise ValueError(f"Unknown tool: {name}")

    return server


async def run_stdio_server() -> None:
    """Run MCP server on stdio transport."""
    from mcp.server.stdio import stdio_server

    server = _build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_client_demo() -> None:
    """Run a local client that connects to this script in server mode."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_params = StdioServerParameters(command="uv", args=["run", "python", "src/mcp_basic.py", "--mode", "server"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("\n🔧 list_tools result:")
            print(tools)

            add_result = await session.call_tool("add", {"a": 7, "b": 5})
            policy_result = await session.call_tool("read_policy", {"topic": "enterprise payment incident"})

            print("\n📌 call_tool(add) result:")
            print(add_result)

            print("\n📌 call_tool(read_policy) result:")
            print(policy_result)


def ensure_sdk_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except Exception:
        print("❌ MCP SDK not installed in this environment.")
        print("Install with: uv add mcp")
        return False


def parse_mode() -> str:
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return "demo"


def main() -> None:
    print("\n" + "=" * 80)
    print("MCP BASIC — OFFICIAL SDK + STDIO HANDSHAKE")
    print("=" * 80)

    if not ensure_sdk_available():
        return

    mode = parse_mode()
    if mode == "server":
        asyncio.run(run_stdio_server())
        return

    asyncio.run(run_client_demo())


if __name__ == "__main__":
    main()
