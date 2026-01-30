"""
Model Context Protocol (MCP): Basic Implementation

This example demonstrates a simplified implementation of the Model Context Protocol (MCP),
showing how to create an MCP server that exposes tools and resources through a standardized
protocol, and how an LLM client can discover and use these capabilities.

Key concepts:
- MCP Server: Exposes tools and resources through protocol
- Tool Discovery: Client dynamically finds available tools
- Tool Invocation: Execute tools through standardized interface
- Resource Access: Read data from exposed resources
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

console = Console()


# ═══════════════════════════════════════════════════════════════
# MCP Protocol - Simplified Implementation
# ═══════════════════════════════════════════════════════════════


@dataclass
class MCPTool:
    """Represents an MCP tool with metadata and execution logic"""

    name: str
    description: str
    inputSchema: dict
    execute: Callable


@dataclass
class MCPResource:
    """Represents an MCP resource (data source)"""

    uri: str
    name: str
    mimeType: str
    description: str
    content: Any


@dataclass
class SimpleMCPServer:
    """
    Simplified MCP Server implementation that exposes tools and resources.

    In a real MCP implementation, this would use JSON-RPC 2.0 over stdio/HTTP/WebSocket.
    This simplified version demonstrates the core concepts without the protocol complexity.
    """

    name: str
    tools: dict[str, MCPTool] = field(default_factory=dict)
    resources: dict[str, MCPResource] = field(default_factory=dict)
    invocation_log: list[dict] = field(default_factory=list)

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the server"""
        self.tools[tool.name] = tool
        console.print(f"[green]✓[/green] Registered tool: [cyan]{tool.name}[/cyan]")

    def register_resource(self, resource: MCPResource) -> None:
        """Register a resource with the server"""
        self.resources[resource.uri] = resource
        console.print(f"[green]✓[/green] Registered resource: [cyan]{resource.uri}[/cyan]")

    def list_tools(self) -> dict:
        """
        MCP Protocol: List all available tools

        Returns standardized tool descriptions that clients can use
        to understand capabilities.
        """
        return {
            "tools": [
                {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
                for tool in self.tools.values()
            ]
        }

    def invoke_tool(self, name: str, arguments: dict) -> dict:
        """
        MCP Protocol: Invoke a tool with given arguments

        Returns standardized response with result or error.
        """
        timestamp = datetime.now()

        if name not in self.tools:
            error_response = {"error": {"code": "TOOL_NOT_FOUND", "message": f"Tool '{name}' not found"}}
            self.invocation_log.append(
                {"timestamp": timestamp, "tool": name, "arguments": arguments, "success": False, "response": error_response}
            )
            return error_response

        tool = self.tools[name]

        try:
            # Execute the tool
            result = tool.execute(**arguments)

            response = {"result": result}
            self.invocation_log.append(
                {"timestamp": timestamp, "tool": name, "arguments": arguments, "success": True, "response": response}
            )

            return response

        except Exception as e:
            error_response = {
                "error": {"code": "EXECUTION_ERROR", "message": f"Tool execution failed: {str(e)}", "details": str(e)}
            }

            self.invocation_log.append(
                {"timestamp": timestamp, "tool": name, "arguments": arguments, "success": False, "response": error_response}
            )

            return error_response

    def list_resources(self) -> dict:
        """
        MCP Protocol: List all available resources
        """
        return {
            "resources": [
                {"uri": res.uri, "name": res.name, "mimeType": res.mimeType, "description": res.description}
                for res in self.resources.values()
            ]
        }

    def read_resource(self, uri: str) -> dict:
        """
        MCP Protocol: Read a resource's content
        """
        if uri not in self.resources:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": f"Resource '{uri}' not found"}}

        resource = self.resources[uri]
        return {"contents": [{"uri": uri, "mimeType": resource.mimeType, "text": resource.content}]}


# ═══════════════════════════════════════════════════════════════
# Example MCP Server with Tools
# ═══════════════════════════════════════════════════════════════


def create_file_system_server() -> SimpleMCPServer:
    """Create an MCP server with file system tools"""

    server = SimpleMCPServer(name="filesystem-server")

    # Simulated file system
    mock_files = {
        "/config/settings.json": json.dumps({"theme": "dark", "language": "en", "notifications": True, "autosave": True}, indent=2),
        "/data/users.txt": "alice@example.com\nbob@example.com\ncharlie@example.com",
        "/logs/app.log": "2024-01-30 10:00:00 INFO Application started\n2024-01-30 10:01:23 INFO User logged in\n2024-01-30 10:05:45 WARNING High memory usage",
    }

    # Tool 1: Read File
    def read_file_impl(path: str) -> str:
        """Read contents of a file"""
        if path in mock_files:
            return f"File: {path}\n\n{mock_files[path]}"
        return f"Error: File '{path}' not found"

    read_file_tool = MCPTool(
        name="read_file",
        description="Read the contents of a file from the file system. Supports text files, JSON, logs, etc.",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute path to the file to read"}},
            "required": ["path"],
        },
        execute=read_file_impl,
    )

    # Tool 2: List Files
    def list_files_impl(directory: str = "/") -> str:
        """List files in a directory"""
        files = [path for path in mock_files.keys() if path.startswith(directory)]
        if not files:
            return f"No files found in directory: {directory}"

        return f"Files in {directory}:\n" + "\n".join(f"  - {path}" for path in files)

    list_files_tool = MCPTool(
        name="list_files",
        description="List all files in a directory. Returns file paths.",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path to list (default: /)", "default": "/"}
            },
        },
        execute=list_files_impl,
    )

    # Tool 3: Calculator
    def calculator_impl(expression: str) -> str:
        """Evaluate a mathematical expression"""
        try:
            # In production, use a safer evaluation method
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {str(e)}"

    calculator_tool = MCPTool(
        name="calculator",
        description="Perform mathematical calculations. Supports basic arithmetic operations (+, -, *, /, **, %).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"}
            },
            "required": ["expression"],
        },
        execute=calculator_impl,
    )

    # Register tools
    server.register_tool(read_file_tool)
    server.register_tool(list_files_tool)
    server.register_tool(calculator_tool)

    # Register resources
    config_resource = MCPResource(
        uri="file:///config/settings.json",
        name="Application Settings",
        mimeType="application/json",
        description="Main configuration file for the application",
        content=mock_files["/config/settings.json"],
    )

    server.register_resource(config_resource)

    return server


# ═══════════════════════════════════════════════════════════════
# MCP Client - LangChain Integration
# ═══════════════════════════════════════════════════════════════


def convert_mcp_to_langchain_tools(mcp_server: SimpleMCPServer) -> list:
    """
    Convert MCP tools to LangChain tools.

    This allows seamless integration of MCP servers with LangChain agents.
    """

    langchain_tools = []

    for tool_info in mcp_server.list_tools()["tools"]:
        tool_name = tool_info["name"]
        tool_description = tool_info["description"]

        # Create a closure to capture the tool name
        def make_tool_func(name: str):
            def tool_func(**kwargs) -> str:
                result = mcp_server.invoke_tool(name, kwargs)
                if "result" in result:
                    return str(result["result"])
                elif "error" in result:
                    return f"Error: {result['error']['message']}"
                return str(result)

            return tool_func

        # Create LangChain tool
        langchain_tool = tool(make_tool_func(tool_name))
        langchain_tool.name = tool_name
        langchain_tool.description = tool_description

        langchain_tools.append(langchain_tool)

    return langchain_tools


# ═══════════════════════════════════════════════════════════════
# Visualization & Demo
# ═══════════════════════════════════════════════════════════════


def display_server_info(server: SimpleMCPServer):
    """Display server information and capabilities"""

    # Create a tree view of server capabilities
    tree = Tree(f"[bold cyan]MCP Server: {server.name}[/bold cyan]")

    # Tools section
    tools_branch = tree.add("[bold green]🔧 Tools[/bold green]")
    for tool_info in server.list_tools()["tools"]:
        tool_node = tools_branch.add(f"[cyan]{tool_info['name']}[/cyan]")
        tool_node.add(f"[dim]{tool_info['description']}[/dim]")

        # Show parameters
        if "properties" in tool_info["inputSchema"]:
            params = tool_info["inputSchema"]["properties"]
            params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
            tool_node.add(f"[yellow]Parameters:[/yellow] {params_str}")

    # Resources section
    resources_branch = tree.add("[bold blue]📦 Resources[/bold blue]")
    for res_info in server.list_resources()["resources"]:
        res_node = resources_branch.add(f"[cyan]{res_info['uri']}[/cyan]")
        res_node.add(f"[dim]{res_info['description']}[/dim]")
        res_node.add(f"[yellow]Type:[/yellow] {res_info['mimeType']}")

    console.print(Panel(tree, title="[bold]MCP Server Capabilities[/bold]", border_style="cyan"))


def display_invocation_log(server: SimpleMCPServer):
    """Display tool invocation history"""

    if not server.invocation_log:
        console.print("[yellow]No tool invocations yet[/yellow]")
        return

    table = Table(title="Tool Invocation Log", show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", style="dim")
    table.add_column("Tool", style="cyan")
    table.add_column("Arguments", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Result/Error", style="white")

    for entry in server.invocation_log[-5:]:  # Show last 5 invocations
        timestamp = entry["timestamp"].strftime("%H:%M:%S")
        tool = entry["tool"]
        arguments = json.dumps(entry["arguments"], indent=None)
        status = "✓ Success" if entry["success"] else "✗ Failed"

        response = entry["response"]
        if "result" in response:
            result_preview = str(response["result"])[:50] + "..." if len(str(response["result"])) > 50 else str(response["result"])
        else:
            result_preview = response["error"]["message"]

        table.add_row(timestamp, tool, arguments, status, result_preview)

    console.print(table)


# ═══════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════


def run_demo():
    """Run the MCP basic demonstration"""

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Model Context Protocol (MCP) - Basic Implementation[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Step 1: Create MCP Server
    console.print("[bold green]Step 1:[/bold green] Creating MCP Server with tools...\n")
    server = create_file_system_server()
    console.print()

    # Step 2: Display server capabilities
    console.print("[bold green]Step 2:[/bold green] Server capabilities registered\n")
    display_server_info(server)
    console.print()

    # Step 3: Tool discovery
    console.print("[bold green]Step 3:[/bold green] Client discovering available tools...\n")
    tools_list = server.list_tools()
    console.print(f"[cyan]Discovered {len(tools_list['tools'])} tools:[/cyan]")
    for tool_info in tools_list["tools"]:
        console.print(f"  • [cyan]{tool_info['name']}[/cyan] - {tool_info['description'][:60]}...")
    console.print()

    # Step 4: Direct tool invocation (simulating LLM decisions)
    console.print("[bold green]Step 4:[/bold green] Demonstrating direct tool invocations...\n")

    # Invocation 1: List files
    console.print("[yellow]→[/yellow] Invoking: [cyan]list_files[/cyan]")
    result1 = server.invoke_tool("list_files", {"directory": "/"})
    console.print(f"[green]Result:[/green]\n{result1['result']}\n")

    # Invocation 2: Read config file
    console.print("[yellow]→[/yellow] Invoking: [cyan]read_file[/cyan]")
    result2 = server.invoke_tool("read_file", {"path": "/config/settings.json"})
    console.print(f"[green]Result:[/green]\n{result2['result']}\n")

    # Invocation 3: Calculator
    console.print("[yellow]→[/yellow] Invoking: [cyan]calculator[/cyan]")
    result3 = server.invoke_tool("calculator", {"expression": "42 * 3.14"})
    console.print(f"[green]Result:[/green] {result3['result']}\n")

    # Step 5: Resource access
    console.print("[bold green]Step 5:[/bold green] Accessing resources...\n")
    resources = server.list_resources()
    console.print(f"[cyan]Available resources:[/cyan]")
    for res in resources["resources"]:
        console.print(f"  • [cyan]{res['uri']}[/cyan] - {res['description']}")

    resource_content = server.read_resource("file:///config/settings.json")
    console.print(f"\n[green]Resource content:[/green]\n{resource_content['contents'][0]['text']}\n")

    # Step 6: LangChain integration
    console.print("[bold green]Step 6:[/bold green] Integrating with LangChain agent...\n")

    # Convert MCP tools to LangChain tools
    langchain_tools = convert_mcp_to_langchain_tools(server)
    console.print(f"[cyan]Converted {len(langchain_tools)} MCP tools to LangChain tools[/cyan]\n")

    # Create LLM agent
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent = create_react_agent(llm, langchain_tools)

    # Example query
    query = "List all files in the /config directory, read the settings.json file, and tell me what the theme setting is."

    console.print(f"[bold yellow]User Query:[/bold yellow] {query}\n")
    console.print("[dim]Agent is thinking and using MCP tools...[/dim]\n")

    # Run agent
    result = agent.invoke({"messages": [("user", query)]})

    # Display final answer
    final_message = result["messages"][-1]
    console.print(Panel(final_message.content, title="[bold green]Agent Response[/bold green]", border_style="green"))

    # Step 7: Show invocation log
    console.print("\n[bold green]Step 7:[/bold green] Tool invocation history\n")
    display_invocation_log(server)

    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold green]✓ Demo Complete![/bold green]")
    console.print("\n[bold]Key MCP Concepts Demonstrated:[/bold]")
    console.print("  • [cyan]Tool Registration[/cyan] - Tools are registered with the MCP server")
    console.print("  • [cyan]Tool Discovery[/cyan] - Clients can list available tools dynamically")
    console.print("  • [cyan]Tool Invocation[/cyan] - Tools are called through standardized protocol")
    console.print("  • [cyan]Resource Access[/cyan] - Data sources are exposed as resources")
    console.print("  • [cyan]LangChain Integration[/cyan] - MCP tools work with existing agent frameworks")
    console.print("  • [cyan]Observability[/cyan] - All invocations are logged for debugging")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")


if __name__ == "__main__":
    run_demo()
