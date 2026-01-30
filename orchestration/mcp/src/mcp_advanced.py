"""
Model Context Protocol (MCP): Advanced Multi-Server Implementation

This example demonstrates advanced MCP concepts:
- Multiple specialized MCP servers with different capabilities
- Server composition and orchestration
- Resource subscriptions and updates
- Error handling and validation
- Context management across tool calls
- Comprehensive monitoring and logging
- LangChain integration with complex workflows

This shows how MCP enables building a rich ecosystem of composable AI tools.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

console = Console()


# ═══════════════════════════════════════════════════════════════
# Enhanced MCP Protocol Implementation
# ═══════════════════════════════════════════════════════════════


@dataclass
class MCPTool:
    """Enhanced MCP tool with metadata and validation"""

    name: str
    description: str
    inputSchema: dict
    execute: Callable
    category: str = "general"
    version: str = "1.0.0"


@dataclass
class MCPResource:
    """Enhanced MCP resource with subscriptions"""

    uri: str
    name: str
    mimeType: str
    description: str
    content: Any
    subscribers: list[Callable] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MCPServerMetrics:
    """Metrics for monitoring MCP server performance"""

    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_duration_ms: float = 0.0
    tools_by_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class EnhancedMCPServer:
    """
    Enhanced MCP Server with monitoring, validation, and subscriptions.
    """

    name: str
    description: str
    tools: dict[str, MCPTool] = field(default_factory=dict)
    resources: dict[str, MCPResource] = field(default_factory=dict)
    invocation_log: list[dict] = field(default_factory=list)
    metrics: MCPServerMetrics = field(default_factory=MCPServerMetrics)

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with validation"""
        self.tools[tool.name] = tool
        self.metrics.tools_by_usage[tool.name] = 0

    def register_resource(self, resource: MCPResource) -> None:
        """Register a resource"""
        self.resources[resource.uri] = resource

    def list_tools(self) -> dict:
        """List all available tools with categories"""
        tools_by_category = {}

        for tool in self.tools.values():
            if tool.category not in tools_by_category:
                tools_by_category[tool.category] = []

            tools_by_category[tool.category].append(
                {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema, "version": tool.version}
            )

        return {"tools": [tool_info for tools in tools_by_category.values() for tool_info in tools], "categories": tools_by_category}

    def invoke_tool(self, name: str, arguments: dict) -> dict:
        """Invoke tool with monitoring and error handling"""
        start_time = time.time()
        timestamp = datetime.now()

        self.metrics.total_invocations += 1

        if name not in self.tools:
            self.metrics.failed_invocations += 1
            error_response = {"error": {"code": "TOOL_NOT_FOUND", "message": f"Tool '{name}' not found in {self.name}"}}
            self.invocation_log.append(
                {
                    "timestamp": timestamp,
                    "server": self.name,
                    "tool": name,
                    "arguments": arguments,
                    "success": False,
                    "duration_ms": 0,
                    "response": error_response,
                }
            )
            return error_response

        tool = self.tools[name]

        try:
            # Validate input against schema (simplified)
            self._validate_input(tool.inputSchema, arguments)

            # Execute the tool
            result = tool.execute(**arguments)

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.successful_invocations += 1
            self.metrics.total_duration_ms += duration_ms
            self.metrics.tools_by_usage[name] = self.metrics.tools_by_usage.get(name, 0) + 1

            response = {"result": result, "metadata": {"duration_ms": duration_ms, "server": self.name}}

            self.invocation_log.append(
                {
                    "timestamp": timestamp,
                    "server": self.name,
                    "tool": name,
                    "arguments": arguments,
                    "success": True,
                    "duration_ms": duration_ms,
                    "response": response,
                }
            )

            return response

        except ValueError as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.failed_invocations += 1

            error_response = {"error": {"code": "INVALID_ARGUMENT", "message": str(e), "tool": name}}

            self.invocation_log.append(
                {
                    "timestamp": timestamp,
                    "server": self.name,
                    "tool": name,
                    "arguments": arguments,
                    "success": False,
                    "duration_ms": duration_ms,
                    "response": error_response,
                }
            )

            return error_response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.failed_invocations += 1

            error_response = {
                "error": {"code": "EXECUTION_ERROR", "message": f"Tool execution failed: {str(e)}", "tool": name, "details": str(e)}
            }

            self.invocation_log.append(
                {
                    "timestamp": timestamp,
                    "server": self.name,
                    "tool": name,
                    "arguments": arguments,
                    "success": False,
                    "duration_ms": duration_ms,
                    "response": error_response,
                }
            )

            return error_response

    def _validate_input(self, schema: dict, arguments: dict) -> None:
        """Basic input validation against JSON schema"""
        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                raise ValueError(f"Missing required field: {field}")

    def list_resources(self) -> dict:
        """List all available resources"""
        return {
            "resources": [
                {
                    "uri": res.uri,
                    "name": res.name,
                    "mimeType": res.mimeType,
                    "description": res.description,
                    "last_updated": res.last_updated.isoformat(),
                }
                for res in self.resources.values()
            ]
        }

    def read_resource(self, uri: str) -> dict:
        """Read a resource's content"""
        if uri not in self.resources:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": f"Resource '{uri}' not found"}}

        resource = self.resources[uri]
        return {"contents": [{"uri": uri, "mimeType": resource.mimeType, "text": resource.content}]}

    def subscribe_resource(self, uri: str, callback: Callable) -> str:
        """Subscribe to resource updates"""
        if uri not in self.resources:
            raise ValueError(f"Resource '{uri}' not found")

        self.resources[uri].subscribers.append(callback)
        return f"subscription-{uri}-{len(self.resources[uri].subscribers)}"

    def update_resource(self, uri: str, new_content: Any) -> None:
        """Update resource and notify subscribers"""
        if uri not in self.resources:
            raise ValueError(f"Resource '{uri}' not found")

        resource = self.resources[uri]
        old_content = resource.content
        resource.content = new_content
        resource.last_updated = datetime.now()

        # Notify all subscribers
        for callback in resource.subscribers:
            try:
                callback(uri, old_content, new_content)
            except Exception as e:
                console.print(f"[red]Error in subscription callback: {e}[/red]")


# ═══════════════════════════════════════════════════════════════
# Specialized MCP Servers
# ═══════════════════════════════════════════════════════════════


def create_database_server() -> EnhancedMCPServer:
    """Create an MCP server for database operations"""

    server = EnhancedMCPServer(name="database", description="Database query and management server")

    # Mock database
    mock_database = {
        "users": [
            {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "city": "Seattle", "status": "active"},
            {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "city": "Portland", "status": "active"},
            {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "city": "San Francisco", "status": "inactive"},
            {"id": 4, "name": "Diana Prince", "email": "diana@example.com", "city": "Seattle", "status": "active"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "product": "Laptop", "amount": 1200, "date": "2024-01-15"},
            {"id": 102, "user_id": 2, "product": "Mouse", "amount": 25, "date": "2024-01-20"},
            {"id": 103, "user_id": 1, "product": "Keyboard", "amount": 80, "date": "2024-01-25"},
        ],
    }

    # Tool: Query Users
    def query_users_impl(filter_field: str = None, filter_value: str = None) -> str:
        """Query users with optional filtering"""
        users = mock_database["users"]

        if filter_field and filter_value:
            users = [u for u in users if str(u.get(filter_field, "")).lower() == filter_value.lower()]

        return json.dumps(users, indent=2)

    query_users_tool = MCPTool(
        name="query_users",
        description="Query users from the database with optional filtering by field and value",
        inputSchema={
            "type": "object",
            "properties": {
                "filter_field": {"type": "string", "description": "Field to filter by (e.g., 'city', 'status')"},
                "filter_value": {"type": "string", "description": "Value to filter for"},
            },
        },
        execute=query_users_impl,
        category="database",
    )

    # Tool: Get User Orders
    def get_user_orders_impl(user_id: int) -> str:
        """Get all orders for a specific user"""
        orders = [o for o in mock_database["orders"] if o["user_id"] == user_id]
        return json.dumps(orders, indent=2)

    get_orders_tool = MCPTool(
        name="get_user_orders",
        description="Get all orders for a specific user by user ID",
        inputSchema={"type": "object", "properties": {"user_id": {"type": "integer", "description": "User ID"}}, "required": ["user_id"]},
        execute=get_user_orders_impl,
        category="database",
    )

    server.register_tool(query_users_tool)
    server.register_tool(get_orders_tool)

    # Register database schema as resource
    schema_resource = MCPResource(
        uri="db://schema",
        name="Database Schema",
        mimeType="application/json",
        description="Database schema and table definitions",
        content=json.dumps({"tables": ["users", "orders"], "users_fields": list(mock_database["users"][0].keys())}, indent=2),
    )
    server.register_resource(schema_resource)

    return server


def create_weather_server() -> EnhancedMCPServer:
    """Create an MCP server for weather information"""

    server = EnhancedMCPServer(name="weather", description="Weather information and forecasts")

    # Mock weather data
    weather_data = {
        "seattle": {"temp": 55, "condition": "Rainy", "humidity": 85, "wind": "10 mph NW"},
        "portland": {"temp": 58, "condition": "Cloudy", "humidity": 75, "wind": "8 mph N"},
        "san francisco": {"temp": 65, "condition": "Sunny", "humidity": 60, "wind": "12 mph W"},
    }

    # Tool: Get Current Weather
    def get_weather_impl(city: str) -> str:
        """Get current weather for a city"""
        city_lower = city.lower()
        if city_lower in weather_data:
            data = weather_data[city_lower]
            return f"Weather in {city.title()}:\n" + "\n".join(f"  {k.title()}: {v}" for k, v in data.items())
        return f"Weather data not available for: {city}"

    weather_tool = MCPTool(
        name="get_current_weather",
        description="Get current weather conditions for a specified city",
        inputSchema={"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]},
        execute=get_weather_impl,
        category="weather",
    )

    server.register_tool(weather_tool)

    # Register weather alert resource
    alert_resource = MCPResource(
        uri="weather://alerts",
        name="Weather Alerts",
        mimeType="application/json",
        description="Active weather alerts and warnings",
        content=json.dumps({"alerts": [{"city": "Seattle", "type": "Heavy Rain Warning", "severity": "moderate"}]}, indent=2),
    )
    server.register_resource(alert_resource)

    return server


def create_analytics_server() -> EnhancedMCPServer:
    """Create an MCP server for data analytics"""

    server = EnhancedMCPServer(name="analytics", description="Data analysis and statistics")

    # Tool: Calculate Statistics
    def calculate_stats_impl(numbers: list[float]) -> str:
        """Calculate basic statistics for a list of numbers"""
        if not numbers:
            return "Error: Empty list provided"

        mean = sum(numbers) / len(numbers)
        sorted_nums = sorted(numbers)
        median = sorted_nums[len(sorted_nums) // 2]
        min_val = min(numbers)
        max_val = max(numbers)

        return f"Statistics:\n  Mean: {mean:.2f}\n  Median: {median:.2f}\n  Min: {min_val}\n  Max: {max_val}\n  Count: {len(numbers)}"

    stats_tool = MCPTool(
        name="calculate_statistics",
        description="Calculate mean, median, min, max for a list of numbers",
        inputSchema={
            "type": "object",
            "properties": {"numbers": {"type": "array", "items": {"type": "number"}, "description": "List of numbers to analyze"}},
            "required": ["numbers"],
        },
        execute=calculate_stats_impl,
        category="analytics",
    )

    server.register_tool(stats_tool)

    return server


# ═══════════════════════════════════════════════════════════════
# MCP Orchestrator - Multi-Server Composition
# ═══════════════════════════════════════════════════════════════


@dataclass
class MCPOrchestrator:
    """
    Orchestrates multiple MCP servers, providing unified access to all capabilities.

    This demonstrates how MCP enables composability - combining tools from different
    servers into a single cohesive interface.
    """

    servers: dict[str, EnhancedMCPServer] = field(default_factory=dict)
    global_invocation_log: list[dict] = field(default_factory=list)

    def add_server(self, server: EnhancedMCPServer) -> None:
        """Register an MCP server"""
        self.servers[server.name] = server
        console.print(f"[green]✓[/green] Registered MCP server: [cyan]{server.name}[/cyan] - {server.description}")

    def discover_all_tools(self) -> dict:
        """Discover tools from all registered servers"""
        all_tools = []

        for server_name, server in self.servers.items():
            tools_response = server.list_tools()

            for tool in tools_response["tools"]:
                # Namespace tool names with server name
                tool["name"] = f"{server_name}.{tool['name']}"
                tool["server"] = server_name
                all_tools.append(tool)

        return {"tools": all_tools, "server_count": len(self.servers)}

    def invoke_tool(self, qualified_name: str, arguments: dict) -> dict:
        """
        Invoke tool using qualified name (server.tool).

        This routing enables transparent access to tools across multiple servers.
        """
        if "." not in qualified_name:
            return {"error": {"code": "INVALID_TOOL_NAME", "message": "Tool name must be qualified with server (e.g., 'database.query')"}}

        server_name, tool_name = qualified_name.split(".", 1)

        if server_name not in self.servers:
            return {"error": {"code": "SERVER_NOT_FOUND", "message": f"Server '{server_name}' not registered"}}

        result = self.servers[server_name].invoke_tool(tool_name, arguments)

        # Log at orchestrator level
        self.global_invocation_log.append(
            {"timestamp": datetime.now(), "server": server_name, "tool": qualified_name, "arguments": arguments, "result": result}
        )

        return result

    def list_all_resources(self) -> dict:
        """List resources from all servers"""
        all_resources = []

        for server_name, server in self.servers.items():
            resources = server.list_resources()["resources"]
            for res in resources:
                res["server"] = server_name
                all_resources.append(res)

        return {"resources": all_resources}

    def get_metrics_summary(self) -> dict:
        """Get aggregated metrics from all servers"""
        total_invocations = 0
        total_successful = 0
        total_failed = 0
        avg_duration = 0.0
        tool_usage = {}

        for server in self.servers.values():
            metrics = server.metrics
            total_invocations += metrics.total_invocations
            total_successful += metrics.successful_invocations
            total_failed += metrics.failed_invocations

            if metrics.total_invocations > 0:
                avg_duration += metrics.total_duration_ms / metrics.total_invocations

            for tool, count in metrics.tools_by_usage.items():
                tool_usage[f"{server.name}.{tool}"] = count

        num_servers = len(self.servers)
        avg_duration = avg_duration / num_servers if num_servers > 0 else 0

        return {
            "total_invocations": total_invocations,
            "successful": total_successful,
            "failed": total_failed,
            "success_rate": (total_successful / total_invocations * 100) if total_invocations > 0 else 0,
            "avg_duration_ms": avg_duration,
            "tool_usage": tool_usage,
            "servers": num_servers,
        }


# ═══════════════════════════════════════════════════════════════
# LangChain Integration
# ═══════════════════════════════════════════════════════════════


def convert_orchestrator_to_langchain_tools(orchestrator: MCPOrchestrator) -> list:
    """Convert all tools from orchestrator to LangChain tools"""

    langchain_tools = []
    tools_response = orchestrator.discover_all_tools()

    for tool_info in tools_response["tools"]:
        tool_name = tool_info["name"]
        tool_description = f"[{tool_info['server']}] {tool_info['description']}"

        def make_tool_func(name: str):
            def tool_func(**kwargs) -> str:
                result = orchestrator.invoke_tool(name, kwargs)
                if "result" in result:
                    return str(result["result"])
                elif "error" in result:
                    return f"Error: {result['error']['message']}"
                return str(result)

            return tool_func

        langchain_tool = tool(make_tool_func(tool_name))
        langchain_tool.name = tool_name
        langchain_tool.description = tool_description

        langchain_tools.append(langchain_tool)

    return langchain_tools


# ═══════════════════════════════════════════════════════════════
# Rich Visualization Dashboard
# ═══════════════════════════════════════════════════════════════


def create_dashboard(orchestrator: MCPOrchestrator) -> Layout:
    """Create a rich dashboard showing MCP system status"""

    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=5))

    # Header
    header_text = "[bold cyan]Model Context Protocol (MCP) - Live Dashboard[/bold cyan]"
    layout["header"].update(Panel(header_text, style="cyan"))

    # Body - split into left and right
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    # Left: Server Tree
    tree = Tree("[bold green]🌐 MCP Ecosystem[/bold green]")

    for server_name, server in orchestrator.servers.items():
        server_node = tree.add(f"[bold cyan]📡 {server_name}[/bold cyan] - {server.description}")

        # Tools
        tools_node = server_node.add("[yellow]🔧 Tools[/yellow]")
        for tool in server.tools.values():
            tools_node.add(f"[cyan]{tool.name}[/cyan] - {tool.description[:40]}...")

        # Resources
        resources_node = server_node.add("[blue]📦 Resources[/blue]")
        for resource in server.resources.values():
            resources_node.add(f"[cyan]{resource.uri}[/cyan]")

    layout["left"].update(Panel(tree, title="Server Hierarchy", border_style="green"))

    # Right: Metrics
    metrics = orchestrator.get_metrics_summary()

    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Metric", style="yellow")
    metrics_table.add_column("Value", style="cyan")

    metrics_table.add_row("Total Invocations", str(metrics["total_invocations"]))
    metrics_table.add_row("Successful", f"✓ {metrics['successful']}")
    metrics_table.add_row("Failed", f"✗ {metrics['failed']}")
    metrics_table.add_row("Success Rate", f"{metrics['success_rate']:.1f}%")
    metrics_table.add_row("Avg Duration", f"{metrics['avg_duration_ms']:.2f}ms")
    metrics_table.add_row("Active Servers", str(metrics["servers"]))

    layout["right"].update(Panel(metrics_table, title="System Metrics", border_style="blue"))

    # Footer: Recent Invocations
    inv_table = Table(show_header=True, header_style="bold magenta", box=None)
    inv_table.add_column("Time", style="dim", width=8)
    inv_table.add_column("Server", style="cyan", width=10)
    inv_table.add_column("Tool", style="yellow", width=20)
    inv_table.add_column("Status", style="green", width=10)

    for entry in orchestrator.global_invocation_log[-3:]:
        timestamp = entry["timestamp"].strftime("%H:%M:%S")
        server = entry["server"]
        tool = entry["tool"].split(".", 1)[1] if "." in entry["tool"] else entry["tool"]
        status = "✓ Success" if "result" in entry["result"] else "✗ Failed"

        inv_table.add_row(timestamp, server, tool, status)

    layout["footer"].update(Panel(inv_table, title="Recent Tool Invocations", border_style="magenta"))

    return layout


# ═══════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════


def run_demo():
    """Run the advanced MCP demonstration"""

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Model Context Protocol (MCP) - Advanced Implementation[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Step 1: Create orchestrator and register servers
    console.print("[bold green]Step 1:[/bold green] Creating MCP servers and orchestrator...\n")

    orchestrator = MCPOrchestrator()

    # Create and register specialized servers
    db_server = create_database_server()
    weather_server = create_weather_server()
    analytics_server = create_analytics_server()

    orchestrator.add_server(db_server)
    orchestrator.add_server(weather_server)
    orchestrator.add_server(analytics_server)

    console.print()

    # Step 2: Display ecosystem
    console.print("[bold green]Step 2:[/bold green] MCP Ecosystem Overview\n")

    initial_dashboard = create_dashboard(orchestrator)
    console.print(initial_dashboard)
    console.print()

    # Step 3: Tool discovery
    console.print("[bold green]Step 3:[/bold green] Discovering all tools across servers...\n")

    all_tools = orchestrator.discover_all_tools()
    console.print(f"[cyan]Discovered {len(all_tools['tools'])} tools from {all_tools['server_count']} servers:[/cyan]\n")

    tool_table = Table(show_header=True, header_style="bold magenta")
    tool_table.add_column("Server", style="cyan")
    tool_table.add_column("Tool", style="yellow")
    tool_table.add_column("Description", style="white")

    for tool in all_tools["tools"]:
        server_name = tool["server"]
        tool_name = tool["name"].split(".", 1)[1]
        description = tool["description"][:50] + "..." if len(tool["description"]) > 50 else tool["description"]
        tool_table.add_row(server_name, tool_name, description)

    console.print(tool_table)
    console.print()

    # Step 4: Direct invocations
    console.print("[bold green]Step 4:[/bold green] Testing cross-server tool invocations...\n")

    # Query database
    console.print("[yellow]→[/yellow] [cyan]database.query_users[/cyan] (filter by city='Seattle')")
    result1 = orchestrator.invoke_tool("database.query_users", {"filter_field": "city", "filter_value": "Seattle"})
    console.print(f"[green]Result:[/green] Found users in Seattle\n")

    # Get weather
    console.print("[yellow]→[/yellow] [cyan]weather.get_current_weather[/cyan] (city='Seattle')")
    result2 = orchestrator.invoke_tool("weather.get_current_weather", {"city": "Seattle"})
    console.print(f"[green]Result:[/green]\n{result2['result']}\n")

    # Calculate stats
    console.print("[yellow]→[/yellow] [cyan]analytics.calculate_statistics[/cyan] (order amounts)")
    result3 = orchestrator.invoke_tool("analytics.calculate_statistics", {"numbers": [1200, 25, 80]})
    console.print(f"[green]Result:[/green]\n{result3['result']}\n")

    # Step 5: Resource subscriptions
    console.print("[bold green]Step 5:[/bold green] Resource subscriptions and updates...\n")

    # Subscribe to weather alerts
    def on_alert_update(uri: str, old_content: Any, new_content: Any):
        console.print(f"[yellow]🔔 Resource updated:[/yellow] {uri}")
        console.print(f"[dim]New content:[/dim] {new_content[:100]}...")

    weather_server.subscribe_resource("weather://alerts", on_alert_update)
    console.print("[cyan]✓ Subscribed to weather alerts[/cyan]")

    # Simulate resource update
    new_alert = json.dumps(
        {"alerts": [{"city": "Seattle", "type": "Heavy Rain Warning", "severity": "moderate"}, {"city": "Portland", "type": "Wind Advisory", "severity": "low"}]},
        indent=2,
    )
    weather_server.update_resource("weather://alerts", new_alert)
    console.print()

    # Step 6: LangChain integration
    console.print("[bold green]Step 6:[/bold green] LangChain agent using all MCP servers...\n")

    # Convert to LangChain tools
    langchain_tools = convert_orchestrator_to_langchain_tools(orchestrator)
    console.print(f"[cyan]Converted {len(langchain_tools)} MCP tools to LangChain tools[/cyan]\n")

    # Create agent
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent = create_react_agent(llm, langchain_tools)

    # Complex query using multiple servers
    query = """Find all users in Seattle from the database, get the current weather for Seattle,
    and calculate statistics for their order amounts. Provide a comprehensive summary."""

    console.print(f"[bold yellow]User Query:[/bold yellow] {query}\n")
    console.print("[dim]Agent coordinating across multiple MCP servers...[/dim]\n")

    # Run agent
    result = agent.invoke({"messages": [("user", query)]})

    # Display final answer
    final_message = result["messages"][-1]
    console.print(Panel(final_message.content, title="[bold green]Agent Response (Multi-Server Coordination)[/bold green]", border_style="green"))
    console.print()

    # Step 7: Updated dashboard
    console.print("[bold green]Step 7:[/bold green] Updated system dashboard\n")

    final_dashboard = create_dashboard(orchestrator)
    console.print(final_dashboard)
    console.print()

    # Step 8: Metrics summary
    console.print("[bold green]Step 8:[/bold green] Performance metrics\n")

    metrics = orchestrator.get_metrics_summary()

    metrics_panel = Table(show_header=True, header_style="bold cyan")
    metrics_panel.add_column("Metric", style="yellow")
    metrics_panel.add_column("Value", style="green")

    metrics_panel.add_row("Total Tool Invocations", str(metrics["total_invocations"]))
    metrics_panel.add_row("Successful Invocations", f"{metrics['successful']} ({metrics['success_rate']:.1f}%)")
    metrics_panel.add_row("Failed Invocations", str(metrics["failed"]))
    metrics_panel.add_row("Average Latency", f"{metrics['avg_duration_ms']:.2f}ms")
    metrics_panel.add_row("Active Servers", str(metrics["servers"]))

    console.print(metrics_panel)
    console.print()

    # Tool usage breakdown
    if metrics["tool_usage"]:
        console.print("[bold cyan]Tool Usage Breakdown:[/bold cyan]")
        for tool, count in sorted(metrics["tool_usage"].items(), key=lambda x: x[1], reverse=True):
            console.print(f"  [cyan]{tool}[/cyan]: {count} invocations")

    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold green]✓ Advanced Demo Complete![/bold green]")
    console.print("\n[bold]Advanced MCP Concepts Demonstrated:[/bold]")
    console.print("  • [cyan]Multi-Server Architecture[/cyan] - Multiple specialized MCP servers")
    console.print("  • [cyan]Server Orchestration[/cyan] - Unified access to distributed capabilities")
    console.print("  • [cyan]Namespaced Tools[/cyan] - Conflict-free tool composition (server.tool)")
    console.print("  • [cyan]Resource Subscriptions[/cyan] - Event-driven updates for resources")
    console.print("  • [cyan]Error Handling[/cyan] - Comprehensive validation and error reporting")
    console.print("  • [cyan]Monitoring[/cyan] - Real-time metrics and performance tracking")
    console.print("  • [cyan]Cross-Server Coordination[/cyan] - LLM agents using tools from multiple servers")
    console.print("  • [cyan]Composability[/cyan] - Building complex workflows from simple tools")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")


if __name__ == "__main__":
    run_demo()
