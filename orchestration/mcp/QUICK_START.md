# Model Context Protocol (MCP) - Quick Start Guide

Get started with the Model Context Protocol pattern in under 5 minutes!

## What You'll Learn

- How to create a basic MCP server with tools
- How to connect an LLM client to use MCP tools
- How to compose multiple MCP servers
- How to integrate MCP with LangChain agents

## Prerequisites

- Python 3.11+
- OpenAI API key (set in `../../.env`)

## Installation

```bash
# Navigate to the mcp directory
cd orchestration/mcp

# Install dependencies
pip install -e .

# Or use uv (faster)
uv pip install -e .
```

## Basic Usage

### Example 1: Basic MCP Server (5 minutes)

Run a simple MCP server with file and calculation tools:

```bash
./run.sh basic
```

This demonstrates:
- Creating an MCP server
- Registering tools (file operations, calculator)
- Tool discovery by LLM client
- Tool invocation through the protocol
- Structured responses

**What you'll see:**
```
[Server] Registered tools: read_file, write_file, calculator
[Client] Discovering available tools...
[Client] Found 3 tools
[LLM] Using tool: read_file
[Server] Executing: read_file(path="/config/settings.json")
[Server] Result: {...}
[LLM] Answer: Your configuration contains...
```

### Example 2: Advanced Multi-Server Setup (10 minutes)

Run multiple MCP servers with different capabilities:

```bash
./run.sh advanced
```

This demonstrates:
- Multiple specialized MCP servers (filesystem, database, weather)
- Server composition and orchestration
- Resource exposure and access
- Context management across servers
- LangChain integration
- Rich monitoring dashboard

**What you'll see:**
- Server startup with capability registration
- Tool discovery across multiple servers
- Coordinated multi-server tool usage
- Real-time monitoring metrics
- Resource subscriptions and updates

## Code Structure

```
mcp/
├── README.md              # Comprehensive documentation
├── QUICK_START.md         # This file
├── pyproject.toml         # Dependencies
├── run.sh                 # Execution script
└── src/
    ├── mcp_basic.py       # Basic MCP implementation
    └── mcp_advanced.py    # Advanced multi-server setup
```

## Key Concepts in 60 Seconds

### 1. MCP Server
A server that exposes tools and resources through a standardized protocol:

```python
server = SimpleMCPServer("my-server")
server.register_tool(read_file_tool)
server.register_resource(config_resource)
```

### 2. Tool Discovery
LLMs can dynamically discover what tools are available:

```python
tools = server.list_tools()
# Returns: [{"name": "read_file", "description": "...", "inputSchema": {...}}]
```

### 3. Tool Invocation
Execute tools through the protocol:

```python
result = server.invoke_tool("read_file", {"path": "/data/file.txt"})
# Returns: {"result": "file contents..."}
```

### 4. Resources
Access data sources through URIs:

```python
resources = server.list_resources()
content = server.read_resource("file:///config.json")
```

## Common Tasks

### Create a Custom Tool

```python
from src.mcp_basic import MCPTool, SimpleMCPServer

# Define your tool
def my_custom_logic(param1: str, param2: int) -> str:
    return f"Processed {param1} with {param2}"

my_tool = MCPTool(
    name="custom_tool",
    description="Does something useful",
    inputSchema={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "integer", "description": "Second parameter"}
        },
        "required": ["param1", "param2"]
    },
    execute=my_custom_logic
)

# Register it
server = SimpleMCPServer("my-server")
server.register_tool(my_tool)
```

### Connect LLM to MCP Server

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Convert MCP tools to LangChain tools
langchain_tools = convert_mcp_tools(server)

# Create agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, langchain_tools)

# Use the agent
result = agent.invoke({
    "messages": [("user", "Read the config file and summarize it")]
})
```

### Compose Multiple Servers

```python
orchestrator = MCPOrchestrator()
orchestrator.add_server("files", file_server)
orchestrator.add_server("db", database_server)
orchestrator.add_server("weather", weather_server)

# LLM can now use tools from all servers
# Tools are namespaced: files.read_file, db.query, weather.get_current
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mcp'"

**Solution**: The official MCP SDK isn't used. We implement a simplified protocol. No additional installation needed.

### Issue: "OPENAI_API_KEY not found"

**Solution**: Create `../../.env` file with:
```bash
OPENAI_API_KEY=sk-...your-key-here...
```

### Issue: Tools not being discovered

**Solution**: Check that:
1. Tools are registered with `server.register_tool(tool)`
2. Tool schemas are valid JSON Schema
3. Server is started before client connects

### Issue: High latency

**Solution**:
- Cache tool discovery results
- Use stdio transport for local servers
- Batch multiple tool calls when possible

## Next Steps

1. **Read the full README.md** for comprehensive understanding
2. **Modify the examples** to add your own tools
3. **Experiment with resources** and prompt templates
4. **Build a multi-server system** for your use case
5. **Add authentication** and monitoring for production

## Examples to Try

### Example: File System Analysis
```bash
# Run basic example and ask:
"Read all JSON files in the /config directory and summarize their contents"
```

### Example: Multi-Source Data Integration
```bash
# Run advanced example and ask:
"Query the database for active users, get weather for their locations, and create a summary report"
```

### Example: Resource Subscriptions
```bash
# Run advanced example and observe:
# - Real-time updates when resources change
# - Automatic notification to subscribed clients
# - Event-driven architecture in action
```

## Learning Path

**Beginner** (1-2 hours):
1. Run `./run.sh basic`
2. Understand tool registration and discovery
3. Create a custom tool
4. Test with different LLM queries

**Intermediate** (3-4 hours):
1. Run `./run.sh advanced`
2. Study multi-server composition
3. Implement resource subscriptions
4. Add error handling and validation

**Advanced** (5+ hours):
1. Build production-ready MCP server
2. Add authentication and authorization
3. Implement streaming for large data
4. Create monitoring dashboard
5. Deploy with Docker/Kubernetes

## Additional Resources

- **Official MCP Spec**: https://modelcontextprotocol.io
- **LangChain Integration**: See `src/mcp_advanced.py`
- **JSON-RPC 2.0**: https://www.jsonrpc.org/specification
- **JSON Schema**: https://json-schema.org/

## Getting Help

- Check the README.md for detailed explanations
- Review code comments in source files
- Examine error messages for debugging hints
- Monitor server logs for troubleshooting

---

**Ready to build?** Start with `./run.sh basic` and explore! 🚀
