# Model Context Protocol (MCP)

## Overview

The **Model Context Protocol (MCP)** is a standardized communication protocol that enables Large Language Models (LLMs) to seamlessly interact with external tools, data sources, and services through a unified interface. It provides a structured way for AI systems to discover, invoke, and compose capabilities from multiple providers without tight coupling or custom integrations.

Think of MCP as **USB for AI**—just as USB provides a universal standard for connecting peripherals to computers, MCP provides a universal standard for connecting tools and data sources to LLMs. Whether you need file system access, database queries, API integration, or custom business logic, MCP allows you to expose these capabilities through a consistent protocol that any MCP-compatible client can understand and use.

## Why Use This Pattern?

Traditional approaches to LLM tool integration face significant challenges:

- **Custom integrations**: Each tool requires bespoke code to connect with LLMs
- **Tight coupling**: Tools are hardcoded into specific applications
- **No discoverability**: LLMs can't dynamically find available capabilities
- **Inconsistent interfaces**: Every tool has different invocation patterns
- **Poor reusability**: Tools built for one system can't be used in another
- **Maintenance overhead**: Changes to tool interfaces break integrations
- **Limited composability**: Difficult to combine tools from different providers

The Model Context Protocol solves these problems by:

- **Standardization**: All tools and resources follow the same protocol
- **Dynamic discovery**: LLMs can query what tools and resources are available
- **Loose coupling**: Tools are independent services, not embedded code
- **Interoperability**: MCP tools work with any MCP-compatible client
- **Reusability**: Build once, use across multiple applications
- **Composability**: Easily combine capabilities from different MCP servers
- **Version management**: Protocol supports graceful capability evolution

### Real-World Analogy

**Without MCP (Traditional Approach)**:
- Each appliance has a unique plug and socket (tools have custom interfaces)
- Need different adapters and converters for each device (custom integration code)
- When you get a new device, you might need to rewire your house (refactor application)
- Can't easily move devices between rooms (can't reuse tools across applications)

**With MCP (Standardized Protocol)**:
- All devices use standard USB-C ports (unified MCP interface)
- Plug and play—device announces its capabilities (dynamic discovery)
- Move devices anywhere with compatible ports (reusable across applications)
- Add new devices without system changes (composable architecture)

## How It Works

MCP defines a client-server architecture with three core concepts:

### 1. MCP Servers

Servers expose capabilities to clients through the protocol:

```python
class MCPServer:
    def list_tools() -> list[Tool]          # Announce available tools
    def invoke_tool(name, args) -> result   # Execute tool functionality
    def list_resources() -> list[Resource]  # Announce available data sources
    def read_resource(uri) -> content       # Access resource data
    def list_prompts() -> list[Prompt]      # Announce prompt templates
    def get_prompt(name, args) -> messages  # Generate prompt content
```

### 2. MCP Clients

Clients connect to servers and use their capabilities:

```python
class MCPClient:
    def connect(server_url) -> connection
    def discover_capabilities() -> capabilities
    def call_tool(tool_name, params) -> result
    def access_resource(resource_uri) -> data
    def use_prompt(prompt_name, context) -> messages
```

### 3. Protocol Messages

Standardized JSON-RPC 2.0 messages for communication:

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/list",
  "params": {}
}

{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "tools": [
      {
        "name": "read_file",
        "description": "Read contents of a file",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": {"type": "string"}
          },
          "required": ["path"]
        }
      }
    ]
  }
}
```

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         LLM Client                          │
│  - OpenAI, Anthropic, Local Models                         │
│  - Decides which MCP tools/resources to use                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client Layer                        │
│  - Discovers available servers and capabilities             │
│  - Translates LLM requests to MCP protocol                 │
│  - Manages connections and sessions                        │
└─────────────┬───────────────────────┬───────────────────────┘
              │                       │
    ┌─────────┴──────┐    ┌──────────┴─────────┐
    ↓                ↓    ↓                    ↓
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  MCP    │    │  MCP    │    │  MCP    │    │  MCP    │
│ Server  │    │ Server  │    │ Server  │    │ Server  │
│   #1    │    │   #2    │    │   #3    │    │   #4    │
├─────────┤    ├─────────┤    ├─────────┤    ├─────────┤
│FileSystem│   │Database │    │  APIs   │    │ Custom  │
│  Tools   │   │ Access  │    │ Weather │    │Business │
│Resources │   │  Tools  │    │ Stocks  │    │  Logic  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Tool ecosystem**: Building a marketplace or library of AI tools
- **Multi-tenant systems**: Different users need access to different tool sets
- **Enterprise integration**: Connecting LLMs to internal systems and APIs
- **Plugin architectures**: Allow third parties to extend your AI application
- **Microservices**: Each service exposes capabilities via MCP
- **Data source abstraction**: Unified access to databases, APIs, files
- **Cross-application reusability**: Share tools across multiple AI agents
- **Dynamic capability management**: Add/remove tools without code changes
- **Compliance and security**: Centralized control over tool access

### ❌ When NOT to Use

- **Simple single-tool use**: Direct function calling is simpler
- **Tightly integrated systems**: When tools are part of core application logic
- **Performance critical**: Protocol overhead may be unacceptable
- **Static tool sets**: When tools never change, hardcoding is simpler
- **Prototype/MVP**: Protocol complexity not justified for quick experiments
- **Offline systems**: When network communication isn't available

## Rule of Thumb

**Use MCP when:**
1. You have **3+ tools or data sources** to integrate
2. Tools need to be **reusable across applications**
3. You want **dynamic discovery** of capabilities
4. **Multiple providers** will build integrations
5. You need **standardized interfaces** for consistency
6. **Loose coupling** is important for maintainability

**Don't use MCP when:**
1. You have 1-2 simple tools (use direct function calls)
2. Tools are application-specific (embed directly)
3. Performance is critical (avoid protocol overhead)
4. No need for reusability (simpler integration sufficient)

## Core Components

### 1. Tools

Executable functions that perform actions:

```python
class Tool:
    name: str                  # Unique identifier (e.g., "read_file")
    description: str           # What the tool does
    inputSchema: dict         # JSON Schema for parameters

    # Example
    {
        "name": "search_database",
        "description": "Search customer database by criteria",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        }
    }
```

### 2. Resources

Data sources and content that can be read:

```python
class Resource:
    uri: str                   # Unique identifier (e.g., "file:///data/config.json")
    name: str                  # Human-readable name
    mimeType: str             # Content type
    description: str          # What this resource contains

    # Example
    {
        "uri": "db://customers",
        "name": "Customer Database",
        "mimeType": "application/json",
        "description": "Access to customer records and profiles"
    }
```

### 3. Prompts

Reusable prompt templates with dynamic arguments:

```python
class Prompt:
    name: str                  # Template identifier
    description: str           # What this prompt does
    arguments: list[Argument]  # Dynamic parameters

    # Example
    {
        "name": "code_review",
        "description": "Generate code review for a file",
        "arguments": [
            {"name": "language", "description": "Programming language"},
            {"name": "code", "description": "Code to review"}
        ]
    }
```

### 4. Transport Layers

Communication mechanisms between clients and servers:

- **stdio**: Standard input/output (for local processes)
- **HTTP/SSE**: HTTP with Server-Sent Events (for remote servers)
- **WebSocket**: Bidirectional persistent connections
- **IPC**: Inter-process communication (Unix sockets, named pipes)

## Implementation Approaches

### Approach 1: Simplified MCP Protocol

For educational purposes or when official SDK isn't available:

```python
class SimpleMCPServer:
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
        self.resources = {}

    def register_tool(self, tool: Tool):
        """Register a tool with the server"""
        self.tools[tool.name] = tool

    def register_resource(self, resource: Resource):
        """Register a resource with the server"""
        self.resources[resource.uri] = resource

    def list_tools(self) -> dict:
        """Return all available tools"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in self.tools.values()
            ]
        }

    def invoke_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool with given arguments"""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}

        tool = self.tools[name]
        try:
            result = tool.execute(arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
```

### Approach 2: LangChain Integration

Integrate MCP servers as LangChain tools:

```python
from langchain_core.tools import Tool as LangChainTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Convert MCP tools to LangChain tools
def mcp_to_langchain_tool(mcp_tool):
    def tool_func(**kwargs):
        return mcp_server.invoke_tool(mcp_tool.name, kwargs)

    return LangChainTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        func=tool_func
    )

# Discover and convert all tools
mcp_server = SimpleMCPServer("my-server")
langchain_tools = [
    mcp_to_langchain_tool(tool)
    for tool in mcp_server.list_tools()["tools"]
]

# Create agent with MCP tools
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, langchain_tools)
```

### Approach 3: Multiple Server Composition

Combine capabilities from multiple MCP servers:

```python
class MCPOrchestrator:
    def __init__(self):
        self.servers = {}

    def add_server(self, name: str, server: MCPServer):
        """Register an MCP server"""
        self.servers[name] = server

    def discover_all_tools(self) -> list[Tool]:
        """Get tools from all registered servers"""
        all_tools = []
        for server_name, server in self.servers.items():
            tools = server.list_tools()["tools"]
            # Prefix tool names with server name to avoid conflicts
            for tool in tools:
                tool["name"] = f"{server_name}.{tool['name']}"
                all_tools.append(tool)
        return all_tools

    def invoke_tool(self, qualified_name: str, args: dict):
        """Invoke tool using qualified name (server.tool)"""
        server_name, tool_name = qualified_name.split(".", 1)
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")

        return self.servers[server_name].invoke_tool(tool_name, args)
```

## Key Benefits

### 🔌 Standardization

- **Consistent interfaces**: All tools follow the same protocol
- **Reduced complexity**: No need to learn different APIs for each tool
- **Better documentation**: Standard schemas describe all capabilities
- **Easier testing**: Standard protocol enables generic test frameworks

### ♻️ Reusability

- **Build once, use everywhere**: MCP tools work across applications
- **Tool marketplaces**: Share and discover tools from community
- **Cross-team collaboration**: Different teams can build compatible tools
- **Investment protection**: Tools remain useful as applications evolve

### 🧩 Composability

- **Mix and match**: Combine tools from different providers
- **Incremental adoption**: Add new capabilities without refactoring
- **Server composition**: Aggregate multiple servers into unified interface
- **Capability layering**: Build higher-level tools on top of primitives

### 🔍 Discoverability

- **Dynamic exploration**: LLMs can find available tools at runtime
- **Self-documenting**: Tools describe their own capabilities
- **Capability negotiation**: Clients can request specific features
- **Version compatibility**: Protocol supports feature detection

### 🛡️ Security & Control

- **Centralized policy**: Control tool access in one place
- **Audit trails**: Log all tool invocations through protocol
- **Permission management**: Fine-grained access control per tool
- **Sandboxing**: Isolate tool execution from main application

## Trade-offs

### ⚠️ Protocol Overhead

**Issue**: Additional layer between LLM and tools adds latency and complexity

**Impact**:
- Network roundtrips for discovery and invocation
- JSON serialization/deserialization costs
- Protocol versioning and compatibility management

**Mitigation**:
- Cache tool discovery results
- Use efficient transport (stdio for local, HTTP/2 for remote)
- Batch tool calls when possible
- Implement connection pooling

### 🖥️ Server Management

**Issue**: MCP servers are separate processes that need lifecycle management

**Impact**:
- Must start/stop/monitor server processes
- Handle server failures and restarts
- Manage server configurations and secrets
- Resource consumption (memory, CPU per server)

**Mitigation**:
- Use process managers (systemd, Docker, Kubernetes)
- Implement health checks and auto-restart
- Shared server infrastructure for multiple clients
- Serverless deployment for auto-scaling

### 🔧 Compatibility Challenges

**Issue**: Protocol evolution and version mismatches between clients/servers

**Impact**:
- Breaking changes require coordinated upgrades
- Feature detection adds complexity
- Multiple protocol versions to support

**Mitigation**:
- Semantic versioning for protocol versions
- Feature flags and capability negotiation
- Backward compatibility periods
- Clear deprecation policies

### 📚 Learning Curve

**Issue**: Team needs to understand MCP protocol and architecture

**Impact**:
- Additional documentation and training required
- More complex debugging across protocol boundaries
- Harder to trace issues across client-server divide

**Mitigation**:
- Provide SDK libraries that abstract protocol details
- Comprehensive examples and tutorials
- Good logging and debugging tools
- Start simple, add complexity gradually

## Best Practices

### 1. Tool Design

**Clear, focused tools**:
```python
# Good: Single responsibility
@mcp_tool
def read_file(path: str) -> str:
    """Read and return contents of a single file"""
    return read_file_contents(path)

# Bad: Multiple responsibilities
@mcp_tool
def file_operations(operation: str, path: str, content: str = None):
    """Do various file operations based on operation type"""
    if operation == "read":
        return read_file(path)
    elif operation == "write":
        return write_file(path, content)
    # ... many more operations
```

**Rich descriptions and schemas**:
```python
{
    "name": "search_documents",
    "description": """Search through document database using full-text search.

    Supports:
    - Boolean operators (AND, OR, NOT)
    - Phrase matching with quotes
    - Wildcard matching with *
    - Field-specific search (title:keyword)

    Results are ranked by relevance score.""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query with optional operators"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
                "description": "Maximum number of results to return"
            },
            "filters": {
                "type": "object",
                "description": "Optional filters for date, category, etc."
            }
        },
        "required": ["query"]
    }
}
```

### 2. Error Handling

**Structured error responses**:
```python
{
    "error": {
        "code": "INVALID_ARGUMENT",
        "message": "File path must be absolute",
        "details": {
            "provided_path": "relative/path.txt",
            "expected_format": "/absolute/path.txt"
        }
    }
}
```

**Graceful degradation**:
```python
def invoke_tool(self, name: str, args: dict) -> dict:
    try:
        result = self._execute_tool(name, args)
        return {"result": result}
    except ToolNotFoundError:
        return {"error": {"code": "TOOL_NOT_FOUND", "message": f"Tool '{name}' does not exist"}}
    except InvalidArgumentError as e:
        return {"error": {"code": "INVALID_ARGUMENT", "message": str(e), "details": e.details}}
    except PermissionError:
        return {"error": {"code": "PERMISSION_DENIED", "message": "Insufficient permissions"}}
    except Exception as e:
        logger.exception(f"Unexpected error invoking tool {name}")
        return {"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}
```

### 3. Resource Management

**Lazy loading and streaming**:
```python
class StreamingResource:
    def read_chunks(self, uri: str, chunk_size: int = 4096):
        """Stream large resources in chunks"""
        with open(self._uri_to_path(uri), 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk

    def get_metadata(self, uri: str) -> dict:
        """Get resource metadata without loading content"""
        return {
            "size": os.path.getsize(path),
            "modified": os.path.getmtime(path),
            "mimeType": self._detect_mime_type(path)
        }
```

**Resource subscriptions**:
```python
class ResourceSubscription:
    """Notify clients when resources change"""

    def subscribe(self, uri: str, callback):
        """Subscribe to resource changes"""
        self.subscriptions[uri].append(callback)

    def notify_change(self, uri: str, change_type: str):
        """Notify all subscribers of change"""
        for callback in self.subscriptions.get(uri, []):
            callback(uri, change_type)
```

### 4. Security

**Input validation**:
```python
def validate_tool_input(self, tool: Tool, args: dict) -> tuple[bool, str]:
    """Validate arguments against tool's input schema"""
    try:
        jsonschema.validate(args, tool.inputSchema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"Invalid input: {e.message}"
```

**Access control**:
```python
class MCPServerWithAuth:
    def invoke_tool(self, name: str, args: dict, client_id: str) -> dict:
        # Check if client has permission
        if not self.check_permission(client_id, name):
            return {"error": {"code": "PERMISSION_DENIED"}}

        # Invoke with audit logging
        self.audit_log.record(client_id, name, args)
        return super().invoke_tool(name, args)
```

### 5. Monitoring & Observability

**Comprehensive logging**:
```python
def invoke_tool(self, name: str, args: dict) -> dict:
    start_time = time.time()
    logger.info(f"Tool invocation started: {name}", extra={"args": args})

    try:
        result = self._execute_tool(name, args)
        duration = time.time() - start_time

        logger.info(
            f"Tool invocation succeeded: {name}",
            extra={"duration_ms": duration * 1000, "result_size": len(str(result))}
        )

        return {"result": result}
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Tool invocation failed: {name}",
            extra={"duration_ms": duration * 1000, "error": str(e)},
            exc_info=True
        )
        raise
```

**Metrics collection**:
```python
class MetricsCollector:
    def record_tool_call(self, tool_name: str, duration: float, success: bool):
        self.tool_call_count.labels(tool=tool_name, success=success).inc()
        self.tool_call_duration.labels(tool=tool_name).observe(duration)

    def record_server_health(self):
        self.active_connections.set(len(self.connections))
        self.tools_registered.set(len(self.tools))
```

## Performance Metrics

Track these metrics to ensure MCP deployment is healthy:

### Latency Metrics
- **Tool discovery time**: How long to list all available tools (target: < 100ms)
- **Tool invocation latency**: End-to-end time for tool execution (target: < 500ms for simple tools)
- **Resource access time**: Time to read/stream resources (target: < 200ms for metadata, depends on size for content)
- **Connection establishment**: Time to connect client to server (target: < 50ms)

### Throughput Metrics
- **Tools per second**: Number of tool invocations handled (target: > 100/sec per server)
- **Concurrent connections**: Number of simultaneous clients (target: > 50)
- **Resource streaming bandwidth**: MB/sec for large resources (target: > 10 MB/sec)

### Reliability Metrics
- **Tool success rate**: % of successful invocations (target: > 99%)
- **Server uptime**: % time server is available (target: > 99.9%)
- **Error rate**: % of requests that fail (target: < 1%)
- **Timeout rate**: % of requests that exceed deadline (target: < 0.1%)

### Resource Metrics
- **Memory per server**: RAM used by server process (target: < 500MB for simple servers)
- **CPU utilization**: % CPU during tool execution (target: < 70% average)
- **Open connections**: Number of active client connections (monitor for leaks)

## Example Scenarios

### Scenario 1: File System Operations

```
User: "Read my configuration file and summarize its settings"

LLM: I'll use the MCP file system server to access the file.

Tool Discovery:
- Connect to MCP server "filesystem"
- List available tools: read_file, write_file, list_directory, file_info

Tool Invocation:
- Tool: filesystem.read_file
- Arguments: {"path": "/Users/john/.config/app.json"}
- Result: {"content": "{\"theme\": \"dark\", \"language\": \"en\", ...}"}

Analysis:
- Parse JSON content
- Summarize settings

Answer: Your configuration has the following settings:
- Theme: Dark mode
- Language: English
- Auto-save: Enabled
- Notifications: Disabled
```

### Scenario 2: Database Query with Multiple Servers

```
User: "Find customers who haven't ordered in 90 days and check current weather for their locations"

LLM: I'll need to use both the database server and weather API server.

Server Discovery:
- MCP Server "database" - provides customer and order queries
- MCP Server "weather" - provides weather information

Step 1 - Query inactive customers:
- Tool: database.query_customers
- Arguments: {
    "filter": "last_order_date < DATE_SUB(NOW(), INTERVAL 90 DAY)",
    "fields": ["id", "name", "city"]
  }
- Result: [
    {"id": 123, "name": "Alice", "city": "Seattle"},
    {"id": 456, "name": "Bob", "city": "Portland"}
  ]

Step 2 - Get weather for each location:
- Tool: weather.get_current
- Arguments: {"city": "Seattle"}
- Result: {"temp": 55, "condition": "Rainy"}

- Tool: weather.get_current
- Arguments: {"city": "Portland"}
- Result: {"temp": 58, "condition": "Cloudy"}

Answer: Found 2 customers without orders in 90 days:
- Alice (Seattle) - Currently rainy, 55°F
- Bob (Portland) - Currently cloudy, 58°F
```

### Scenario 3: API Integration with Resources

```
User: "Check our API rate limits and show me the usage documentation"

LLM: I'll access the API management MCP server.

Resource Discovery:
- Connect to "api-management" server
- List resources:
  - api://rate-limits (current rate limit status)
  - api://docs/usage (API usage documentation)

Resource Access:
- Resource: api://rate-limits
- Read: {
    "limit": 10000,
    "used": 7234,
    "remaining": 2766,
    "reset_time": "2024-01-30T14:00:00Z"
  }

- Resource: api://docs/usage
- Read: "# API Usage Guide\n\n## Rate Limits\n..."

Answer: Your API usage:
- Used: 7,234 / 10,000 requests (72.3%)
- Remaining: 2,766 requests
- Resets at: 2:00 PM UTC

[Documentation provided shows authentication, endpoints, and best practices]
```

### Scenario 4: Code Generation with Prompts

```
User: "Generate a REST API endpoint for user authentication"

LLM: I'll use the code generation MCP server's prompt templates.

Prompt Discovery:
- Connect to "code-gen" server
- List prompts: rest_api_endpoint, database_model, test_suite

Prompt Usage:
- Prompt: rest_api_endpoint
- Arguments: {
    "endpoint_name": "authenticate_user",
    "method": "POST",
    "description": "Authenticate user with email and password"
  }
- Result: [Generated code template with authentication logic]

Answer: Here's your authentication endpoint:
[Generated code with security best practices, input validation, error handling]
```

## Advanced Patterns

### 1. Server Composition

Combine multiple MCP servers into a unified facade:

```python
class CompositeServer:
    """Aggregate multiple MCP servers into single interface"""

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}

    def add_server(self, namespace: str, server: MCPServer):
        """Add server with namespace prefix"""
        self.servers[namespace] = server

    def list_tools(self) -> dict:
        """List all tools from all servers"""
        all_tools = []
        for namespace, server in self.servers.items():
            tools = server.list_tools()["tools"]
            for tool in tools:
                # Namespace the tool name
                tool["name"] = f"{namespace}.{tool['name']}"
                all_tools.append(tool)
        return {"tools": all_tools}

    def invoke_tool(self, qualified_name: str, args: dict) -> dict:
        """Route tool call to appropriate server"""
        namespace, tool_name = qualified_name.split(".", 1)
        if namespace not in self.servers:
            return {"error": {"code": "SERVER_NOT_FOUND"}}

        return self.servers[namespace].invoke_tool(tool_name, args)
```

### 2. Resource Subscriptions

Enable push notifications when resources change:

```python
class SubscribableResource:
    """Resource that notifies clients of changes"""

    def __init__(self):
        self.subscribers: dict[str, list[callable]] = {}
        self.resources: dict[str, Any] = {}

    def subscribe(self, uri: str, callback: callable) -> str:
        """Subscribe to resource updates"""
        subscription_id = str(uuid.uuid4())
        if uri not in self.subscribers:
            self.subscribers[uri] = []
        self.subscribers[uri].append((subscription_id, callback))
        return subscription_id

    def unsubscribe(self, subscription_id: str):
        """Remove subscription"""
        for uri, subs in self.subscribers.items():
            self.subscribers[uri] = [
                (sid, cb) for sid, cb in subs if sid != subscription_id
            ]

    def update_resource(self, uri: str, new_content: Any):
        """Update resource and notify subscribers"""
        old_content = self.resources.get(uri)
        self.resources[uri] = new_content

        # Notify all subscribers
        for subscription_id, callback in self.subscribers.get(uri, []):
            try:
                callback(uri, old_content, new_content)
            except Exception as e:
                logger.error(f"Subscription callback failed: {e}")
```

### 3. Streaming Tool Results

For tools that produce large or progressive output:

```python
class StreamingTool:
    """Tool that yields results progressively"""

    def execute_streaming(self, name: str, args: dict):
        """Execute tool and stream results"""
        if name == "search_large_dataset":
            # Yield results as they're found
            for batch in self._search_in_batches(args["query"]):
                yield {
                    "type": "progress",
                    "data": batch,
                    "completed": False
                }

            yield {
                "type": "complete",
                "data": {"total_results": self.total},
                "completed": True
            }

        elif name == "generate_report":
            # Yield report sections as they're generated
            for section in self._generate_sections(args):
                yield {
                    "type": "section",
                    "data": section,
                    "completed": False
                }

            yield {"type": "complete", "completed": True}
```

### 4. Capability Negotiation

Clients and servers negotiate supported features:

```python
class VersionedServer:
    """Server that supports multiple protocol versions"""

    SUPPORTED_VERSIONS = ["1.0", "1.1", "2.0"]

    def initialize(self, client_version: str) -> dict:
        """Negotiate protocol version with client"""
        if client_version not in self.SUPPORTED_VERSIONS:
            # Find best compatible version
            compatible = self._find_compatible_version(client_version)
            if not compatible:
                return {
                    "error": "Protocol version not supported",
                    "client_version": client_version,
                    "supported_versions": self.SUPPORTED_VERSIONS
                }

        # Return server capabilities for negotiated version
        return {
            "protocol_version": client_version,
            "server_info": {
                "name": self.name,
                "version": self.version
            },
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": client_version >= "1.1",
                "streaming": client_version >= "2.0",
                "subscriptions": client_version >= "2.0"
            }
        }
```

### 5. Tool Chaining

Server-side composition of multiple tool calls:

```python
class ChainableTool:
    """Tool that can invoke other tools"""

    def execute_chain(self, chain_definition: dict) -> dict:
        """Execute a chain of tool calls"""
        results = {}

        for step in chain_definition["steps"]:
            tool_name = step["tool"]

            # Substitute variables from previous results
            args = self._substitute_variables(step["args"], results)

            # Execute tool
            result = self.invoke_tool(tool_name, args)

            # Store result for later steps
            results[step["name"]] = result

        return results[chain_definition["output"]]

    def _substitute_variables(self, args: dict, results: dict) -> dict:
        """Replace ${step.field} with actual values"""
        substituted = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("${"):
                # Extract reference like ${previous_step.result}
                ref = value[2:-1]
                step_name, field = ref.split(".", 1)
                substituted[key] = results[step_name][field]
            else:
                substituted[key] = value
        return substituted
```

## Comparison with Related Patterns

| Aspect | MCP | Tool Use | Function Calling | Plugins |
|--------|-----|----------|------------------|---------|
| **Standardization** | Full protocol spec | Ad-hoc | Provider-specific | App-specific |
| **Discoverability** | Dynamic, built-in | Manual listing | Manual listing | Plugin manifest |
| **Reusability** | Cross-application | Single application | Single provider | Single application |
| **Deployment** | Separate servers | Embedded | Embedded | Plugin system |
| **Composability** | High (multiple servers) | Medium | Low | Medium |
| **Protocol overhead** | Yes (JSON-RPC) | Minimal | Minimal | Varies |
| **Use case** | Tool ecosystems | Direct tool use | Model-native tools | App extensions |

**When to choose each:**

- **MCP**: Building tool ecosystem, multiple providers, need reusability
- **Tool Use**: Simple integration, single application, performance critical
- **Function Calling**: Using provider-native capabilities (OpenAI, Anthropic)
- **Plugins**: App-specific extensions with custom logic

## Common Pitfalls

### 1. Over-Engineering Simple Tools

**Problem**: Using full MCP protocol for 1-2 simple tools

**Solution**: Start with direct function calls, migrate to MCP when you have 3+ tools or need reusability

### 2. Inadequate Error Handling

**Problem**: Servers crash or return cryptic errors

**Solution**:
- Implement comprehensive error codes
- Return structured error messages
- Log errors server-side for debugging
- Provide error recovery guidance

### 3. Poor Schema Design

**Problem**: Vague tool descriptions, missing validation, unclear parameters

**Solution**:
- Write detailed descriptions with examples
- Use JSON Schema fully (types, constraints, defaults)
- Document edge cases and limitations
- Test schema validation thoroughly

### 4. Security Oversights

**Problem**: Unrestricted tool access, no input validation, exposed secrets

**Solution**:
- Implement authentication and authorization
- Validate all inputs against schemas
- Use principle of least privilege
- Audit all tool invocations
- Never pass secrets in arguments

### 5. Ignoring Performance

**Problem**: Slow tool discovery, high latency, resource leaks

**Solution**:
- Cache tool/resource listings
- Implement timeouts and limits
- Use connection pooling
- Monitor and profile server performance
- Stream large resources instead of loading in memory

### 6. Version Management Neglect

**Problem**: Breaking changes without versioning, incompatible clients/servers

**Solution**:
- Version your protocol implementation
- Support capability negotiation
- Maintain backward compatibility
- Document breaking changes clearly
- Provide migration guides

## Conclusion

The Model Context Protocol (MCP) represents a paradigm shift in how we integrate tools and data sources with Large Language Models. By providing a standardized, discoverable, and composable protocol, MCP enables the creation of rich ecosystems of AI capabilities that can be shared, reused, and combined in powerful ways.

**Use MCP when:**
- Building tool libraries for multiple applications
- Need dynamic discovery and composition of capabilities
- Want standardized interfaces for consistency
- Multiple teams or vendors will provide integrations
- Loose coupling and reusability are priorities
- Enterprise integration with diverse systems

**Implementation checklist:**
- ✅ Design clear, focused tools with single responsibilities
- ✅ Provide rich descriptions and JSON schemas
- ✅ Implement comprehensive error handling
- ✅ Use appropriate transport (stdio/HTTP/WebSocket)
- ✅ Add authentication and authorization
- ✅ Cache discovery results for performance
- ✅ Monitor latency, throughput, and errors
- ✅ Support protocol versioning and capability negotiation
- ✅ Document tools and resources thoroughly
- ✅ Test across different clients and use cases

**Key Takeaways:**
- 🔌 MCP standardizes tool integration like USB standardized hardware
- 🔍 Dynamic discovery enables flexible, runtime composition
- ♻️ Reusable tools reduce development time and improve quality
- 🧩 Composable servers allow mixing capabilities from multiple providers
- 🛡️ Protocol layer enables centralized security and governance
- ⚖️ Trade protocol overhead for flexibility and maintainability
- 🚀 Ideal for tool ecosystems, enterprise integration, and multi-tenant systems

---

*MCP transforms LLMs from closed systems into open platforms—enabling a future where AI capabilities can be shared, discovered, and composed as easily as we use apps on our phones or packages in programming languages.*
