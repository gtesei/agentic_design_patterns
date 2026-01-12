# Tool Use Pattern

## Overview

The **Tool Use Pattern** (also known as Function Calling) enables agentic systems to break out of the LLM's internal knowledge and interact with the external world through structured API calls, database queries, code execution, and real-world actions.

## Why Use This Pattern?

LLMs are trained on static datasets with knowledge cutoffs, making them unable to:
- Access real-time information (current weather, stock prices, news)
- Query private or proprietary databases
- Perform precise calculations or execute code
- Take actions in external systems (send emails, control devices, update databases)
- Retrieve user-specific data or context

The Tool Use pattern solves this by:
- **Describing available functions** to the LLM in a structured format
- **Enabling dynamic decision-making** about when and which tools to use
- **Generating structured function calls** with appropriate parameters
- **Executing external operations** and feeding results back to the LLM
- **Augmenting static knowledge** with dynamic, real-world data

### Example: Without vs. With Tool Use
```
Without Tool Use (Limited):
User: "What's the weather in San Francisco?"
LLM: "I don't have access to real-time weather data..."

With Tool Use (Powerful):
User: "What's the weather in San Francisco?"
LLM → Decides to use weather_api tool
    → Calls get_weather(location="San Francisco")
    → Receives: {"temp": 62°F, "conditions": "Partly cloudy"}
LLM: "It's currently 62°F and partly cloudy in San Francisco."
```

## How It Works

1. **Tool Definition**: Define available functions with descriptions, parameters, and return types
2. **Tool Registration**: Register tools with the LLM/agent framework
3. **User Query**: User submits a request requiring external data or action
4. **Tool Selection**: LLM analyzes the query and decides which tool(s) to use
5. **Function Call Generation**: LLM generates structured function call with parameters
6. **Execution**: Orchestration layer executes the actual function call
7. **Result Integration**: Tool output is fed back to the LLM
8. **Response Generation**: LLM incorporates results into final natural language response

### Typical Architecture
```
User Input
    ↓
LLM Agent (with tool awareness)
    ↓
Decision: Need external data?
    ↓
  YES → Generate Function Call
         {
           "tool": "weather_api",
           "parameters": {"location": "San Francisco"}
         }
         ↓
      Tool Execution Layer
         ↓
      External API/Database/System
         ↓
      Return Result: {"temp": 62, "conditions": "cloudy"}
         ↓
      Feed back to LLM
         ↓
      Generate Natural Language Response
         ↓
      Final Output to User
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Real-time data retrieval**: Weather, stock prices, news, sports scores
- **Private/proprietary data access**: Company databases, CRM systems, internal documentation
- **Precise calculations**: Mathematical operations, unit conversions, financial computations
- **Code execution**: Running Python/JavaScript code, data analysis, visualization
- **External actions**: Sending emails, creating tickets, updating records, controlling IoT devices
- **API integrations**: Search engines, translation services, payment processors
- **File operations**: Reading, writing, parsing documents
- **Database queries**: SQL queries, vector search, data aggregation

### ❌ When NOT to Use

- **Information within LLM's training data**: Historical facts, general knowledge
- **Simple reasoning tasks**: No external data needed
- **Privacy-sensitive operations**: Avoid exposing sensitive systems without proper authentication
- **High-latency tolerance issues**: Tool calls add execution time
- **Unpredictable APIs**: Unreliable external services may cause failures

## Rule of Thumb

**Use Tool Use when:**
1. Task requires **real-time or dynamic data** not in the LLM's training
2. Need to access **private/proprietary information** (databases, documents)
3. Task requires **precise calculations** or **code execution**
4. Need to **trigger actions** in external systems
5. Information **changes frequently** (prices, availability, status)

**Don't use Tool Use when:**
1. LLM's existing knowledge suffices
2. External system access is unavailable or unreliable
3. Security/privacy concerns outweigh benefits
4. Simple prompt engineering can solve the problem

## Framework Support

### LangChain

LangChain provides the `@tool` decorator and built-in tool utilities:
```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Define a tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: City name or zip code
        
    Returns:
        Weather description string
    """
    # In production, call actual weather API
    return f"The weather in {location} is 72°F and sunny"

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")
        
    Returns:
        Result of the calculation
    """
    return eval(expression)  # Use safely in production!

# Create agent with tools
llm = ChatOpenAI(model="gpt-4")
tools = [get_weather, calculate]

agent = create_react_agent(llm, tools)

# Use the agent
response = agent.invoke({
    "messages": [("user", "What's the weather in Tokyo and what's 15% of 250?")]
})
```

### LangGraph (Advanced)

For more control over tool execution flow:
```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Define state and nodes
def agent_node(state: MessagesState):
    # LLM decides to use tools
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

### Google ADK

Google's Agent Development Kit provides pre-built tools:
```python
from google.generativeai import tools

# Use built-in Google Search
search_tool = tools.GoogleSearch()

# Code execution tool
code_exec_tool = tools.CodeExecution()

# Vertex AI Search
vertex_search_tool = tools.VertexAISearch(
    search_engine_id="your-search-engine-id"
)

# Create agent with tools
agent = Agent(
    model="gemini-pro",
    tools=[search_tool, code_exec_tool, vertex_search_tool]
)
```

### OpenAI Function Calling

Direct OpenAI API usage:
```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or zip code"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if function was called
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    # Execute the function
    result = get_weather(**function_args)
    
    # Send result back to LLM for final response
    # ...
```

## Key Benefits

### 🌐 Real-World Interaction
- **Live data access**: Current information beyond training cutoff
- **Action execution**: Trigger real-world effects (send emails, update databases)
- **System integration**: Connect to existing enterprise systems

### 🎯 Accuracy & Precision
- **Factual correctness**: Use authoritative sources instead of LLM memory
- **Precise calculations**: Offload math to specialized functions
- **Data validation**: Verify information against ground truth

### 🔧 Extensibility
- **Modular design**: Add new capabilities by adding tools
- **Custom integrations**: Wrap any API or function as a tool
- **Domain specialization**: Tailor agents to specific use cases

### 💰 Cost Efficiency
- **Reduced token usage**: Don't include large datasets in prompts
- **Targeted retrieval**: Fetch only necessary information
- **Efficient processing**: Use specialized tools for specific tasks

## Important Considerations

### ⚠️ Security & Safety

**Authentication & Authorization:**
- Validate user permissions before executing tools
- Use API keys and secure credential management
- Implement rate limiting to prevent abuse

**Input Validation:**
- Sanitize all tool parameters
- Prevent SQL injection, code injection attacks
- Validate data types and ranges

**Sandboxing:**
- Execute code in isolated environments
- Limit tool capabilities and permissions
- Monitor tool execution for suspicious activity

### 🔍 Error Handling

**Tool Failures:**
```python
@tool
def search_database(query: str) -> str:
    """Search company database"""
    try:
        result = db.execute(query)
        return result
    except DatabaseError as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
```

**Graceful Degradation:**
- Provide fallback responses when tools fail
- Log errors for debugging
- Inform user about limitations

### 📊 Performance Impact

**Latency Considerations:**
- Each tool call adds network/execution time
- Multiple tool calls compound delays
- Consider async/parallel execution for independent tools

**Cost Implications:**
- External API calls may have usage costs
- Additional LLM calls to process tool results
- Balance tool use with direct LLM reasoning

### 🎯 Tool Design Best Practices

**Clear Descriptions:**
```python
@tool
def get_stock_price(symbol: str, date: str = None) -> dict:
    """Get stock price information for a given symbol.
    
    This tool retrieves current or historical stock prices from 
    financial markets. Use this when users ask about stock values,
    market prices, or trading information.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        date: Optional date in YYYY-MM-DD format. If not provided,
              returns current price.
              
    Returns:
        Dictionary with price, volume, and change information
        
    Examples:
        get_stock_price("AAPL") -> Current Apple stock price
        get_stock_price("TSLA", "2024-01-15") -> Tesla price on Jan 15
    """
    # Implementation
```

**Structured Parameters:**
- Use type hints for clarity
- Provide enums for limited options
- Include default values where appropriate
- Document parameter constraints

**Atomic Operations:**
- Each tool should do ONE thing well
- Avoid overly complex multi-purpose tools
- Compose simple tools for complex workflows

## Performance Metrics

Track these metrics for tool-using agents:

- **Tool selection accuracy**: % of correct tool choices
- **Tool execution success rate**: % of successful tool calls
- **Average latency per tool**: Time from call to result
- **Tool usage frequency**: Which tools are used most
- **Error rate by tool**: Identify unreliable tools
- **Cost per tool call**: API usage costs
- **User satisfaction**: Does tool use improve responses?

## Example Scenarios

### Scenario 1: Customer Support Agent
```python
@tool
def get_order_status(order_id: str) -> dict:
    """Get current status of a customer order"""
    return db.orders.find_one({"id": order_id})

@tool
def update_shipping_address(order_id: str, new_address: str) -> bool:
    """Update shipping address for an order"""
    return db.orders.update({"id": order_id}, {"address": new_address})

@tool
def initiate_refund(order_id: str, amount: float) -> str:
    """Process a refund for an order"""
    refund_id = payment_api.refund(order_id, amount)
    return f"Refund initiated: {refund_id}"

# User: "What's the status of order #12345?"
# Agent uses get_order_status tool → Returns tracking info
```

### Scenario 2: Data Analysis Assistant
```python
@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query on the analytics database"""
    results = analytics_db.execute(query)
    return results.to_json()

@tool
def create_chart(data: dict, chart_type: str) -> str:
    """Generate a visualization from data"""
    chart_path = visualization_engine.create(data, chart_type)
    return chart_path

# User: "Show me sales trends for Q4"
# Agent uses execute_sql → Then create_chart → Returns visualization
```

### Scenario 3: Smart Home Controller
```python
@tool
def set_temperature(room: str, temp: float) -> str:
    """Adjust thermostat in a specific room"""
    iot_api.set_thermostat(room, temp)
    return f"Temperature set to {temp}°F in {room}"

@tool
def control_lights(room: str, state: str, brightness: int = 100) -> str:
    """Turn lights on/off or adjust brightness"""
    iot_api.lights(room, state, brightness)
    return f"Lights in {room}: {state} at {brightness}%"

# User: "Make the living room cooler and dim the lights"
# Agent uses set_temperature + control_lights
```

## Tool Composition Patterns

### Sequential Tool Use
```
User Query → Tool A → Result A → Tool B (uses Result A) → Final Response
```

### Parallel Tool Use
```
User Query → Tool A ↘
             Tool B  → Combine Results → Final Response
             Tool C ↗
```

### Conditional Tool Use
```
User Query → Tool A → If condition met → Tool B
                     → Else → Tool C
```

### Iterative Tool Use
```
User Query → Tool A → Evaluate → Not satisfied → Tool A (refined params)
                                → Satisfied → Final Response
```

## Related Patterns

- **Prompt Chaining**: Tools can be steps in a chain
- **Routing**: Route to different tool sets based on query type
- **Reflection**: Use tools to validate/critique outputs
- **Parallelization**: Execute multiple tool calls simultaneously
- **RAG**: Tools retrieve context for generation

## Conclusion

The Tool Use pattern is fundamental to building capable agentic systems that can interact with the real world. By giving LLMs the ability to call external functions, you transform them from knowledge repositories into active agents that can retrieve live data, perform actions, and integrate with existing systems.

**Use Tool Use when:**
- Need real-time or dynamic information
- Accessing private/proprietary data
- Performing precise calculations or code execution
- Triggering actions in external systems
- Integrating with existing APIs and services

**Implementation checklist:**
- ✅ Define tools with clear, detailed descriptions
- ✅ Implement robust error handling
- ✅ Secure tool execution with proper authentication
- ✅ Validate all inputs and outputs
- ✅ Monitor tool usage and performance
- ✅ Provide fallbacks for tool failures
- ✅ Document tool capabilities and limitations

**Key Takeaways:**
- 🔧 Tool Use enables LLMs to interact with external systems
- 🎯 Clear tool descriptions improve selection accuracy
- 🔒 Security and validation are critical
- ⚡ Tool calls add latency but enable powerful capabilities
- 🏗️ Start simple, add tools as needed
- 📊 Monitor and optimize tool performance

---

*Tool Use transforms static LLMs into dynamic agents capable of real-world interaction—making them invaluable for production applications.*