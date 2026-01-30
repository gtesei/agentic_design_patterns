# Tool Use - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd foundational_design_patterns/5_tool_use
```

### Step 2: Run the Example
```bash
bash run.sh
```

---

## 📖 Understanding Tool Use in 30 Seconds

**Tool Use** enables LLMs to interact with external systems:

```
Without Tools:                    With Tools:
User: "Weather in NYC?"          User: "Weather in NYC?"
LLM: "I don't have real-time     LLM → Calls weather_api("NYC")
      data..."                        → Returns: 72°F, Sunny
                                  LLM: "It's 72°F and sunny!"
```

LLMs can now access live data, perform calculations, and take actions!

---

## 🎯 What This Example Does

The example demonstrates **tool-using agents**:

1. **Search** - Query databases or APIs
2. **Calculate** - Perform mathematical operations
3. **Retrieve** - Get real-time data
4. **Execute** - Take actions in external systems

---

## 💡 Example Flow

```
Query: "What's 15% of the current Apple stock price?"
    ↓
LLM Decides: "I need the stock price first"
    ↓
Action: Call get_stock_price("AAPL")
    ↓
Result: $178.52
    ↓
LLM Decides: "Now I'll calculate 15%"
    ↓
Action: Call calculator("178.52 * 0.15")
    ↓
Result: 26.78
    ↓
Response: "15% of Apple's stock price ($178.52) is $26.78"
```

---

## 🔧 Key Concepts

### Function Calling
LLM generates structured function calls with parameters.

### External Integration
Connect to APIs, databases, and external systems.

### Dynamic Decision-Making
LLM decides when and which tools to use.

### Result Integration
Tool outputs are incorporated into responses.

---

## 🎨 When to Use Tool Use

✅ **Good For:**
- Real-time data retrieval (weather, stocks, news)
- Private/proprietary data access (databases, CRM)
- Precise calculations
- Code execution
- External actions (emails, updates, control)
- API integrations

❌ **Not Ideal For:**
- Information in LLM's training data
- Simple reasoning tasks
- Privacy-sensitive operations without proper auth
- High-latency sensitive applications

---

## 🛠️ Defining Tools

### Using @tool Decorator (LangChain)
```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name or zip code

    Returns:
        Weather description
    """
    # Call actual weather API
    return f"Weather in {location}: 72°F, sunny"

@tool
def calculator(expression: str) -> float:
    """Perform mathematical calculations.

    Args:
        expression: Math expression (e.g., "2 + 2")

    Returns:
        Calculation result
    """
    return eval(expression)
```

### Creating an Agent
```python
from langgraph.prebuilt import create_react_agent

tools = [get_weather, calculator]
agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [("user", "What's the weather in Tokyo?")]
})
```

---

## 📊 Tool Use Benefits

| Capability | Without Tools | With Tools |
|------------|--------------|------------|
| Real-time Data | ❌ | ✅ |
| Precise Calculations | ⚠️ (approximate) | ✅ (exact) |
| External Actions | ❌ | ✅ |
| Private Data Access | ❌ | ✅ |
| Cost | Lower | Higher (extra calls) |
| Accuracy | Lower | Higher (grounded) |

---

## 💡 Common Tool Patterns

### 1. Search Tools
```python
@tool
def search(query: str) -> str:
    """Search the web or knowledge base"""
    return search_api.query(query)
```

### 2. Calculation Tools
```python
@tool
def calculator(expression: str) -> float:
    """Perform math operations"""
    return eval(expression)
```

### 3. API Tools
```python
@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price"""
    return stock_api.get_price(symbol)
```

### 4. Database Tools
```python
@tool
def query_database(sql: str) -> list:
    """Execute SQL query"""
    return db.execute(sql)
```

### 5. Action Tools
```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    email_api.send(to, subject, body)
    return "Email sent successfully"
```

---

## 🔧 Customization Tips

### Add Custom Tools
```python
@tool
def my_custom_tool(param: str) -> str:
    """Description of what this tool does"""
    # Your implementation
    result = do_something(param)
    return result

# Add to agent
tools = [get_weather, calculator, my_custom_tool]
agent = create_react_agent(llm, tools)
```

### Tool with Multiple Parameters
```python
@tool
def book_meeting(date: str, time: str, attendees: list) -> str:
    """Book a meeting on the calendar.

    Args:
        date: Meeting date (YYYY-MM-DD)
        time: Meeting time (HH:MM)
        attendees: List of email addresses
    """
    calendar_api.create_event(date, time, attendees)
    return f"Meeting booked for {date} at {time}"
```

### Error Handling in Tools
```python
@tool
def safe_tool(param: str) -> str:
    """Tool with error handling"""
    try:
        result = risky_operation(param)
        return f"Success: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

---

## 🐛 Common Issues & Solutions

### Issue: Tool Not Being Called
**Solution**: Improve tool description with clear use cases.

### Issue: Wrong Parameters
**Solution**: Add detailed parameter descriptions and examples.

### Issue: Tool Errors Breaking Agent
**Solution**: Implement error handling in tool functions.

### Issue: High Latency
**Solution**: Optimize tool execution, use caching, or async calls.

---

## 🔒 Security Considerations

### Input Validation
```python
@tool
def safe_query(sql: str) -> str:
    """Execute safe SQL query"""
    # Validate input
    if "DROP" in sql.upper() or "DELETE" in sql.upper():
        return "Error: Unsafe operation"
    return db.execute(sql)
```

### Authentication
```python
@tool
def authenticated_api_call(token: str, endpoint: str) -> str:
    """Make authenticated API call"""
    if not validate_token(token):
        return "Error: Invalid authentication"
    return api.call(endpoint, token)
```

### Rate Limiting
```python
@tool
@rate_limit(calls=10, period=60)  # 10 calls per minute
def rate_limited_tool(param: str) -> str:
    """Tool with rate limiting"""
    return expensive_api_call(param)
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)
- **Related Patterns**:
  - Pattern 8 (ReAct) - Tool use with reasoning
  - Pattern 7 (Multi-Agent) - Tools per agent

---

## 🎓 Next Steps

1. ✅ Run the tool use examples
2. ✅ Create a custom tool
3. ✅ Test with different queries
4. ✅ Implement error handling
5. ✅ Add authentication to tools

---

**Pattern Type**: External Integration

**Complexity**: ⭐⭐⭐ (Intermediate)

**Best For**: Real-time data, precise calculations, external actions

**Key Benefit**: Grounded, accurate responses with live data
