
Write README.md for Multi-Agent Collaboration

::::This is the template::::
====
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
=====

::::these are the notes that you need to use::::
=====
0:00
We've talked a lot about how to build a single agent to complete tasks for you.
0:04
In a multi-agent or multi-agentic workflow, we instead have a collection of multiple agents
0:09
collaborate to do things for you.
0:12
When some people hear for the first time about multi-agent systems, they wonder, why do I
0:16
need multiple agents?
0:18
It's just the same LLM that I'm prompting over and over, or just one computer.
0:22
Why do I need multiple agents?
0:25
I find that one useful analogy is, even though I may do things on a single computer, we do
0:31
decompose work in a single computer into maybe multiple processes or multiple threads.
0:37
And as a developer, thinking, even though it's one CPU on a computer, say, thinking
0:42
about how to take work and decompose it into multiple processes and multi-computer programs
0:47
to run, that makes it easier for me as a developer to write code.
0:52
And in a similar way too, if you have a complex task to carry out, sometimes, instead of thinking
0:59
about how to hire one person to do it for you, you might think about hiring a team of
1:04
a few people to do different pieces of the task for you.
1:08
And so in practice, I found that for many developers of agentic systems, having this
1:12
mental framework of not asking, what's the one person I might hire to do something, but
1:17
instead, would it make sense to hire people with three or four different roles to do this
1:22
overall task for me, that helps give another way to take a complex thing and decompose
1:28
it into sub-tasks and to build for those individual sub-tasks one at a time.
1:33
Let's take a look at some examples of how this works.
1:36
Take the task of creating marketing assets, say you want to market sunglasses.
1:40
Can you come up with a marketing brochure for that?
1:43
You might need a researcher on your team to look at trends on sunglasses and what competitors
1:48
are offering.
1:49
You might also have a graphic designer on your team to render charts or nice-looking
1:54
graphics of your sunglasses.
1:56
And then also a writer to take the research, take the graphic assets and put it all together
2:00
into a nice-looking brochure.
2:02
Or to write a research article, you might want a researcher to do online research, a
2:06
statistician to calculate statistics, a lead writer, and then an editor to come up with
2:10
a polished report.
2:11
Or to prepare a legal case, real law firms will often have associates, paralegals, maybe
2:16
an investigator.
2:18
And we naturally, because of the way human teams do work, can think of different ways
2:24
that complex tasks can be broken down into different individuals with different roles.
2:31
So these are examples of when a complex task were already naturally decomposed into sub-tasks
2:38
that different people with different skills can carry out.
2:41
Take the example of creating marketing assets.
2:44
Look into detail into what a researcher, graphic designer, and writer might do.
2:49
A researcher might have the task of analyzing market trends and researching competitors.
2:55
And when designing the research agents, one question to keep in mind is what are the tools
3:01
that the researcher may need in order to come up with a research report on market trends
3:06
and what competitors are doing.
3:07
So one natural tool that an agentic researcher might need to use would be web search.
3:14
Because as a human researcher, asked to do these tasks might need to search online in
3:18
order to come up with their report.
3:20
Or for a graphic designer agent, they might be tasked with creating visualizations and
3:25
artwork.
3:26
And so what are the tools that an agentic software graphic designer might need?
3:31
Well, they may need image generation and manipulation APIs.
3:36
Or maybe, similar to what you saw with the coffee machine example, maybe it needs code
3:41
execution to generate charts.
3:44
And lastly, the writer has transformed the research into report text and marketing copy.
3:49
And in this case, they don't need any tools other than what an LLM can already do to generate
3:54
text.
3:55
In this and the next video, I'm going to use these purple boxes to denote an agent.
4:00
And the way you build individual agents is by prompting an LLM to play the role of a
4:06
researcher or a graphic designer or a writer, depending on which agent it is part of.
4:11
So for example, for the research agents, you might prompt it to say, you are a research
4:15
agent, expert at analyzing market trends and competitors, carry out online research to
4:22
analyze market trends for the sunglasses product and give a summary as well of what competitors
4:26
are doing.
4:27
So that would allow you to build a researcher agent.
4:30
And similarly, by prompting an LLM to act as a graphic designer with the appropriate
4:35
tools and to act as a writer, that's how you can build a graphic designer as well as
4:41
a writer agent.
4:43
Having built these three agents, one way to have them work together to generate your final
4:49
reports would be to use a simple linear audit workflow or a linear plan in this case.
4:56
So if you want to create a summer marketing campaign for sunglasses, you might give that
5:00
prompt to the research agents.
5:02
The research agent then writes a report that says, here are the current sunglasses trends
5:06
and competitive offerings.
5:08
This research report can then be fed to the graphic designer that looks at the data the
5:13
research has found and creates a few data visualizations and artwork options.
5:17
All these assets can then be passed to the writer that then takes the research and the
5:23
graphic output and writes the final marketing brochure.
5:27
The advantage of building a multi-agent workflow in this case is when designing a researcher
5:32
or graphic designer or writer, you can focus on one thing at a time.
5:36
So I can spend some time building maybe the best graphic designer agents I can, while
5:41
maybe my collaborators are building research agents and writer agents.
5:45
And in the end, we string it all together to come up with this multi-agent system.
5:50
And in some cases, I'm seeing developers start to reuse some agents as well.
5:56
So having built a graphic designer for marketing brochures, maybe I'll think about if I can
6:01
build a more general graphic designer that can help me write marketing brochures as well
6:05
as social media posts, as well as help me illustrate online webpages.
6:10
So by coming up with what are the agents you might hire to do a task, and this will sometimes
6:16
correspond to who are the types of human employees you might hire to do a task.
6:22
You can come up with a workflow like this with maybe even building agents that you could
6:27
choose to reuse in other applications as well.
6:30
Now, what you see here is a linear plan where one agent, the researcher does his work, then
6:36
the graphic designer, and then the writer.
6:38
With agents, you can also, as an alternative to a linear plan, you can also have agents
6:44
interact with each other in more complex ways.
6:47
Let me illustrate with an example of planning using multiple agents.
6:51
So previously, you saw how we may give an LLM a set of tools that we can call to carry
6:56
out different tasks.
6:57
In what I want to show you, we will instead give an LLM the option to call on different
7:03
agents to ask the different agents to help complete different tasks.
7:07
So in detail, you might write a prompt like you're a marketing manager, have the following
7:11
team of agents to work with, and then give a description of the agents.
7:14
And this is very much similar to what we're doing with planning and using tools, except
7:19
the tools, the green boxes, are replaced with agents, these purple boxes that the LLM can
7:24
call on.
7:25
You can also ask it to return a step-by-step plan to carry out the user's request.
7:28
And in this case, the LLM may ask the researcher to research current sunglasses trends and
7:33
then report back.
7:34
Then it will ask the graphic designer to create the images and then report back, then ask
7:38
the writer to create a report, and then maybe the LLM will choose to review or reflect on
7:42
and improve the report one final time.
7:45
In executing this plan, you would then take the step one text of the researcher, carry
7:50
out research, then pass that to the graphic designer, pass it to the writer, and then
7:55
maybe do one final reflection step, and then you'd be done.
7:59
One interesting view of this workflow is as if you have these three agents up here, but
8:04
this LLM on the left is actually like a fourth agent that's a marketing manager, that is
8:10
a manager of a marketing team, that is setting direction and then delegating tasks to the
8:15
researcher, the graphic designer, and the writer agents.
8:18
So this becomes actually a collection of four agents for a marketing manager agent coordinating
8:22
the work of the researcher, the graphic designer, and the writer.
8:26
In this video, you saw two communication patterns.
8:30
One was a linear one where your agents took actions one at a time until you got to the
8:35
end.
8:36
And the second had a marketing manager coordinating the activity of a few other agents.
8:41
It turns out that one of the key design decisions you may end up having to make when building
8:46
multi-agentic systems is what is the communication pattern between your different agents?
8:51
This is an area of hard research and there are multiple patterns emerging, but in the
8:56
next video, I want to show you what are some of the most common communication patterns
8:59
for getting your agents to work with each other.
9:02
Let's go see that in the next video.
=====
# M5 Agentic AI - Market Research Team

## 1. Introduction

### 1.1. Lab Overview  

In this lab, you will step into the role of a **technical AI lead at a fashion brand** preparing a summer sunglasses campaign. Your task is to design a **fully automated creative pipeline** that mirrors a real-world business scenario. Instead of handling each piece manually, you will guide a system that scans online sources for emerging fashion trends, matches those trends with sunglasses in the internal catalog, designs a campaign visual, generates a short marketing quote, and finally packages everything into an **executive-ready report**.  

The goal is to experience how multiple agents, tools, and models can be orchestrated into a single, coherent workflow. By the end of this lab, you will have built a pipeline that feels less like a script of isolated steps and more like a small team working together to solve a creative challenge.  

### 1.2. 🎯 Learning outcome

By completing this lab, you will see how to move beyond single-turn interactions with a model and instead design **multi-agent pipelines** that coordinate planning, research, and creative generation. You will learn how to ground agent reasoning in external tools so that outputs are not just imaginative but also supported by real data. You will also experiment with reflection and packaging steps that enforce quality control and prepare results for an executive audience.  

In short, this lab is about learning how to combine the **imagination of large language models** with the **discipline of structured workflows**, giving you a practical pattern for building autonomous systems that are both creative and reliable.  

## 2. Setup: Import libraries and load environment

As in previous labs, you now import the required libraries, load environment variables, and set up helper utilities.


```python
# =========================
# Imports
# =========================

# --- Standard library ---
import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO

# --- Third-party ---
import requests
import openai
from PIL import Image
from dotenv import load_dotenv
from IPython.display import Markdown, display
import aisuite

# --- Local / project ---
import tools
import utils


# =========================
# Environment & Client
# =========================
load_dotenv()
client = aisuite.Client()

```

## 3. Available Tools  

Agentic pipelines only become effective when the model is given **explicit capabilities** beyond its base reasoning. Declaring these tools upfront makes the agent’s action space unambiguous, ensures that prompts naturally guide tool selection, and keeps orchestration and testing transparent through well-defined interfaces.  

You will assemble a **marketing research team**, a group of specialized agents collaborating to design a summer sunglasses campaign. To empower them, we start by defining the tools that will ground their reasoning in real data.  

The first tool is `tools.tavily_search_tool`, which performs live web searches to uncover evidence of current fashion trends. Try it now by running a simple query for *“trends in sunglasses fashion”*:  


```python
tools.tavily_search_tool('trends in sunglasses fashion')
```




    [{'title': "The Top Sunglasses Trends of 2025: Styles Everyone's ...",
      'content': 'The top trends in 2025 include oversized frames, futuristic shield sunglasses, transparent frames, retro-inspired designs, color-tinted lenses,',
      'url': 'https://topfoxx.com/en-au/blogs/news/the-top-sunglasses-trends-of-2025-styles-everyone-s-talking-about?srsltid=AfmBOopUy0gXNo4-LaNkP_vTFxk1VLYWqP-MRaI8iKyjcjMf2A-srSzF'},
     {'title': 'The Biggest Sunglasses Trends of 2024',
      'content': 'The best sunglasses trends for 2024 include oversized cat eye, retro bug eye lenses, oversized aviators and squares, as well as rectangles.',
      'url': 'https://www.diffeyewear.com/collections/the-biggest-sunglasses-trends-of-2024?srsltid=AfmBOoqIR7uyGWdl1cWwovFJndX1r4U06znAV0sTJLi0OIpgnT7XfVSh'},
     {'title': 'Sunglasses Trends 2026',
      'content': 'The must-have 2026 sunglass trends · Soft oval · Angular square · Gradient tones · Cool metal · Maxi statements · Glasses Chains · Get your style matches.',
      'url': 'https://miaburton.com/en/sunglasses/trends?srsltid=AfmBOopR-8dXClXYcKxUUwVSWx9ic7R3fCwFdYkZrW_MDLL5vjRej7h3'},
     {'title': 'Sunglasses Trends of the Moment',
      'content': 'Versace gives a nod to the enduring elegance of Western hardware with this style’s iconic Medusa logo. Perfect for those seeking the freedom of the open range, Persol redefines rodeo style with these iconic pilot sunglasses. Shop even more Western-inspired styles. ## Step back in time with a selection of retro-inspired shades and show your unique personality through a combo of icons & new styles. Inspired by the rebellious spirit of British pop, these Burberry shades feature a striking new tubular construction and echo the styles of the 1970s. Featuring the iconic Burberry check in deep purple, this style embraces modern femininity with a nod to vintage charm. The perfect blend of iconic design and contemporary flair, this Ray-Ban Reverse pilot style captures the spirit of the past while looking confidently to the future. Perfectly crafted to evoke the vibrant and joyful spirit of bygone days with a contemporary twist, these sunnies feature a harmonious blend of lilac tones that add a whimsical yet elegant touch to any ensemble. Shop even more nostalgia-inspired styles.',
      'url': 'https://www.sunglasshut.com/us/sunglasses/trends-of-the-moment'},
     {'title': '2025 eyewear trends: The best sunglasses to shop now',
      'content': "# This summer's 7 sunglasses trends to shop now. Sunglasses are a summer essential, but each season brings fresh updates to classic styles. Characterised by full cheeks, a rounded chin, and equal width and length, round faces benefit from angular frames such as rectangular, square, or cat-eye sunglasses. Aviators, rectangular frames, and oversized sunglasses are excellent choices. Bottom-heavy frames, aviators, or round sunglasses soften the forehead and add width to the lower part of the face. Flattering options include oval frames, rimless sunglasses, and cat-eye glasses. Oversized sunglasses, wraparound styles, and rectangular frames are perfect choices. ##### Seashell sunglasses. ##### Signature sunglasses. ##### Kemp sunglasses. ## White frame sunglasses. ##### Jodie sunglasses. ## Cat eye sunglasses. ##### Spike sunglasses. ##### 1951C sunglasses. ## Futuristic sunglasses. ##### Runway sunglasses. ##### Sutro sunglasses. ##### Metal frame sunglasses. ##### Crystal sunglasses. ##### Square-frame acetate and gold-tone sunglasses. ##### Wayfarer sunglasses. ## Oversized sunglasses. ##### Square sunglasses. ##### Sunglasses square shape. ## Aviator sunglasses. ##### Aly sunglasses. ##### Dixie sunglasses.",
      'url': 'https://www.voguescandinavia.com/articles/best-eyewear-sunglasses-trends-shop-2025'}]



The second tool is `tools.product_catalog_tool`, which returns the internal sunglasses catalog. Each entry includes details such as product name, ID, description, stock quantity, and price. This structured data will allow the agents to connect online fashion trends with actual items in stock:


```python
tools.product_catalog_tool()
```




    [{'name': 'Aviator',
      'item_id': 'SG001',
      'description': 'Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.',
      'quantity_in_stock': 23,
      'price': 103},
     {'name': 'Wayfarer',
      'item_id': 'SG002',
      'description': 'Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.',
      'quantity_in_stock': 6,
      'price': 92},
     {'name': 'Mystique',
      'item_id': 'SG003',
      'description': 'Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.',
      'quantity_in_stock': 3,
      'price': 88},
     {'name': 'Sport',
      'item_id': 'SG004',
      'description': 'Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.',
      'quantity_in_stock': 11,
      'price': 144},
     {'name': 'Round',
      'item_id': 'SG005',
      'description': 'Circular lenses set in minimalist frames create a thoughtful, artistic appearance. These sunglasses evoke a scholarly or creative vibe while remaining effortlessly stylish.',
      'quantity_in_stock': 10,
      'price': 86}]



With these tools in place, you’ve defined a clear action space and reliable data sources. In the next section, you’ll build the agents that use them to turn raw fashion signals into structured insights and campaign assets.

## 4. Agent Definitions — Building Your Team

Now that you have defined the tools, it’s time to put them to work. In this phase, you will assemble a **marketing research team**, a group of specialized agents that you direct with natural instructions.  

Each agent relies on the tools you introduced earlier, and together they transform raw trend data into a polished campaign report.  We will define them one by one, introducing their role and showing the code that implements each.  

### 4.1. Market Research Agent  

With the **Market Research Agent**, you take the first step in building your campaign.  You ask it to scan the web with `tavily_search_tool` and uncover what’s trending in sunglasses fashion right now. Then you direct it to cross-check those signals against your internal catalog using `product_catalog_tool`, so you know which of your products fit the moment.  

The agent hands back a concise brief: the top fashion insights it found, the products that align with them, and a short explanation of why those picks make sense for your summer push. This gives you a clear, data-driven foundation to shape the rest of the campaign.  

You can now run the following cell to define the **Market Research Agent** in code.  


```python
def market_research_agent(return_messages: bool = False):

    utils.log_agent_title_html("Market Research Agent", "🕵️‍♂️")

    prompt_ = f"""
You are a fashion market research agent tasked with preparing a trend analysis for a summer sunglasses campaign.

Your goal:
1. Explore current fashion trends related to sunglasses using web search.
2. Review the internal product catalog to identify items that align with those trends.
3. Recommend one or more products from the catalog that best match emerging trends.
4. If needed, today date is {datetime.now().strftime("%Y-%m-%d")}.

You can call the following tools:
- tavily_search_tool: to discover external web trends.
- product_catalog_tool: to inspect the internal sunglasses catalog.

Once your analysis is complete, summarize:
- The top 2–3 trends you found.
- The product(s) from the catalog that fit these trends.
- A justification of why they are a good fit for the summer campaign.
"""
    messages = [{"role": "user", "content": prompt_}]
    tools_ = tools.get_available_tools()

    while True:
        response = client.chat.completions.create(
            model="openai:o4-mini",
            messages=messages,
            tools=tools_,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if msg.content:
            utils.log_final_summary_html(msg.content)
            return (msg.content, messages) if return_messages else msg.content

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                utils.log_tool_call_html(tool_call.function.name, tool_call.function.arguments)
                result = tools.handle_tool_call(tool_call)
                utils.log_tool_result_html(result)

                messages.append(msg)
                messages.append(tools.create_tool_response_message(tool_call, result))
        else:
            utils.log_unexpected_html()
            return ("[⚠️ Unexpected: No tool_calls or content returned]", messages) if return_messages else "[⚠️ Unexpected: No tool_calls or content returned]"
```

Let’s try to get some advice from the **Market Research Agent** about our summer sunglasses campaign.  


```python
market_research_result = market_research_agent()
```



<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    🕵️‍♂️ Market Research Agent
  </h2>
</div>





<div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
            background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
  <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
    📞 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">tavily_search_tool</span>
  </div>
  <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
               font-size:13px;white-space:pre-wrap;">{&quot;query&quot;: &quot;2026 summer sunglasses fashion trends&quot;, &quot;max_results&quot;: 5}</code>
</div>





<div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
            background-color:#f1f8e9;color:#33691E;">
  <strong>✅ Tool Result:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">[{&#x27;error&#x27;: &#x27;500 Server Error: Internal Server Error for url: http://jupyter-api-proxy.internal.dlai/rev-proxy/tavily_search_bearer/search&#x27;}]</pre>
</div>





<div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
            background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
  <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
    📞 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">tavily_search_tool</span>
  </div>
  <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
               font-size:13px;white-space:pre-wrap;">{&quot;query&quot;:&quot;2026 summer sunglasses fashion trends&quot;,&quot;max_results&quot;:5}</code>
</div>





<div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
            background-color:#f1f8e9;color:#33691E;">
  <strong>✅ Tool Result:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">[{&#x27;title&#x27;: &#x27;Sunglasses Trends 2026&#x27;, &#x27;content&#x27;: &#x27;The must-have 2026 sunglass trends · Soft oval · Angular square · Gradient tones · Cool metal · Maxi statements · Glasses Chains · Get your style matches.&#x27;, &#x27;url&#x27;: &#x27;https://miaburton.com/en/sunglasses/trends?srsltid=AfmBOoqS6T6Kk8USpzHKM8YlZwEu8fMj798TPFNGeIPe7URsPdSqSJj7&#x27;}, {&#x27;title&#x27;: &#x27;Spring/Summer 2026 Sunglasses: We want you to buy ...&#x27;, &#x27;content&#x27;: &#x27;For Summer 2026, the shape becomes more &quot;angular,&quot; mixing clean lines and a modern roll. This defines &quot;power chic&quot;. It\&#x27;s a current elegance, it&#x27;, &#x27;url&#x27;: &#x27;https://www.opticabassol.com/en-us/blogs/news/sun-glasses-spring-summer-2026-trends?srsltid=AfmBOop9Q89dot7L2aMxAOMQ_wGzNMWz1pUxp_Yb_rkzMYuCx0DV_3y-&#x27;}, {&#x27;title&#x27;: &#x27;Tiny Sunglasses Style Trends For Small Faces in 2026&#x27;, &#x27;content&#x27;: &#x27;Small face? Big style! Discover the 2026 tiny sunglasses trend that combines minimalist flair with bold attitude—perfectly sized frames that&#x27;, &#x27;url&#x27;: &#x27;https://www.kraywoods.com/blogs/our-stories/tiny-sunglasses-fashion-trend?srsltid=AfmBOoqP-NxcmEtTp2mDBvt1gyBI8o_XvHRY7DoKRB_56Bu6KQ6QlgLn&#x27;}, {&#x27;title&#x27;: &quot;The 2026 Eyewear Trends We Can&#x27;t Wait to Wear&quot;, &#x27;content&#x27;: &#x27;Every item on this page was chosen by an ELLE Canada writer. We may earn commission on some of the items you choose to buy. by: Lauren Knowles- Jan 23rd, 2026. On the subject of 2026 eyewear trends, we can expect to see lots of oversized sunnies, sophisticated spectacles, wire-y frames, and more over the next 365 days. Read on as we unpack the coolest styles of the year and where to shop them. ## SHOP THE TREND. ## SHOP THE TREND. ## SHOP THE TREND. ## 2026 Eyewear Trend: Wire Frames. ## More from Trends. ## The Best Jackets to Bundle Up in This Winter. ## The Best Wicked Merch Collaborations Worth Shopping. ## The Gift They’ll Treasure Forever? ## J.T. Hiskey Unleashes a Multi-Album Era. Four albums in one year: J.T. Hiskey’s biggest leap yet. by: Contributor Content- Jan 23rd, 2026. by: Melissa Fejtek- Jan 23rd, 2026. ## The Perfect V-Day Gift? by: ELLE Canada- Jan 20th, 2026.&#x27;, &#x27;url&#x27;: &#x27;https://www.ellecanada.com/fashion/trends/2026-eyewear-trends&#x27;}, {&#x27;title&#x27;: &quot;The biggest women&#x27;s sunglasses trends for 2026&quot;, &#x27;content&#x27;: &quot;The biggest women&#x27;s sunglasses trends for 2026. Visit. Save. Visit. Save. More to explore. Summer Fashion Style. More about this Pin.&quot;, &#x27;url&#x27;: &#x27;https://www.pinterest.com/pin/511651207687359058/&#x27;}]</pre>
</div>





<div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
            background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
  <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
    📞 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">product_catalog_tool</span>
  </div>
  <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
               font-size:13px;white-space:pre-wrap;">{&quot;max_items&quot;:10}</code>
</div>





<div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
            background-color:#f1f8e9;color:#33691E;">
  <strong>✅ Tool Result:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">[{&#x27;name&#x27;: &#x27;Aviator&#x27;, &#x27;item_id&#x27;: &#x27;SG001&#x27;, &#x27;description&#x27;: &#x27;Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.&#x27;, &#x27;quantity_in_stock&#x27;: 23, &#x27;price&#x27;: 103}, {&#x27;name&#x27;: &#x27;Wayfarer&#x27;, &#x27;item_id&#x27;: &#x27;SG002&#x27;, &#x27;description&#x27;: &#x27;Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.&#x27;, &#x27;quantity_in_stock&#x27;: 6, &#x27;price&#x27;: 92}, {&#x27;name&#x27;: &#x27;Mystique&#x27;, &#x27;item_id&#x27;: &#x27;SG003&#x27;, &#x27;description&#x27;: &#x27;Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.&#x27;, &#x27;quantity_in_stock&#x27;: 3, &#x27;price&#x27;: 88}, {&#x27;name&#x27;: &#x27;Sport&#x27;, &#x27;item_id&#x27;: &#x27;SG004&#x27;, &#x27;description&#x27;: &#x27;Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.&#x27;, &#x27;quantity_in_stock&#x27;: 11, &#x27;price&#x27;: 144}, {&#x27;name&#x27;: &#x27;Round&#x27;, &#x27;item_id&#x27;: &#x27;SG005&#x27;, &#x27;description&#x27;: &#x27;Circular lenses set in minimalist frames create a thoughtful, artistic appearance. These sunglasses evoke a scholarly or creative vibe while remaining effortlessly stylish.&#x27;, &#x27;quantity_in_stock&#x27;: 10, &#x27;price&#x27;: 86}]</pre>
</div>





      <div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
                  background-color:#e8f5e9;color:#1B5E20;">
        <strong>✅ Final Summary:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">Trend Analysis Summary (Date: 2026-01-24)

1. Top Trends for Summer 2026  
   • “Power Chic” Angular Squares – Clean lines, bold acetate frames in a squared silhouette.  
   • Cool Metal/Wire Frames – Lightweight metal constructions or ultra-thin wire profiles for a minimalist, high-tech look.  
   • Maxi Statements – Oversized lenses that offer maximum coverage and a strong style statement.

2. Recommended Catalog Items  
   • Wayfarer (SG002)  
   • Aviator (SG001)

3. Justification  
   – Wayfarer (SG002): Its thick, angular acetate frame directly echoes the “Power Chic” square-frame trend. The confident silhouette makes a statement in any summer wardrobe.  
   – Aviator (SG001): Features classic teardrop lenses set in thin metal wire frames—matching both the Cool Metal/Wire Frames and Maxi Statement trends with its large-coverage lenses and lightweight construction.  

These two models give the campaign a balanced offering—one bold acetate shape and one refined metal silhouette—capitalizing on the season’s strongest sunglasses directions.</pre>
      </div>



Next, you’ll turn this brief into a visual concept with the Graphic Designer Agent.

### 4.2. Graphic Designer Agent  

With the **Graphic Designer Agent**, you move from analysis to creativity.  
You take the brief from your Market Research Agent and ask this one to translate it into a visual concept.  
Because `aisuite` does not yet support direct image generation (like DALL·E), you guide the process in two stages:  

1. First, the agent uses `aisuite` with an OpenAI text model (`o4-mini`) to craft a vivid **prompt** and a short, engaging **caption**.  
2. Then, the prompt is sent to OpenAI’s `dall-e-3` API to generate the **campaign image** itself.  

The result gives you everything you need to move forward: the generated image (saved locally for reuse), the exact prompt that produced it (useful for iteration), and a polished caption for campaign storytelling.  

<div style="border:1px solid #fca5a5; border-left:6px solid #ef4444; background:#fee2e2; border-radius:6px; padding:12px 14px; color:#111827; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
  <strong>Note:</strong> At this point, <code>aisuite</code> does <strong>not support direct image generation</strong>.  
  That’s why you combine its text-based output (prompt + caption) with OpenAI’s <code>dall-e-3</code> to produce the final campaign visual.
</div>  

You can now run the following cell to define the **Graphic Designer Agent** in code.  


```python
def graphic_designer_agent(trend_insights: str, caption_style: str = "short punchy", size: str = "1024x1024") -> dict:

    """
    Uses aisuite to generate a marketing prompt/caption and OpenAI (directly) to generate the image.

    Args:
        trend_insights (str): Trend summary from the researcher agent.
        caption_style (str): Optional style hint for caption.
        size (str): Image resolution (e.g., '1024x1024').

    Returns:
        dict: A dictionary with image_url, prompt, and caption.
    """

    utils.log_agent_title_html("Graphic Designer Agent", "🎨")

    # Step 1: Generate prompt and caption using aisuite
    system_message = (
        "You are a visual marketing assistant. Based on the input trend insights, "
        "write a creative and visual prompt for an AI image generation model, and also a short caption."
    )

    user_prompt = f"""
Trend insights:
{trend_insights}

Please output:
1. A vivid, descriptive prompt to guide image generation.
2. A marketing caption in style: {caption_style}.

Respond in this format:
{{"prompt": "...", "caption": "..."}}
"""

    chat_response = client.chat.completions.create(
        model="openai:o4-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = chat_response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {"error": "No JSON returned", "raw": content}

    prompt = parsed["prompt"]
    caption = parsed["caption"]

    # Step 2: Generate image directly using openai-python
    openai_client = openai.OpenAI()

    image_response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
        response_format="url"
    )

    image_url = image_response.data[0].url

    # Save image locally
    img_bytes = requests.get(image_url).content
    img = Image.open(BytesIO(img_bytes))

    filename = os.path.basename(image_url.split("?")[0])
    image_path = filename
    img.save(image_path)


    # Log summary with local image
    utils.log_final_summary_html(f"""
        <h3>Generated Image and Caption</h3>

        <p><strong>Image Path:</strong> <code>{image_path}</code></p>

        <p><strong>Generated Image:</strong></p>
        <img src="{image_path}" alt="Generated Image" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;">

        <p><strong>Prompt:</strong> {prompt}</p>
    """)


    return {
        "image_url": image_url,
        "prompt": prompt,
        "caption": caption,
        "image_path": image_path  
    }


```

Now let’s run the `graphic_designer_agent` to generate a campaign image, using the trend insights provided by the **Market Research Agent**.


```python
graphic_designer_agent_result = graphic_designer_agent(
    trend_insights=market_research_result,
)

```



<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    🎨 Graphic Designer Agent
  </h2>
</div>





<div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
            background-color:#e8f5e9;color:#1B5E20;">
  <strong>✅ Final Summary:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">&lt;h3&gt;Generated Image and Caption&lt;/h3&gt;

  &lt;p&gt;&lt;strong&gt;Image Path:&lt;/strong&gt; &lt;code&gt;img-pm1wjMOKflspvr611HeSF1D0.png&lt;/code&gt;&lt;/p&gt;

  &lt;p&gt;&lt;strong&gt;Generated Image:&lt;/strong&gt;&lt;/p&gt;
  &lt;img src=&quot;img-pm1wjMOKflspvr611HeSF1D0.png&quot; alt=&quot;Generated Image&quot; style=&quot;max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;&quot;&gt;

  &lt;p&gt;&lt;strong&gt;Prompt:&lt;/strong&gt; A sun-drenched rooftop terrace at golden hour: in the foreground, a model wears SG002 Wayfarer sunglasses—bold, angular matte-black acetate frames casting crisp geometric shadows across their cheekbones. Beside them, another model sports SG001 Aviator sunglasses—slender gold wire frames and oversized teardrop lenses that mirror the glowing city skyline and swaying palms. Warm sunset hues, high-contrast lighting, and a subtle lens flare evoke the ultimate summer power-chic vibe.&lt;/p&gt;</pre>
</div>



With a visual in hand, you’ll craft the campaign voice using the Copywriter Agent.

### 4.3. Copywriter Agent  

Once the **Market Research Agent** and **Graphic Designer Agent** have done their work, you now turn to the **Copywriter Agent**. With both the campaign image and the trend summary in hand, you ask this agent to create the voice of your campaign.  

It takes the visual and the analysis together as multimodal input and crafts a short, elegant marketing quote that captures the essence of the message. Along with the quote, it gives you a clear justification—why the phrase fits the image and how it ties back to the trends.  

This way, you don’t just get a catchy line, you also get the reasoning behind it, making it easier to defend and refine in front of stakeholders.  




```python
def copywriter_agent(image_path: str, trend_summary: str, model: str = "openai:o4-mini") -> dict:

    """
    Uses aisuite (OpenAI only) to send an image and a trend summary and return a campaign quote.

    Args:
        image_path (str): URL of the image to be analyzed.
        trend_summary (str): Text from the researcher agent.
        model (str): OpenAI model (e.g., openai:o4-mini, openai:gpt-4o)

    Returns:
        dict: {
            "quote": "...",
            "justification": "...",
            "image_path": "..."
        }
    """

    utils.log_agent_title_html("Copywriter Agent", "✍️")

    # Step 1: Load local image and encode as base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Step 2: Build OpenAI-compliant multimodal message
    messages = [
        {
            "role": "system",
            "content": "You are a copywriter that creates elegant campaign quotes based on an image and a marketing trend summary."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "auto"
                    }
                },
                {
                    "type": "text",
                    "text": f"""
Here is a visual marketing image and a trend analysis:

Trend summary:
\"\"\"{trend_summary}\"\"\"

Please return a JSON object like:
{{
  "quote": "A short, elegant campaign phrase (max 12 words)",
  "justification": "Why this quote matches the image and trend"
}}"""
                }
            ]
        }
    ]

    # Step 3: Send request via aisuite
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Step 4: Parse JSON response
    content = response.choices[0].message.content.strip()

    utils.log_final_summary_html(content)

    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"error": "No valid JSON returned"}
    except Exception as e:
        parsed = {"error": f"Failed to parse: {e}", "raw": content}


    parsed["image_path"] = image_path
    return parsed

```

Next, let’s call the Copywriter Agent to generate a short campaign quote based on the marketing image and the trend insights produced earlier.


```python
copywriter_agent_result = copywriter_agent(
    image_path=graphic_designer_agent_result["image_path"],
    trend_summary=market_research_result,
)
```



<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    ✍️ Copywriter Agent
  </h2>
</div>





<div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
            background-color:#e8f5e9;color:#1B5E20;">
  <strong>✅ Final Summary:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">{&quot;quote&quot;:&quot;Bold Angles, Sleek Metal: Your Golden Summer Statement&quot;,&quot;justification&quot;:&quot;This phrase captures the image’s juxtaposition of power-chic square acetate and minimalist metal aviators bathed in golden light, directly reflecting Summer 2026’s top trends—“Power Chic” angular frames and cool wire silhouettes with maxi coverage.&quot;}</pre>
</div>



With a quote and justification ready, you’ll package everything into an executive-ready artifact using the Packaging Agent.

### 4.4. Packaging Agent  

Finally, you bring in the **Packaging Agent** to tie everything together. After the **Market Research Agent**, **Graphic Designer Agent**, and **Copywriter Agent** have each contributed their part, this agent compiles the entire story into one polished artifact.  

You ask it to take the trend summary, the campaign visual, the crafted quote, and the justification, and assemble them into an executive-ready markdown report. Along the way, it rewrites the trend insights for clarity and tone, ensures the quote is styled properly with the image, and organizes everything so the final document looks professional and persuasive.  

With this step, you end up with a complete campaign package—easy to share, visually engaging, and ready for CEO-level review.  


```python
def packaging_agent(trend_summary: str, image_url: str, quote: str, justification: str, output_path: str = "campaign_summary.md") -> str:

    """
    Packages the campaign assets into a beautifully formatted markdown report for executive review.

    Args:
        trend_summary (str): Summary of the market trends.
        image_url (str): URL of the campaign image.
        quote (str): Marketing quote to overlay.
        justification (str): Explanation for the quote.
        output_path (str): Path to save the markdown report.

    Returns:
        str: Path to the saved markdown file.
    """

    utils.log_agent_title_html("Packaging Agent", "📦")

    # We use this path in the src of the <img>
    styled_image_html = f"""
![Open the generated file to see]({image_url})
    """

    beautified_summary = client.chat.completions.create(
        model="openai:o4-mini",
        messages=[
            {"role": "system", "content": "You are a marketing communication expert writing elegant campaign summaries for executives."},
            {"role": "user", "content": f"""
Please rewrite the following trend summary to be clear, professional, and engaging for a CEO audience:

\"\"\"{trend_summary.strip()}\"\"\"
"""}
        ]
    ).choices[0].message.content.strip()

    utils.log_tool_result_html(beautified_summary)

    # Combine all parts into markdown
    markdown_content = f"""# 🕶️ Summer Sunglasses Campaign – Executive Summary

## 📊 Refined Trend Insights
{beautified_summary}

## 🎯 Campaign Visual
{styled_image_html}

## ✍️ Campaign Quote
{quote.strip()}

## ✅ Why This Works
{justification.strip()}

---

*Report generated on {datetime.now().strftime('%Y-%m-%d')}*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_path


```

With your trend summary, campaign image, and quote ready, you now hand everything to the **Packaging Agent**. Its job is to pull these pieces together into a polished, executive-ready report. Run the next cell to generate it.  



```python
packaging_agent_result = packaging_agent(
    trend_summary=market_research_result,
    image_url=graphic_designer_agent_result["image_path"],
    quote=copywriter_agent_result["quote"],
    justification=copywriter_agent_result["justification"],
    output_path=f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
)
```



<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    📦 Packaging Agent
  </h2>
</div>





      <div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
                  background-color:#f1f8e9;color:#33691E;">
        <strong>✅ Tool Result:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">Trend Analysis – Summer 2026 (as of January 24, 2026)

Executive Summary  
We’ve isolated three defining eyewear trends set to drive consumer demand next summer. By featuring one bold acetate silhouette and one refined metal design, our campaign will speak directly to both the Power Chic and Cool Metal movements, while capitalizing on oversized, statement-making styles.

1. Key Summer ’26 Trends  
   • Power Chic (Angular Squares): Strong, squared acetate frames with sharp lines for an authoritative, fashion-forward look.  
   • Cool Metal/Wire Frames: Ultra-lightweight metal and wire constructions that emphasize technical precision and minimalist elegance.  
   • Maxi Statements: Large, full-coverage lenses that blend functionality and high-impact style.

2. Recommended Styles  
   • Wayfarer (SG002) – A thick, angular acetate frame that embodies the Power Chic trend with bold geometry and confident presence.  
   • Aviator (SG001) – Classic teardrop lenses in a slender metal wire frame, delivering both the Cool Metal aesthetic and the oversized coverage of the Maxi Statement.

3. Strategic Rationale  
   – Distinctive Balance: Pairing SG002’s assertive acetate shape with SG001’s sleek metal form addresses the two strongest customer appetites—powerful self-expression and refined minimalism.  
   – Market Appeal: Oversized coverage meets technical craftsmanship, ensuring broad consumer resonance across fashion-driven and performance-oriented segments.  
   – Season-Long Relevance: These designs will headline summer collections, offering versatile styling for both premium and accessible price points.

Next Steps  
Finalize photography and creative treatments to highlight each frame’s unique attributes. Position SG002 and SG001 as the cornerstone pieces in digital, retail and wholesale presentations. This targeted approach will maximize seasonal impact and reinforce our leadership in trend-forward eyewear.</pre>
      </div>



The final result will be a beautifully formatted campaign report that you can view directly in the notebook. It will include:  

- A refined trend summary that you will see rewritten for executive clarity  
- A visually styled image with your campaign quote overlaid using HTML  
- A clear justification so you understand why the visual and message align with current trends  
- A timestamp showing you exactly when the report was generated  

You can view it with:  


```python
# Load and render the markdown content
with open(packaging_agent_result, "r", encoding="utf-8") as f:
    md_content = f.read()

display(Markdown(md_content))

```


# 🕶️ Summer Sunglasses Campaign – Executive Summary

## 📊 Refined Trend Insights
Trend Analysis – Summer 2026 (as of January 24, 2026)

Executive Summary  
We’ve isolated three defining eyewear trends set to drive consumer demand next summer. By featuring one bold acetate silhouette and one refined metal design, our campaign will speak directly to both the Power Chic and Cool Metal movements, while capitalizing on oversized, statement-making styles.

1. Key Summer ’26 Trends  
   • Power Chic (Angular Squares): Strong, squared acetate frames with sharp lines for an authoritative, fashion-forward look.  
   • Cool Metal/Wire Frames: Ultra-lightweight metal and wire constructions that emphasize technical precision and minimalist elegance.  
   • Maxi Statements: Large, full-coverage lenses that blend functionality and high-impact style.

2. Recommended Styles  
   • Wayfarer (SG002) – A thick, angular acetate frame that embodies the Power Chic trend with bold geometry and confident presence.  
   • Aviator (SG001) – Classic teardrop lenses in a slender metal wire frame, delivering both the Cool Metal aesthetic and the oversized coverage of the Maxi Statement.

3. Strategic Rationale  
   – Distinctive Balance: Pairing SG002’s assertive acetate shape with SG001’s sleek metal form addresses the two strongest customer appetites—powerful self-expression and refined minimalism.  
   – Market Appeal: Oversized coverage meets technical craftsmanship, ensuring broad consumer resonance across fashion-driven and performance-oriented segments.  
   – Season-Long Relevance: These designs will headline summer collections, offering versatile styling for both premium and accessible price points.

Next Steps  
Finalize photography and creative treatments to highlight each frame’s unique attributes. Position SG002 and SG001 as the cornerstone pieces in digital, retail and wholesale presentations. This targeted approach will maximize seasonal impact and reinforce our leadership in trend-forward eyewear.

## 🎯 Campaign Visual

![Open the generated file to see](img-pm1wjMOKflspvr611HeSF1D0.png)
    

## ✍️ Campaign Quote
Bold Angles, Sleek Metal: Your Golden Summer Statement

## ✅ Why This Works
This phrase captures the image’s juxtaposition of power-chic square acetate and minimalist metal aviators bathed in golden light, directly reflecting Summer 2026’s top trends—“Power Chic” angular frames and cool wire silhouettes with maxi coverage.

---

*Report generated on 2026-01-24*



Finally, you’ll wrap the entire workflow into a single callable function to run the full pipeline in one step.

## 5. Full Campaign Pipeline – `run_sunglasses_campaign_pipeline`

In this step, you will define a single function, `run_sunglasses_campaign_pipeline`, that ties all the pieces together into one seamless workflow for your summer sunglasses campaign.  

The function will:  
- Run market research to scan fashion trends and match them to your catalog.  
- Generate a visually styled image and caption.  
- Create a short, elegant campaign quote with justification.  
- Package everything into a polished markdown report tailored for executive review.  

By defining this function, you make it easy to run the **entire pipeline in one call** while still being able to trace intermediate results and view the final report.  


```python
def run_sunglasses_campaign_pipeline(output_path: str = "campaign_summary.md") -> dict:
    """
    Runs the full summer sunglasses campaign pipeline:
    1. Market research (search trends + match products)
    2. Generate visual + caption
    3. Generate quote based on image + trend
    4. Create executive markdown report

    Returns:
        dict: Dictionary containing all intermediate results + path to final report
    """
    # 1. Run market research agent
    trend_summary = market_research_agent()
    print("✅ Market research completed")

    # 2. Generate image + caption
    visual_result = graphic_designer_agent(trend_insights=trend_summary)
    image_path = visual_result["image_path"]
    print("🖼️ Image generated")

    # 3. Generate quote based on image + trends
    quote_result = copywriter_agent(image_path=image_path, trend_summary=trend_summary)
    quote = quote_result.get("quote", "")
    justification = quote_result.get("justification", "")
    print("💬 Quote created")

    # 4. Generate markdown report
    md_path = packaging_agent(
        trend_summary=trend_summary,
        image_url=image_path,  
        quote=quote,
        justification=justification,
        output_path=f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    )

    print(f"📦 Report generated: {md_path}")

    return {
        "trend_summary": trend_summary,
        "visual": visual_result,
        "quote": quote_result,
        "markdown_path": md_path
    }

```

You can now create a complete campaign report by running the pipeline in a single call. Just execute the next cell:  


```python
results = run_sunglasses_campaign_pipeline()
```



<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    🕵️‍♂️ Market Research Agent
  </h2>
</div>





<div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
            background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
  <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
    📞 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">tavily_search_tool</span>
  </div>
  <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
               font-size:13px;white-space:pre-wrap;">{&quot;query&quot;: &quot;2026 summer sunglasses fashion trends&quot;, &quot;max_results&quot;: 5}</code>
</div>





<div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
            background-color:#f1f8e9;color:#33691E;">
  <strong>✅ Tool Result:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">[{&#x27;title&#x27;: &#x27;Sunglasses Trends 2026&#x27;, &#x27;content&#x27;: &#x27;The must-have 2026 sunglass trends · Soft oval · Angular square · Gradient tones · Cool metal · Maxi statements · Glasses Chains · Get your style matches.&#x27;, &#x27;url&#x27;: &#x27;https://miaburton.com/en/sunglasses/trends?srsltid=AfmBOoqS6T6Kk8USpzHKM8YlZwEu8fMj798TPFNGeIPe7URsPdSqSJj7&#x27;}, {&#x27;title&#x27;: &#x27;Spring/Summer 2026 Sunglasses: We want you to buy ...&#x27;, &#x27;content&#x27;: &#x27;For Summer 2026, the shape becomes more &quot;angular,&quot; mixing clean lines and a modern roll. This defines &quot;power chic&quot;. It\&#x27;s a current elegance, it&#x27;, &#x27;url&#x27;: &#x27;https://www.opticabassol.com/en-us/blogs/news/sun-glasses-spring-summer-2026-trends?srsltid=AfmBOop9Q89dot7L2aMxAOMQ_wGzNMWz1pUxp_Yb_rkzMYuCx0DV_3y-&#x27;}, {&#x27;title&#x27;: &#x27;Tiny Sunglasses Style Trends For Small Faces in 2026&#x27;, &#x27;content&#x27;: &#x27;Small face? Big style! Discover the 2026 tiny sunglasses trend that combines minimalist flair with bold attitude—perfectly sized frames that&#x27;, &#x27;url&#x27;: &#x27;https://www.kraywoods.com/blogs/our-stories/tiny-sunglasses-fashion-trend?srsltid=AfmBOoqP-NxcmEtTp2mDBvt1gyBI8o_XvHRY7DoKRB_56Bu6KQ6QlgLn&#x27;}, {&#x27;title&#x27;: &quot;The 2026 Eyewear Trends We Can&#x27;t Wait to Wear&quot;, &#x27;content&#x27;: &#x27;Every item on this page was chosen by an ELLE Canada writer. We may earn commission on some of the items you choose to buy. by: Lauren Knowles- Jan 23rd, 2026. On the subject of 2026 eyewear trends, we can expect to see lots of oversized sunnies, sophisticated spectacles, wire-y frames, and more over the next 365 days. Read on as we unpack the coolest styles of the year and where to shop them. ## SHOP THE TREND. ## SHOP THE TREND. ## SHOP THE TREND. ## 2026 Eyewear Trend: Wire Frames. ## More from Trends. ## The Best Jackets to Bundle Up in This Winter. ## The Best Wicked Merch Collaborations Worth Shopping. ## The Gift They’ll Treasure Forever? ## J.T. Hiskey Unleashes a Multi-Album Era. Four albums in one year: J.T. Hiskey’s biggest leap yet. by: Contributor Content- Jan 23rd, 2026. by: Melissa Fejtek- Jan 23rd, 2026. ## The Perfect V-Day Gift? by: ELLE Canada- Jan 20th, 2026.&#x27;, &#x27;url&#x27;: &#x27;https://www.ellecanada.com/fashion/trends/2026-eyewear-trends&#x27;}, {&#x27;title&#x27;: &quot;The biggest women&#x27;s sunglasses trends for 2026&quot;, &#x27;content&#x27;: &quot;The biggest women&#x27;s sunglasses trends for 2026. Visit. Save. Visit. Save. More to explore. Summer Fashion Style. More about this Pin.&quot;, &#x27;url&#x27;: &#x27;https://www.pinterest.com/pin/511651207687359058/&#x27;}]</pre>
</div>





<div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
            background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
  <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
    📞 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">product_catalog_tool</span>
  </div>
  <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
               font-size:13px;white-space:pre-wrap;">{&quot;max_items&quot;:10}</code>
</div>





<div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
            background-color:#f1f8e9;color:#33691E;">
  <strong>✅ Tool Result:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">[{&#x27;name&#x27;: &#x27;Aviator&#x27;, &#x27;item_id&#x27;: &#x27;SG001&#x27;, &#x27;description&#x27;: &#x27;Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.&#x27;, &#x27;quantity_in_stock&#x27;: 23, &#x27;price&#x27;: 103}, {&#x27;name&#x27;: &#x27;Wayfarer&#x27;, &#x27;item_id&#x27;: &#x27;SG002&#x27;, &#x27;description&#x27;: &#x27;Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.&#x27;, &#x27;quantity_in_stock&#x27;: 6, &#x27;price&#x27;: 92}, {&#x27;name&#x27;: &#x27;Mystique&#x27;, &#x27;item_id&#x27;: &#x27;SG003&#x27;, &#x27;description&#x27;: &#x27;Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.&#x27;, &#x27;quantity_in_stock&#x27;: 3, &#x27;price&#x27;: 88}, {&#x27;name&#x27;: &#x27;Sport&#x27;, &#x27;item_id&#x27;: &#x27;SG004&#x27;, &#x27;description&#x27;: &#x27;Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.&#x27;, &#x27;quantity_in_stock&#x27;: 11, &#x27;price&#x27;: 144}, {&#x27;name&#x27;: &#x27;Round&#x27;, &#x27;item_id&#x27;: &#x27;SG005&#x27;, &#x27;description&#x27;: &#x27;Circular lenses set in minimalist frames create a thoughtful, artistic appearance. These sunglasses evoke a scholarly or creative vibe while remaining effortlessly stylish.&#x27;, &#x27;quantity_in_stock&#x27;: 10, &#x27;price&#x27;: 86}]</pre>
</div>





      <div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
                  background-color:#e8f5e9;color:#1B5E20;">
        <strong>✅ Final Summary:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">Here’s the summer-2026 sunglasses trend analysis and catalog match:

1. Top Trends  
   • Angular Power Chic: Sharp, square-edged frames signal confidence and modern elegance.  
   • Cool Metal &amp; Gradient Tones: Lightweight metal frames—often finished with gradient lenses—are having a moment for their sleek, tech-inspired look.  
   • Oversized Statement Shapes: Maxi silhouettes remain strong, balancing bold style with extra sun protection.

2. Recommended Products  
   • Wayfarer (SG002)  
     – Why it fits: Thick, angular acetate frames capture the “power chic” square-shape trend and double as an oversized statement. Its sturdy build gives that confident edge shoppers want.  
   • Aviator (SG001)  
     – Why it fits: Classic metal construction aligns perfectly with the cool-metal trend. Its large teardrop lenses can be offered in gradient tints for both style and superior UV coverage.

3. Justification for Summer Campaign  
   – Both styles tap into core 2026 preferences—strong silhouettes and metallic accents—while offering broad appeal (unisex, versatile).  
   – High stock levels (23 Aviators, 6 Wayfarers) mean we can support a robust summer rollout.  
   – Their combination of fashion-forward design and functional sun protection makes them ideal hero SKUs for driving seasonal lift.</pre>
      </div>



    ✅ Market research completed




<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    🎨 Graphic Designer Agent
  </h2>
</div>





<div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
            background-color:#e8f5e9;color:#1B5E20;">
  <strong>✅ Final Summary:</strong>
  <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">&lt;h3&gt;Generated Image and Caption&lt;/h3&gt;

  &lt;p&gt;&lt;strong&gt;Image Path:&lt;/strong&gt; &lt;code&gt;img-k5sdyuGCj9MrF9tWDl6QxFcG.png&lt;/code&gt;&lt;/p&gt;

  &lt;p&gt;&lt;strong&gt;Generated Image:&lt;/strong&gt;&lt;/p&gt;
  &lt;img src=&quot;img-k5sdyuGCj9MrF9tWDl6QxFcG.png&quot; alt=&quot;Generated Image&quot; style=&quot;max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;&quot;&gt;

  &lt;p&gt;&lt;strong&gt;Prompt:&lt;/strong&gt; A confident duo poses on a rooftop at golden hour against a neon-infused city skyline: one model in a crisp white blazer wears thick, angular black acetate wayfarers (SG002), radiating power chic; the other in a metallic bomber jacket sports sleek silver aviators (SG001) with icy blue-to-violet gradient lenses. Oversized silhouettes catch the glowing sunset and urban reflections, while dramatic side lighting and minimalist futuristic architecture frame their bold, modern elegance.&lt;/p&gt;</pre>
</div>



    🖼️ Image generated




<div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
  <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
    ✍️ Copywriter Agent
  </h2>
</div>



### 5.1. Results

Run the following cell to view the outputs generated by the full campaign pipeline.


```python
with open(results["markdown_path"], "r", encoding="utf-8") as f:
    md_content = f.read()
display(Markdown(md_content))

```

## 6. Key Takeaways  

By completing this lab, you have seen how to:  

- Use **multi-agent LLM pipelines** to automate a creative workflow end-to-end.  
- Combine **reasoning, tool-calling, and external data** to ground your outputs in reality.  
- Apply multimodal models (like `gpt-4o`) that process **both text and images** for tasks such as generating campaign quotes.  
- Extend the model’s abilities with tools (`tavily_search_tool`, `product_catalog_tool`) so your outputs are not only imaginative but also practical.  
- Keep execution **transparent and debuggable** with structured logging and HTML-styled blocks.  
- Deliver a polished, **executive-ready report** in Markdown format that blends insights, visuals, and justifications into a single artifact.  



<div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">

🎉 <strong>Congratulations!</strong> 🎉  

Now you have successfully built and run a **multi-agent pipeline**: you researched trends, generated visuals, crafted a campaign quote, and packaged everything into an **executive-ready report**.  

This workflow shows you how to combine the **creativity of LLMs** with the **discipline of structured orchestration**, giving you a repeatable pattern you can adapt to many real-world scenarios. 🌟  
</div>



```python

```


```python

```


```python

```

=========

When you have a team of people working together, the patterns by which they communicate can be
0:06
quite complex. And in fact, designing an organizational chart is actually pretty
0:10
complex to try to figure out what's the best way for people to communicate, to collaborate.
0:16
It turns out, designing communication patterns for multi-agent systems is also quite complex.
0:22
But let me show you some of the most common design patterns I see used by different teams today.
0:26
In a marketing team with a linear plan, where first a researcher worked, then a graphic designer,
0:31
then a writer, the communication pattern was linear. The researcher would communicate with
0:36
the graphic designer, and in both the research and the graphic designer, maybe pass the outputs to
0:40
the writer. And so there's a very linear communication pattern. This is one of the
0:46
two most common communication plans that I see being used today. The second of the two most
0:51
common communication plans would be similar to what you saw in this example, with planning using
0:58
multiple agents, where there is a manager that communicates with a number of team members and
1:05
coordinates their work. So in this example, the marketing manager decides to call on the researcher
1:10
to do some work. Then if you think of the marketing manager as getting the report back, and then
1:15
sending it to the graphic designer, getting a report back, and then sending it to the writer, this would be a
1:20
hierarchical communication pattern. If you're actually implementing a hierarchical communication
1:25
pattern, it'll probably be simpler to have the researcher pass the report back to the marketing
1:29
manager, rather than the researcher pass the results directly to the graphic designer and to
1:34
the writer. But so this type of hierarchy is also a pretty common way to plan the communication
1:40
patterns, where you have one manager coordinating the work of a number of other agents. And just to
1:45
share with you some more advanced and less frequently used, but nonetheless sometimes used in
1:50
practice communication patterns, one would be a deeper hierarchy, where same as before, if you have
1:56
a marketing manager send tasks to the researcher, graphic designer, writer, but maybe the researcher has
2:01
themselves two other agents that they call on, such as a web researcher and a fact checker. Maybe the
2:07
graphic designer just works by themselves, whereas the writer has an initial style writer and a citation
2:13
checker. So this would be a hierarchical organization of agents, in which some agents
2:19
might themselves call other sub-agents. And I also see this used in some applications, but this is
2:25
much more complex than a one-level hierarchy, so used less often today. And then one final pattern
2:31
that is quite challenging to execute, but I see a few experimental projects use it, is the all-to-all
2:38
communication pattern. So in this pattern, anyone is allowed to talk to anyone else at any time. And
2:44
the way you implement this is you prompt all four of your agents, in this case, to tell them that
2:50
there are three other agents they could decide to call on. And whenever one of your agents decides
2:55
to send a message to another agent, that message gets added to the receiver agent's contacts. And
3:02
then a receiver agent can think for a while and decide when to get back to that first agent. And
3:07
so if you can all collaborate in a crowd and talk to each other for a while until, say, each of them
3:13
declares that it is done with this task, and then it starts talking. And maybe when everyone thinks
3:18
it's done, or maybe when the writer concludes it's good enough, that's when you generate the final
3:22
output. In practice, I find the results of all-to-all communication patterns a bit hard to predict.
3:27
So some applications don't need high control. You can run it and see what you get. If the marketing
3:32
brochure isn't good, maybe that's okay. You just run it again and see if you get a different result.
3:37
But I think for applications where you're willing to tolerate a little bit of chaos and
3:41
unpredictability, I do see some developers using this communication pattern. So that, I hope,
3:47
conveys some of the richness of multi-agent systems. Today, there are quite a lot of software
3:54
frameworks as well that support easily building multi-agent systems. And they also make implementing
4:00
some of these communication patterns relatively easy. So maybe if you use your own multi-agent system,
4:06
you'll find some of these frameworks hopeful for exploring these different
4:09
communication patterns as well. And so that now brings us to the final video
4:17
of this module and of this course. Let's go on to the final video to wrap up.
==========
# Graded Lab: Agentic Workflows

In this lab, you will build an agentic system that generates a short research report through planning, external tool usage, and feedback integration. Your workflow will involve:

### Agents

* **Planning Agent / Writer**: Creates an outline and coordinates tasks.
* **Research Agent**: Gathers external information using tools like Arxiv, Tavily, and Wikipedia.
* **Editor Agent**: Reflects on the report and provides suggestions for improvement.

---
<a name='submission'></a>

<h4 style="color:green; font-weight:bold;">TIPS FOR SUCCESSFUL GRADING OF YOUR ASSIGNMENT:</h4>

* All cells are frozen except for the ones where you need to write your solution code or when explicitly mentioned you can interact with it.

* In each exercise cell, look for comments `### START CODE HERE ###` and `### END CODE HERE ###`. These show you where to write the solution code. **Do not add or change any code that is outside these comments**.

* You can add new cells to experiment but these will be omitted by the grader, so don't rely on newly created cells to host your solution code, use the provided places for this.

* Avoid using global variables unless you absolutely have to. The grader tests your code in an isolated environment without running all cells from the top. As a result, global variables may be unavailable when scoring your submission. Global variables that are meant to be used will be defined in UPPERCASE.

* To submit your notebook for grading, first save it by clicking the 💾 icon on the top left of the page and then click on the <span style="background-color: red; color: white; padding: 3px 5px; font-size: 16px; border-radius: 5px;">Submit assignment</span> button on the top right of the page.
---


### Research Tools

By importing `research_tools`, you gain access to several search utilities:

- `research_tools.arxiv_search_tool(query)` → search academic papers from **arXiv**  

  *Example:* `research_tools.arxiv_search_tool("neural networks for climate modeling")`

- `research_tools.tavily_search_tool(query)` → perform web searches with the **Tavily API**  

  *Example:* `research_tools.tavily_search_tool("latest trends in sunglasses fashion")`

- `research_tools.wikipedia_search_tool(query)` → retrieve summaries from **Wikipedia**  

  *Example:* `research_tools.wikipedia_search_tool("Ensemble Kalman Filter")`

Run the cell below to make them available.


```python
# =========================
# Imports
# =========================

# --- Standard library 
from datetime import datetime
import re
import json
import ast


# --- Third-party ---
from IPython.display import Markdown, display
from aisuite import Client

# --- Local / project ---
import research_tools
```


```python
import unittests
```

### Initialize client

Create a shared client instance for upcoming calls.


```python
CLIENT = Client()
```

## Exercise 1: planner_agent

### Objective
Correctly set up a call to a language model (LLM) to generate a research plan.

### Instructions

1. **Focus Areas**:
   - Ensure `CLIENT.chat.completions.create` is correctly configured.
   - Pass the `model` and `messages` parameters correctly:
     - **Model**: Use `"openai:o4-mini"` by default.
     - **Messages**: Set with `{"role": "user", "content": user_prompt}`.
     - **Temperature**: Fixed at 1 for creative outputs.

### Notes

- The prompt is pre-defined and guides the LLM on task requirements.
- Only return a formatted list of steps — no extra text.

Focus on the LLM call setup to complete the task.


```python
# GRADED FUNCTION: planner_agent

def planner_agent(topic: str, model: str = "openai:o4-mini") -> list[str]:
    """
    Generates a plan as a Python list of steps (strings) for a research workflow.

    Args:
        topic (str): Research topic to investigate.
        model (str): Language model to use.

    Returns:
        List[str]: A list of executable step strings.
    """

    
    # Build the user prompt
    user_prompt = f"""
    You are a planning agent responsible for organizing a research workflow with multiple intelligent agents.

    🧠 Available agents:
    - A research agent who can search the web, Wikipedia, and arXiv.
    - A writer agent who can draft research summaries.
    - An editor agent who can reflect and revise the drafts.

    🎯 Your job is to write a clear, step-by-step research plan **as a valid Python list**, where each step is a string.
    Each step should be atomic, executable, and must rely only on the capabilities of the above agents.

    🚫 DO NOT include irrelevant tasks like "create CSV", "set up a repo", "install packages", etc.
    ✅ DO include real research-related tasks (e.g., search, summarize, draft, revise).
    ✅ DO assume tool use is available.
    ✅ DO NOT include explanation text — return ONLY the Python list.
    ✅ The final step should be to generate a Markdown document containing the complete research report.

    Topic: "{topic}"
    """

    # Add the user prompt to the messages list
    messages = [{"role": "user", "content": user_prompt}]

    ### START CODE HERE ###

    # Call the LLM
    response = CLIENT.chat.completions.create( 
        # Pass in the model
        model=model,
        # Define the messages. Remember this is meant to be a user prompt!
        messages=messages,
        # Keep responses creative
        temperature=1, 
    )

    ### END CODE HERE ###

    # Extract message from response
    steps_str = response.choices[0].message.content.strip()

    # Parse steps
    steps = ast.literal_eval(steps_str)

    return steps
```


```python
# Test your code!
unittests.test_planner_agent(planner_agent)
```

    [92m All tests passed!


## Exercise 2: research_agent

### Objective
Set up a call to a language model (LLM) to perform a research task using various tools.

### Instructions

**Focus Areas**:

- **Creating a Custom Prompt**:
  - **Define the Role**: Clearly specify the role, such as "research assistant."
  - **List Available Tools** (as strings inside the prompt, not the actual functions):
    - Use `arxiv_tool` to find academic papers.
    - Use `tavily_tool` for general web searches.
    - Use `wikipedia_tool` for accessing encyclopedic knowledge.
  - **Specify the Task**: Include a placeholder in your prompt for defining the specific task that needs to be accomplished.
  - **Include Date Information**: Add a placeholder for the current date or time to provide context.

- **Creating Messages Dict**:
  - Ensure the `messages` are correctly set with `{"role": "user", "content": prompt}`.

- **Creating Tools List**:
  - Create a list of tools for use, such as `research_tools.arxiv_search_tool`, `research_tools.tavily_search_tool`, and `research_tools.wikipedia_search_tool`.

- **Correctly Setting the Call to the LLM**:
  - Pass the `model`, `messages`, and `tools` parameters accurately.
  - Set `tool_choice` to `"auto"` for automatic tool selection.
  - Limit interactions with `max_turns=6`.

### Notes

- The function provides pre-coded blocks where you need to replace placeholder values.
- The approach allows the LLM to use tools dynamically based on the task.

Focus on accurately setting the messages, tools, and LLM call parameters to complete the task.


```python
# GRADED FUNCTION: research_agent

def research_agent(task: str, model: str = "openai:gpt-4o", return_messages: bool = False):
    """
    Executes a research task using tools via aisuite (no manual loop).
    Returns either the assistant text, or (text, messages) if return_messages=True.
    """
    print("==================================")  
    print("🔍 Research Agent")                 
    print("==================================")

    current_time = datetime.now().strftime('%Y-%m-%d')
    
    ### START CODE HERE ###

    # Create a customizable prompt by defining the role (e.g., "research assistant"),
    # listing tools (arxiv_tool, tavily_tool, wikipedia_tool) for various searches,
    # specifying the task with a placeholder, and including a current_time placeholder.
    prompt = f"""
    You are an expert research assistant designed to execute complex research tasks using external tools.

You have access to the following tools:
- arxiv_tool → for academic papers and technical research
- tavily_tool → for current events, news, and web sources
- wikipedia_tool → for general background knowledge and definitions

Your goal is to produce accurate, well-sourced, and structured research results for the task provided.

Current time: {current_time}

TASK:
{task}

INSTRUCTIONS:

1. Tool Usage Strategy
- Use arxiv_tool for scientific, technical, or academic claims.
- Use tavily_tool for recent information, news, statistics, and industry context.
- Use wikipedia_tool for definitions, background, and historical context.
- Prefer primary sources over summaries when possible.
- Use multiple tools when cross-verification improves reliability.

2. Research Method
- Break the task into sub-questions before searching.
- Gather evidence from tools before forming conclusions.
- Cross-check important facts across at least two sources when feasible.
- Do NOT rely on prior knowledge if a tool can verify it.

3. Output Requirements
Your final answer must:
- Be structured and easy to read.
- Cite the source of each important claim (tool + reference).
- Separate facts from interpretation.
- Highlight any uncertainty or conflicting information.
- Avoid speculation.

4. Quality Bar
- Prioritize correctness over speed.
- Prefer depth over breadth.
- Avoid filler text and generic explanations.
- If information cannot be verified, explicitly say so.

5. Failure Handling
- If tools return insufficient information, explain what is missing.
- If sources disagree, explain the discrepancy.
- Never fabricate information.

6. Style
- Write clearly, professionally, and concisely.
- Use headings, bullets, and sections where helpful.
- Assume the reader values rigor and evidence.

Begin the research process now.
    """
    
    # Create the messages dict to pass to the LLM. Remember this is a user prompt!
    messages = [{"role": "system", "content": prompt}]

    # Save all of your available tools in the tools list. These can be found in the research_tools module.
    # You can identify each tool in your list like this: 
    # research_tools.<name_of_tool>, where <name_of_tool> is replaced with the function name of the tool.
    tools = [research_tools.arxiv_search_tool, research_tools.tavily_search_tool, research_tools.wikipedia_search_tool]
    
    # Call the model with tools enabled
    response = CLIENT.chat.completions.create(  
        # Set the model
        model=model,
        # Pass in the messages. You already defined this!
        messages=messages,
        # Pass in the tools list. You already defined this!
        tools=tools,
        # Set the LLM to automatically choose the tools
        tool_choice="auto",
        # Set the max turns to 6
        max_turns=6
    )  
    
    ### END CODE HERE ###

    content = response.choices[0].message.content
    print("✅ Output:\n", content)

    
    return (content, messages) if return_messages else content  
```


```python
# Test your code!
unittests.test_research_agent(research_agent)
```

    ==================================
    🔍 Research Agent
    ==================================
    ✅ Output:
     ## Summary of Key References on Climate Change and Its Impact on Biodiversity
    
    ### 1. Academic Research on Precipitation Extremes
    - **Source**: ArXiv, "Precipitation extremes under climate change" by Paul A. O'Gorman (2015)
    - **Summary**: The study explores how precipitation extremes are expected to intensify with climate change, though the sensitivity to warming varies with geographic location—higher in the tropics compared to extratropics. The research emphasizes the robust thermodynamic contributions to precipitation changes but notes ongoing development in understanding microphysical and dynamical contributions. The paper identifies knowledge gaps, especially regarding mesoscale convective organization and tropical precipitation sensitivity [ArXiv](https://arxiv.org/abs/1503.07557v1), [PDF](https://arxiv.org/pdf/1503.07557v1).
    
    ### 2. Current Climate Change Impacts on Biodiversity (Royal Society)
    - **Source**: Tavily, Royal Society
    - **Summary**: The Royal Society highlights that climate change exacerbates biodiversity threats through increased fire frequencies, intense storms, and prolonged drought periods. These events disrupt ecological balance, leading to potential losses in biodiversity [Royal Society](https://royalsociety.org/news-resources/projects/biodiversity/climate-change-and-biodiversity/).
    
    ### 3. Wikipedia Overview on Plant Biodiversity
    - **Source**: Wikipedia
    - **Summary**: The Wikipedia entry discusses the ongoing decline in plant biodiversity due to climate change. The primary concern is how changing environmental conditions impact plant distributions and ecosystem functions. Predictive models, such as bioclimatic models, are used to foresee these impacts, indicating changes in biodiversity concomitant with climatic shifts [Wikipedia](https://en.wikipedia.org/wiki/Effects_of_climate_change_on_plant_biodiversity).
    
    These sources provide a comprehensive look at the scientific, current, and general understanding of how climate change affects biodiversity across different ecosystems and contexts.
    ==================================
    🔍 Research Agent
    ==================================
    ✅ Output:
     ### Summary of Two Seminal Papers
    
    1. **Paper on Machine Learning for Biology**  
       **Title:** "DOME: Recommendations for supervised machine learning validation in biology"  
       **Authors:** Ian Walsh et al.  
       **Published:** 2020-06-25  
       This paper addresses the growing reliance on machine learning in biological studies and the necessity for stringent validation standards to ensure reliable outcomes. The authors introduce the DOME framework - Data, Optimization, Model, and Evaluation - to standardize the validation process and enhance transparency. By providing a structured description of methods, DOME aids both reviewers and readers in evaluating the performance and limitations of machine learning models in biological research. This initiative aims to foster a better understanding and assessment of machine learning applications, thus promoting improved implementation standards in the field of biology [source: arxiv_tool, [link](https://arxiv.org/pdf/2006.16189v4)].
    
    2. **Paper on Computer Vision for Precision Pollination**  
       **Title:** "Spatial Monitoring and Insect Behavioural Analysis Using Computer Vision for Precision Pollination"  
       **Authors:** Malika Nisal Ratnayake et al.  
       **Published:** 2022-05-10  
       This work presents a novel system for enhancing pollination monitoring through computer vision. By automating insect tracking and behavioral analysis, the system offers comprehensive data across large agricultural areas. It is capable of video recording, multi-species insect counting, motion tracking, and behavioral analysis, significantly improving data accuracy for agricultural applications. Implemented on a berry farm, the system demonstrated high efficacy, achieving an F-score above 0.8 for various insect species. This advancement promises substantial improvements in data-driven crop pollination strategies, crucial for increasing food production and supporting food security [source: arxiv_tool, [link](https://arxiv.org/pdf/2205.04675v2)].
    [92m All tests passed!


## Exercise 3: writer_agent

### Objective
Set up a call to a language model (LLM) for executing writing tasks like drafting, expanding, or summarizing text.

### Instructions

1. **Focus Areas**:
   - **System Prompt**:
     - Define `system_prompt` to assign the LLM the role of a writing agent focused on generating academic or technical content.
   - **System and User Messages**:
     - Create `system_msg` using `{"role": "system", "content": system_prompt}`.
     - Create `user_msg` using `{"role": "user", "content": task}`.
   - **Messages List**:
     - Combine `system_msg` and `user_msg` into a `messages` list.

### Notes

- The function is designed to produce well-structured text by setting the correct prompts.
- Temperature is set to 1.0 to allow for creative variance in the writing outputs.

Ensure the system prompt and messages are defined properly to achieve a structured output from the LLM.


```python
# GRADED FUNCTION: writer_agent
def writer_agent(task: str, model: str = "openai:gpt-4o") -> str: # @REPLACE def writer_agent(task: str, model: str = None) -> str:
    """
    Executes writing tasks, such as drafting, expanding, or summarizing text.
    """
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")

    ### START CODE HERE ###
    
    # Create the system prompt.
    # This should assign the LLM the role of a writing agent specialized in generating well-structured academic or technical content
    system_prompt = """
You are a professional writing assistant specialized in producing clear, well-structured,
and rigorous academic and technical content.

Your role is to:
- Draft, expand, summarize, or refine text with high clarity and precision
- Organize content logically using sections, headings, and coherent flow
- Avoid filler language, repetition, and vague statements
- Preserve technical correctness while improving readability
- Adapt tone to academic, technical, or professional contexts as appropriate

Writing principles:
- Be concise but complete
- Prefer clarity over verbosity
- Use structured formatting when helpful
- Do not invent facts; work only with the provided information
- Improve the quality of the writing without changing the meaning

Produce polished, publication-quality text.
"""

    # Define the system msg by using the system_prompt and assigning the role of system
    system_msg = {"role": "system", "content": system_prompt}

    # Define the user msg. In this case the user prompt should be the task passed to the function
    user_msg = {"role": "user", "content": task}

    # Add both system and user messages to the messages list
    messages = [system_msg, user_msg]
    
    ### END CODE HERE ###

    response = CLIENT.chat.completions.create(
        model=model, 
        messages=messages,
        temperature=1.0
    )

    return response.choices[0].message.content
```


```python
# Test your code!
unittests.test_writer_agent(writer_agent)
```

    ==================================
    ✍️ Writer Agent
    ==================================
    [92m All tests passed!


## Exercise 4: editor_agent

### Objective
Configure a call to a language model (LLM) to perform editorial tasks such as reflecting, critiquing, or revising drafts.

### Instructions

1. **Focus Areas**:
   - **System Prompt**:
     - Define `system_prompt` to assign the LLM the role of an editor agent whose task is to reflect on, critique, or improve drafts.
   - **System and User Messages**:
     - Create `system_msg` using `{"role": "system", "content": system_prompt}`.
     - Create `user_msg` using `{"role": "user", "content": task}`.
   - **Messages List**:
     - Combine `system_msg` and `user_msg` into a `messages` list.

### Notes

- The editor agent is tailored for enhancing the quality of text by setting an appropriate role and task in the prompts.
- Temperature is set to 0.7, balancing creativity and coherence in editorial outputs.

Ensure the system prompt and messages are accurately set up to perform effective editorial tasks with the LLM.


```python
# GRADED FUNCTION: editor_agent
def editor_agent(task: str, model: str = "openai:gpt-4o") -> str:
    """
    Executes editorial tasks such as reflection, critique, or revision.
    """
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")
    
    ### START CODE HERE ###

    # Create the system prompt.
    # This should assign the LLM the role of an editor agent specialized in reflecting on, critiquing, or improving existing drafts.
    system_prompt = """
You are a professional editor specialized in reviewing, critiquing, and improving
academic and technical writing.

Your role is NOT to rewrite from scratch, but to:
- Identify weaknesses in clarity, structure, logic, and flow
- Point out vague, redundant, or poorly phrased sections
- Suggest concrete improvements
- Preserve the author’s intent and meaning
- Improve readability, precision, and coherence

Editorial principles:
- Be constructive and specific
- Explain why something should change
- Suggest improved wording when helpful
- Focus on structure, argument quality, and clarity
- Do not introduce new ideas or facts not present in the original text

Provide actionable feedback and, when appropriate, revised versions of problematic sections.
"""
    
    # Define the system msg by using the system_prompt and assigning the role of system
    system_msg = {"role": "system", "content": system_prompt}
    
    # Define the user msg. In this case the user prompt should be the task passed to the function
    user_msg = {"role": "user", "content": task}
    
    # Add both system and user messages to the messages list
    messages = [system_msg, user_msg]
    
    ### END CODE HERE ###
    
    response = CLIENT.chat.completions.create(
        model=model, 
        messages=messages,
        temperature=0.7 
    )
    
    return response.choices[0].message.content
```


```python
# Test your code!
unittests.test_editor_agent(editor_agent)
```

    ==================================
    🧠 Editor Agent
    ==================================
    [92m All tests passed!


### 🎯 The Executor Agent

The `executor_agent` manages the workflow by executing each step of a given plan. It:

1. Decides **which agent** (`research_agent`, `writer_agent`, or `editor_agent`) should handle the step.
2. Builds context from the outputs of previous steps.
3. Sends the enriched task to the selected agent.
4. Collects and stores the results in a shared history.

👉 **Do not implement or modify this function.** It is already provided as the orchestration component of the multi-agent pipeline.

Notice that `planner_agent` might return a long list of steps. Because of this, the maximum number of steps is set to a maximum of 4 to keep running time reasonable.


```python
agent_registry = {
    "research_agent": research_agent,
    "editor_agent": editor_agent,
    "writer_agent": writer_agent,
}

def clean_json_block(raw: str) -> str:
    """
    Clean the contents of a JSON block that may come wrapped with Markdown backticks.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()
```


```python
def executor_agent(topic, model: str = "openai:gpt-4o", limit_steps: bool = True):

    plan_steps = planner_agent(topic)
    max_steps = 4

    if limit_steps:
        plan_steps = plan_steps[:min(len(plan_steps), max_steps)]
    
    history = []

    print("==================================")
    print("🎯 Editor Agent")
    print("==================================")

    for i, step in enumerate(plan_steps):

        agent_decision_prompt = f"""
        You are an execution manager for a multi-agent research team.

        Given the following instruction, identify which agent should perform it and extract the clean task.

        Return only a valid JSON object with two keys:
        - "agent": one of ["research_agent", "editor_agent", "writer_agent"]
        - "task": a string with the instruction that the agent should follow

        Only respond with a valid JSON object. Do not include explanations or markdown formatting.

        Instruction: "{step}"
        """
        response = CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": agent_decision_prompt}],
            temperature=0,
        )

        raw_content = response.choices[0].message.content
        cleaned_json = clean_json_block(raw_content)
        agent_info = json.loads(cleaned_json)

        agent_name = agent_info["agent"]
        task = agent_info["task"]

        context = "\n".join([
            f"Step {j+1} executed by {a}:\n{r}" 
            for j, (s, a, r) in enumerate(history)
        ])
        enriched_task = f"""
        You are {agent_name}.

        Here is the context of what has been done so far:
        {context}

        Your next task is:
        {task}
        """

        print(f"\n🛠️ Executing with agent: `{agent_name}` on task: {task}")

        if agent_name in agent_registry:
            output = agent_registry[agent_name](enriched_task)
            history.append((step, agent_name, output))
        else:
            output = f"⚠️ Unknown agent: {agent_name}"
            history.append((step, agent_name, output))

        print(f"✅ Output:\n{output}")

    return history
```


```python
# If you want to see the full workflow without limiting the number of steps. Set limit_steps to False
# Keep in mind this could take more than 10 minutes to complete
executor_history = executor_agent("The ensemble Kalman filter for time series forecasting", limit_steps=True)

md = executor_history[-1][-1].strip("`")  
display(Markdown(md))
```

    ==================================
    🎯 Editor Agent
    ==================================
    
    🛠️ Executing with agent: `research_agent` on task: Search arXiv for recent papers on ensemble Kalman filter applications in time series forecasting
    ==================================
    🔍 Research Agent
    ==================================
    ✅ Output:
     Here's a summary of recent papers from arXiv that discuss applications of the ensemble Kalman filter (EnKF) in time series forecasting:
    
    1. **Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference**  
       - **Authors**: Zhidi Lin, Yiyong Sun, Feng Yin, Alexandre Hoang Thiéry
       - **Published**: December 10, 2023
       - **Summary**: This paper integrates the ensemble Kalman filter with Gaussian process state-space models to address limitations in current variational inference methods under non-mean-field assumptions. The integration allows for improved posterior distribution approximation and supports online learning applications. The proposed EnKF-aided algorithm demonstrates superior learning and inference performance over existing methods across various datasets. [Read more](http://arxiv.org/abs/2312.05910v5) | [PDF](https://arxiv.org/pdf/2312.05910v5)
    
    2. **LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting**
       - **Authors**: Md Kowsher, Md. Shohanur Islam Sobuj, Nusrat Jahan Prottasha, E. Alejandro Alanis, Ozlem Ozmen Garibay, Niloofar Yousefi
       - **Published**: October 15, 2024
       - **Summary**: While not directly focused on EnKF, this paper presents the LLM-Mixer framework for time series forecasting. It combines multiscale time-series decomposition with pre-trained language models, offering insights into advanced methods compatible with EnKF's goals of improved forecasting accuracy. [Read more](http://arxiv.org/abs/2410.11674v2) | [PDF](https://arxiv.org/pdf/2410.11674v2)
    
    The papers collected provide a glimpse into current applications and innovations involving the ensemble Kalman filter, particularly its integration with other advanced modeling techniques to enhance forecasting accuracy and applicability in diverse scenarios.
    ✅ Output:
    Here's a summary of recent papers from arXiv that discuss applications of the ensemble Kalman filter (EnKF) in time series forecasting:
    
    1. **Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference**  
       - **Authors**: Zhidi Lin, Yiyong Sun, Feng Yin, Alexandre Hoang Thiéry
       - **Published**: December 10, 2023
       - **Summary**: This paper integrates the ensemble Kalman filter with Gaussian process state-space models to address limitations in current variational inference methods under non-mean-field assumptions. The integration allows for improved posterior distribution approximation and supports online learning applications. The proposed EnKF-aided algorithm demonstrates superior learning and inference performance over existing methods across various datasets. [Read more](http://arxiv.org/abs/2312.05910v5) | [PDF](https://arxiv.org/pdf/2312.05910v5)
    
    2. **LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting**
       - **Authors**: Md Kowsher, Md. Shohanur Islam Sobuj, Nusrat Jahan Prottasha, E. Alejandro Alanis, Ozlem Ozmen Garibay, Niloofar Yousefi
       - **Published**: October 15, 2024
       - **Summary**: While not directly focused on EnKF, this paper presents the LLM-Mixer framework for time series forecasting. It combines multiscale time-series decomposition with pre-trained language models, offering insights into advanced methods compatible with EnKF's goals of improved forecasting accuracy. [Read more](http://arxiv.org/abs/2410.11674v2) | [PDF](https://arxiv.org/pdf/2410.11674v2)
    
    The papers collected provide a glimpse into current applications and innovations involving the ensemble Kalman filter, particularly its integration with other advanced modeling techniques to enhance forecasting accuracy and applicability in diverse scenarios.
    
    🛠️ Executing with agent: `research_agent` on task: Search Wikipedia for background on Kalman filters and ensemble methods
    ==================================
    🔍 Research Agent
    ==================================
    ✅ Output:
     ### Background on Kalman Filters
    
    **Kalman Filter**:
    - **Definition**: In statistics and control theory, a Kalman filter is an algorithm that processes a series of measurements over time. These measurements include statistical noise and inaccuracies, and the Kalman filter aims to produce estimates of unknown variables that are more accurate than those obtained from a single measurement. The filter does this by estimating a joint probability distribution over the variables for each time-step.
    - **Applications**: It is widely used in technologies such as guidance, navigation, and control of vehicles, including aircraft, spacecraft, and dynamically positioned ships.
    - **Origin**: The filter is named after Rudolf E. Kálmán and is a model of a mean squared error minimizer, also connected to maximum likelihood statistics.
    - [Learn More](https://en.wikipedia.org/wiki/Kalman_filter)
    
    ### Background on Ensemble Methods
    
    **Ensemble Methods**:
    - **Definition**: In machine learning and statistics, ensemble methods employ multiple learning algorithms to enhance predictive performance compared to the performance of any individual algorithm within the ensemble. Instead of an infinite set (common in statistical mechanics), machine learning ensembles are finite but allow for flexible structures among models.
    - **Purpose**: The goal is to search through a hypothesis space to form a composite hypothesis better suited to prediction than any single model, thereby improving predictive analytics.
    - [Learn More](https://en.wikipedia.org/wiki/Ensemble_learning)
    
    The synthesis of these methodologies into the ensemble Kalman filter (EnKF) represents an integration of filtering techniques with ensemble approaches to improve the robustness and accuracy of predictions in time-series forecasting and other applications.
    ✅ Output:
    ### Background on Kalman Filters
    
    **Kalman Filter**:
    - **Definition**: In statistics and control theory, a Kalman filter is an algorithm that processes a series of measurements over time. These measurements include statistical noise and inaccuracies, and the Kalman filter aims to produce estimates of unknown variables that are more accurate than those obtained from a single measurement. The filter does this by estimating a joint probability distribution over the variables for each time-step.
    - **Applications**: It is widely used in technologies such as guidance, navigation, and control of vehicles, including aircraft, spacecraft, and dynamically positioned ships.
    - **Origin**: The filter is named after Rudolf E. Kálmán and is a model of a mean squared error minimizer, also connected to maximum likelihood statistics.
    - [Learn More](https://en.wikipedia.org/wiki/Kalman_filter)
    
    ### Background on Ensemble Methods
    
    **Ensemble Methods**:
    - **Definition**: In machine learning and statistics, ensemble methods employ multiple learning algorithms to enhance predictive performance compared to the performance of any individual algorithm within the ensemble. Instead of an infinite set (common in statistical mechanics), machine learning ensembles are finite but allow for flexible structures among models.
    - **Purpose**: The goal is to search through a hypothesis space to form a composite hypothesis better suited to prediction than any single model, thereby improving predictive analytics.
    - [Learn More](https://en.wikipedia.org/wiki/Ensemble_learning)
    
    The synthesis of these methodologies into the ensemble Kalman filter (EnKF) represents an integration of filtering techniques with ensemble approaches to improve the robustness and accuracy of predictions in time-series forecasting and other applications.
    
    🛠️ Executing with agent: `research_agent` on task: Extract key mathematical formulations and algorithmic pseudocode of the ensemble Kalman filter
    ==================================
    🔍 Research Agent
    ==================================
    ✅ Output:
     ### Key Mathematical Formulations and Algorithmic Overview of the Ensemble Kalman Filter (EnKF)
    
    #### 1. **Overview and Purpose:**
    
    The ensemble Kalman filter (EnKF) is designed as an effective recursive filter to handle complex, large-scale time series prediction tasks, particularly relevant in geophysical models where state variables can be numerous. EnKF applies Monte Carlo methods as part of a Bayesian framework to estimate the posterior state distribution, which improves the prediction process by incorporating measurement updates.
    
    #### 2. **Mathematical Formulation:**
    
    - **State Vector and Forecasting:**
      The fundamental approach involves maintaining an ensemble of state vectors, \(\mathbf{x}_i\), where \(i\) indexes the ensemble members, and applying the forecast model \(\mathbf{x}_i^{f} = M(\mathbf{x}_i^{a}) + \xi_i\), where \(\mathbf{x}_i^{f}\) represents the forecasted state, and \(\xi_i\) accounts for model uncertainties [Wikipedia](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
    
    - **Analysis Update:**
      Analysis incorporates the observations:
      \[
      \mathbf{x}_i^{a} = \mathbf{x}_i^{f} + \mathbf{K}(\mathbf{y} - \mathbf{H}\mathbf{x}_i^{f} + \epsilon_i)
      \]
      where:
      - \(\mathbf{x}_i^{a}\) is the updated state after incorporating observations,
      - \(\mathbf{K}\) is the Kalman gain,
      - \(\mathbf{y}\) denotes the observation vector,
      - \(\mathbf{H}\) is the observation model, and
      - \(\epsilon_i\) represents measurement noise.
    
    - **Sample Covariance:**
      The covariance required for gain computation involves sample covariances of the ensemble. This method assumes Gaussian distributions, leading to efficiency gains over particle filters in similar use-cases.
    
    #### 3. **Algorithmic Pseudocode:**
    
    1. **Initialization:**
       - Sample an initial ensemble \(\{\mathbf{x}_i^0\}\) from a Gaussian distribution.
    
    2. **Forecast Step:**
       - For each ensemble member, \(\mathbf{x}_i^{f} = M(\mathbf{x}_i^{a}) + \xi_i\).
    
    3. **Compute Sample Statistics:**
       - Compute ensemble mean \(\overline{\mathbf{x}^f}\) and sample covariance \(\mathbf{P}^{f}\).
    
    4. **Analysis Step:**
       - Compute Kalman gain:
       \[
       \mathbf{K} = \mathbf{P}^f \mathbf{H}' (\mathbf{H} \mathbf{P}^f \mathbf{H}' + \mathbf{R})^{-1}
       \]
       - Update each ensemble member with observations:
       \[
       \mathbf{x}_i^{a} = \mathbf{x}_i^{f} + \mathbf{K}(\mathbf{y} - \mathbf{H}\mathbf{x}_i^{f} + \epsilon_i)
       \]
    
    5. **Repeat:**
       - The process iterates each time a new set of observations is acquired, cycling between forecast and analysis steps.
    
    ### Sources:
    - **Wikipedia on Ensemble Kalman Filter**: Provides an introduction to the implementation and underlying fundamental concepts of the EnKF [Wikipedia Summary](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
    - **ArXiv Papers**: Offer applications and detailed studies integrating EnKF in various scenarios including Gaussian process models.
    
    This overview of EnKF not only highlights the mathematical foundation but also provides insight into how EnKF is operationalized algorithmically, facilitating better prediction accuracy, especially in contexts with large state spaces. Any additional nuances and specialized adaptations of EnKF would typically be found in advanced research documents or specific case studies.
    ✅ Output:
    ### Key Mathematical Formulations and Algorithmic Overview of the Ensemble Kalman Filter (EnKF)
    
    #### 1. **Overview and Purpose:**
    
    The ensemble Kalman filter (EnKF) is designed as an effective recursive filter to handle complex, large-scale time series prediction tasks, particularly relevant in geophysical models where state variables can be numerous. EnKF applies Monte Carlo methods as part of a Bayesian framework to estimate the posterior state distribution, which improves the prediction process by incorporating measurement updates.
    
    #### 2. **Mathematical Formulation:**
    
    - **State Vector and Forecasting:**
      The fundamental approach involves maintaining an ensemble of state vectors, \(\mathbf{x}_i\), where \(i\) indexes the ensemble members, and applying the forecast model \(\mathbf{x}_i^{f} = M(\mathbf{x}_i^{a}) + \xi_i\), where \(\mathbf{x}_i^{f}\) represents the forecasted state, and \(\xi_i\) accounts for model uncertainties [Wikipedia](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
    
    - **Analysis Update:**
      Analysis incorporates the observations:
      \[
      \mathbf{x}_i^{a} = \mathbf{x}_i^{f} + \mathbf{K}(\mathbf{y} - \mathbf{H}\mathbf{x}_i^{f} + \epsilon_i)
      \]
      where:
      - \(\mathbf{x}_i^{a}\) is the updated state after incorporating observations,
      - \(\mathbf{K}\) is the Kalman gain,
      - \(\mathbf{y}\) denotes the observation vector,
      - \(\mathbf{H}\) is the observation model, and
      - \(\epsilon_i\) represents measurement noise.
    
    - **Sample Covariance:**
      The covariance required for gain computation involves sample covariances of the ensemble. This method assumes Gaussian distributions, leading to efficiency gains over particle filters in similar use-cases.
    
    #### 3. **Algorithmic Pseudocode:**
    
    1. **Initialization:**
       - Sample an initial ensemble \(\{\mathbf{x}_i^0\}\) from a Gaussian distribution.
    
    2. **Forecast Step:**
       - For each ensemble member, \(\mathbf{x}_i^{f} = M(\mathbf{x}_i^{a}) + \xi_i\).
    
    3. **Compute Sample Statistics:**
       - Compute ensemble mean \(\overline{\mathbf{x}^f}\) and sample covariance \(\mathbf{P}^{f}\).
    
    4. **Analysis Step:**
       - Compute Kalman gain:
       \[
       \mathbf{K} = \mathbf{P}^f \mathbf{H}' (\mathbf{H} \mathbf{P}^f \mathbf{H}' + \mathbf{R})^{-1}
       \]
       - Update each ensemble member with observations:
       \[
       \mathbf{x}_i^{a} = \mathbf{x}_i^{f} + \mathbf{K}(\mathbf{y} - \mathbf{H}\mathbf{x}_i^{f} + \epsilon_i)
       \]
    
    5. **Repeat:**
       - The process iterates each time a new set of observations is acquired, cycling between forecast and analysis steps.
    
    ### Sources:
    - **Wikipedia on Ensemble Kalman Filter**: Provides an introduction to the implementation and underlying fundamental concepts of the EnKF [Wikipedia Summary](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
    - **ArXiv Papers**: Offer applications and detailed studies integrating EnKF in various scenarios including Gaussian process models.
    
    This overview of EnKF not only highlights the mathematical foundation but also provides insight into how EnKF is operationalized algorithmically, facilitating better prediction accuracy, especially in contexts with large state spaces. Any additional nuances and specialized adaptations of EnKF would typically be found in advanced research documents or specific case studies.


    
    🛠️ Executing with agent: `research_agent` on task: Identify and summarize case studies applying EnKF to time series forecasting from selected sources
    ==================================
    🔍 Research Agent
    ==================================
    ✅ Output:
     ### Summary of Case Studies Applying EnKF to Time Series Forecasting
    
    #### Case Study 1: Indoor Environment Forecasting
    - **Source**: ScienceDirect
    - **Application**: A study utilized the ensemble Kalman filter (EnKF) for predicting gas dispersion within an indoor environment. This involved simulating contaminant dispersion, using sensor data to assimilate with a stochastic model for improved predictions.
    - **Methodology**: The EnKF algorithm was combined with computational fluid dynamics to achieve spatiotemporal predictions and gas emission source estimation, particularly effective for scenarios involving natural gas release in urban tunnels.
    - **Key Finding**: The model successfully estimated natural gas dispersion in real-time, showcasing EnKF as effective for environmental predictions in complex interiors. [Read More](https://www.sciencedirect.com/science/article/abs/pii/S0360132313000887)
    
    #### Case Study 2: Infectious Disease Modeling
    - **Source**: Nature Scientific Reports
    - **Application**: This study combined real-time EnKF with K-nearest neighbors (KNN) for predicting the spread of COVID-19 using an SEAIQR epidemic model. The approach aimed to improve case predictions by refining data alignment between model predictions and observed epidemic data.
    - **Methodology**: EnKF was used to handle real-time data assimilation, enhancing the predictive accuracy of time-dependent models used for tracking and forecasting outbreak dynamics.
    - **Key Finding**: The hybrid model showed substantial improvement in predictive performance over standalone models, highlighting EnKF's utility in disease spread modeling. [Read More](https://www.nature.com/articles/s41598-025-85593-z)
    
    #### Case Study 3: Weather and Climate Prediction
    - **Source**: Journals of the American Meteorological Society
    - **Application**: EnKF was deployed for data assimilation over the Pacific Northwest, focusing on weather forecasting improvements on a regional scale.
    - **Methodology**: EnKF provided a mechanism to incorporate observational data into limited-area models, enhancing the accuracy of weather forecasts.
    - **Key Finding**: The use of EnKF in weather prediction allowed for more accurate regional forecasts by improving data quality and assimilation techniques. [Read More](https://journals.ametsoc.org/view/journals/mwre/135/4/mwr3358.1.pdf)
    
    ### Reliable and Effective Use of EnKF in Diverse Fields
    
    These case studies highlight the versatility and robustness of EnKF across various fields requiring precise time series forecasting. From urban environmental monitoring to epidemic modeling and regional weather forecasting, EnKF provides a robust framework for integrating observations into complex predictive models, enhancing forecasting accuracy and reliability significantly.
    ✅ Output:
    ### Summary of Case Studies Applying EnKF to Time Series Forecasting
    
    #### Case Study 1: Indoor Environment Forecasting
    - **Source**: ScienceDirect
    - **Application**: A study utilized the ensemble Kalman filter (EnKF) for predicting gas dispersion within an indoor environment. This involved simulating contaminant dispersion, using sensor data to assimilate with a stochastic model for improved predictions.
    - **Methodology**: The EnKF algorithm was combined with computational fluid dynamics to achieve spatiotemporal predictions and gas emission source estimation, particularly effective for scenarios involving natural gas release in urban tunnels.
    - **Key Finding**: The model successfully estimated natural gas dispersion in real-time, showcasing EnKF as effective for environmental predictions in complex interiors. [Read More](https://www.sciencedirect.com/science/article/abs/pii/S0360132313000887)
    
    #### Case Study 2: Infectious Disease Modeling
    - **Source**: Nature Scientific Reports
    - **Application**: This study combined real-time EnKF with K-nearest neighbors (KNN) for predicting the spread of COVID-19 using an SEAIQR epidemic model. The approach aimed to improve case predictions by refining data alignment between model predictions and observed epidemic data.
    - **Methodology**: EnKF was used to handle real-time data assimilation, enhancing the predictive accuracy of time-dependent models used for tracking and forecasting outbreak dynamics.
    - **Key Finding**: The hybrid model showed substantial improvement in predictive performance over standalone models, highlighting EnKF's utility in disease spread modeling. [Read More](https://www.nature.com/articles/s41598-025-85593-z)
    
    #### Case Study 3: Weather and Climate Prediction
    - **Source**: Journals of the American Meteorological Society
    - **Application**: EnKF was deployed for data assimilation over the Pacific Northwest, focusing on weather forecasting improvements on a regional scale.
    - **Methodology**: EnKF provided a mechanism to incorporate observational data into limited-area models, enhancing the accuracy of weather forecasts.
    - **Key Finding**: The use of EnKF in weather prediction allowed for more accurate regional forecasts by improving data quality and assimilation techniques. [Read More](https://journals.ametsoc.org/view/journals/mwre/135/4/mwr3358.1.pdf)
    
    ### Reliable and Effective Use of EnKF in Diverse Fields
    
    These case studies highlight the versatility and robustness of EnKF across various fields requiring precise time series forecasting. From urban environmental monitoring to epidemic modeling and regional weather forecasting, EnKF provides a robust framework for integrating observations into complex predictive models, enhancing forecasting accuracy and reliability significantly.



### Summary of Case Studies Applying EnKF to Time Series Forecasting

#### Case Study 1: Indoor Environment Forecasting
- **Source**: ScienceDirect
- **Application**: A study utilized the ensemble Kalman filter (EnKF) for predicting gas dispersion within an indoor environment. This involved simulating contaminant dispersion, using sensor data to assimilate with a stochastic model for improved predictions.
- **Methodology**: The EnKF algorithm was combined with computational fluid dynamics to achieve spatiotemporal predictions and gas emission source estimation, particularly effective for scenarios involving natural gas release in urban tunnels.
- **Key Finding**: The model successfully estimated natural gas dispersion in real-time, showcasing EnKF as effective for environmental predictions in complex interiors. [Read More](https://www.sciencedirect.com/science/article/abs/pii/S0360132313000887)

#### Case Study 2: Infectious Disease Modeling
- **Source**: Nature Scientific Reports
- **Application**: This study combined real-time EnKF with K-nearest neighbors (KNN) for predicting the spread of COVID-19 using an SEAIQR epidemic model. The approach aimed to improve case predictions by refining data alignment between model predictions and observed epidemic data.
- **Methodology**: EnKF was used to handle real-time data assimilation, enhancing the predictive accuracy of time-dependent models used for tracking and forecasting outbreak dynamics.
- **Key Finding**: The hybrid model showed substantial improvement in predictive performance over standalone models, highlighting EnKF's utility in disease spread modeling. [Read More](https://www.nature.com/articles/s41598-025-85593-z)

#### Case Study 3: Weather and Climate Prediction
- **Source**: Journals of the American Meteorological Society
- **Application**: EnKF was deployed for data assimilation over the Pacific Northwest, focusing on weather forecasting improvements on a regional scale.
- **Methodology**: EnKF provided a mechanism to incorporate observational data into limited-area models, enhancing the accuracy of weather forecasts.
- **Key Finding**: The use of EnKF in weather prediction allowed for more accurate regional forecasts by improving data quality and assimilation techniques. [Read More](https://journals.ametsoc.org/view/journals/mwre/135/4/mwr3358.1.pdf)

### Reliable and Effective Use of EnKF in Diverse Fields

These case studies highlight the versatility and robustness of EnKF across various fields requiring precise time series forecasting. From urban environmental monitoring to epidemic modeling and regional weather forecasting, EnKF provides a robust framework for integrating observations into complex predictive models, enhancing forecasting accuracy and reliability significantly.


## Check grading feedback

If you have collapsed the right panel to have more screen space for your code, as shown below:

<img src="./images/collapsed.png" alt="Collapsed Image" width="800" height="400"/>

You can click on the left-facing arrow button (highlighted in red) to view feedback for your submission after submitting it for grading. Once expanded, it should display like this:

<img src="./images/expanded.png" alt="Expanded Image" width="800" height="400"/>

========
Collaboration can take various forms: Sequential Handoffs: One agent completes a task and passes its output to another agent for the next step in a pipeline (similar to the Planning pattern, but explicitly involving different agents). Parallel Processing: Multiple agents work on different parts of a problem simultaneously, and their results are later combined. Debate and Consensus: Multi-Agent Collaboration where Agents with varied perspectives and information sources engage in discussions to evaluate options, ultimately reaching a consensus or a more informed decision. Hierarchical Structures: A manager agent might delegate tasks to worker agents dynamically based on their tool access or plugin capabilities and synthesize their results. Each agent can also handle relevant groups of tools, rather than a single agent handling all the tools. Expert Teams: Agents with specialized knowledge in different domains (e.g., a researcher, a writer, an editor) collaborate to produce a complex output. Critic-Reviewer: Agents create initial outputs such as plans, drafts, or answers. A second group of agents then critically assesses this output for adherence to policies, security, compliance, correctness, quality, and alignment with organizational objectives. The original creator or a final agent revises the output based on this feedback. This pattern is particularly effective for code generation, research writing, logic checking, and ensuring ethical alignment. The advantages of this approach include increased robustness, improved quality, and a reduced likelihood of hallucinations or errors.

Practical Applications and Use Cases Multi-Agent Collaboration is a powerful pattern applicable across numerous domains: Complex Research and Analysis: A team of agents could collaborate on a research project. One agent might specialize in searching academic databases, another in summarizing findings, a third in identifying trends, and a fourth in synthesizing the information into a report. This mirrors how a human research team might operate. Software Development: Imagine agents collaborating on building software. One agent could be a requirements analyst, another a code generator, a third a tester, and a fourth a documentation writer. They could pass outputs between each other to build and verify components. Creative Content Generation: Creating a marketing campaign could involve a market research agent, a copywriter agent, a graphic design agent (using image generation tools), and a social media scheduling agent, all working together. Financial Analysis: A multi-agent system could analyze financial markets. Agents might specialize in fetching stock data, analyzing news sentiment, performing technical analysis, and generating investment recommendations. Customer Support Escalation: A front-line support agent could handle initial queries, escalating complex issues to a specialist agent (e.g., a technical expert or a billing specialist) when needed, demonstrating a sequential handoff based on problem complexity. Supply Chain Optimization: Agents could represent different nodes in a supply chain (suppliers, manufacturers, distributors) and collaborate to optimize inventory levels, logistics, and scheduling in response to changing demand or disruptions. Network Analysis & Remediation: Autonomous operations benefit greatly from an agentic architecture, particularly in failure pinpointing. Multiple agents can collaborate to triage and remediate issues, suggesting optimal actions. These agents can also integrate with traditional machine learning models and tooling, leveraging existing systems while simultaneously offering the advantages of Generative AI. The capacity to delineate specialized agents and meticulously orchestrate their interrelationships empowers developers to construct systems exhibiting enhanced modularity, scalability, and the ability to address complexities that would prove insurmountable for a singular, integrated agent.

model presents unique advantages and challenges, influencing the overall efficiency, robustness, and adaptability of the multi-agent system. Single Agent: At the most basic level, a “Single Agent” operates autonomously without direct interaction or communication with other entities. While this model is straightforward to implement and manage, its capabilities are inherently limited by the individual agent’s scope and resources. It is suitable for tasks that are decomposable into independent sub-problems, each solvable by a single, self-sufficient agent. Network: The “Network” model represents a significant step towards collaboration, where multiple agents interact directly with each other in a decentralized fashion. Communication typically occurs peer-to-peer, allowing for the sharing of information, resources, and even tasks. This model fosters resilience, as the failure of one agent does not necessarily cripple the entire system. However, managing communication overhead and ensuring coherent decision-making in a large, unstructured network can be challenging. Supervisor: In the “Supervisor” model, a dedicated agent, the “supervisor,” oversees and coordinates the activities of a group of subordinate agents. The supervisor acts as a central hub for communication, task allocation, and conflict resolution. This hierarchical structure offers clear lines of authority and can simplify management and control. However, it introduces a single point of failure (the supervisor) and can become a bottleneck if the supervisor is overwhelmed by a large number of subordinates or complex tasks. Supervisor as a Tool: This model is a nuanced extension of the “Supervisor” concept, where the supervisor’s role is less about direct command and control and more about providing resources, guidance, or analytical support to other agents. The supervisor might offer tools, data, or computational services that enable other agents to perform their tasks more effectively, without necessarily dictating their every action. This approach aims to leverage the supervisor’s capabilities without imposing rigid top-down control. Hierarchical: The “Hierarchical” model expands upon the supervisor concept to create a multi-layered organizational structure. This involves multiple levels of supervisors, with higher-level supervisors overseeing lower-level ones, and ultimately, a collection of operational agents at the lowest tier. This structure is well-suited for complex problems that can be decomposed into sub-problems, each managed by a specific layer of the hierarchy. It provides a structured approach to scalability and complexity management, allowing for distributed decision-making within defined boundaries. Custom: The “Custom” model represents the ultimate flexibility in multi-agent system design. It allows for the creation of unique interrelationship and communication structures tailored precisely to the specific requirements of a given problem or application. This can involve hybrid approaches that combine elements from the previously mentioned models, or entirely novel designs that emerge from the unique constraints and opportunities of the environment. Custom models often arise from the need to optimize for specific performance metrics, handle highly dynamic environments, or incorporate domain-specific knowledge into the system’s architecture. Designing and implementing custom models typically requires a deep understanding of multi-agent systems principles and careful consideration of communication protocols, coordination mechanisms, and emergent behaviors. In summary, the choice of interrelationship and communication model for a multi-agent system is a critical design decision. Each model offers distinct advantages and disadvantages, and the optimal choice depends on factors such as the complexity of the task, the number of agents, the desired level of autonomy, the need for robustness, and the acceptable communication overhead. Future advancements in multi-agent systems will likely continue to explore and refine these models, as well as develop new paradigms for collaborative intelligence.

WhatComplex problems often exceed the capabilities of a single, monolithic LLM-based agent. A solitary agent may lack the diverse, specialized skills or access to the specific tools needed to address all parts of a multifaceted task. This limitation creates a bottleneck, reducing the system’s overall effectiveness and scalability. As a result, tackling sophisticated, multi-domain objectives becomes inefficient and can lead to incomplete or suboptimal outcomes. WhyThe Multi-Agent Collaboration pattern offers a standardized solution by creating a system of multiple, cooperating agents. A complex problem is broken down into smaller, more manageable sub-problems. Each sub-problem is then assigned to a specialized agent with the precise tools and capabilities required to solve it. These agents work together through defined communication protocols and interaction models like sequential handoffs, parallel workstreams, or hierarchical delegation. This agentic, distributed approach creates a synergistic effect, allowing the group to achieve outcomes that would be impossible for any single agent. Rule of ThumbUse this pattern when a task is too complex for a single agent and can be decomposed into distinct sub-tasks requiring specialized skills or tools. It is ideal for problems that benefit from diverse expertise, parallel processing, or a structured workflow with multiple stages, such as complex research and analysis, software development, or creative content generation.

Key Takeaways Multi-Agent collaboration involves multiple agents working together to achieve a common goal. This pattern leverages specialized roles, distributed tasks, and inter-agent communication. Collaboration can take forms like sequential handoffs, parallel processing, debate, or hierarchical structures.

This pattern is ideal for complex problems requiring diverse expertise or multiple distinct stages. Conclusion This chapter explored the Multi-Agent Collaboration pattern, demonstrating the benefits of orchestrating multiple specialized agents within systems. We examined various collaboration models, emphasizing the pattern’s essential role in addressing complex, multifaceted problems across diverse domains. Understanding agent collaboration naturally leads to an inquiry into their interactions with the external environment.

====

Generate README.md -- must be markdown format 
