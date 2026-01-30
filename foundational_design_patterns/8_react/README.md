# ReAct (Reasoning and Acting)

## Overview

The **ReAct Pattern** (Reasoning and Acting) is a synergistic approach that interleaves **reasoning traces** with **action execution**, enabling LLMs to solve complex tasks through a natural cycle of thinking, acting, and observing.

Unlike pure action-based agents that jump straight to tool use, or pure reasoning agents that only think without external interaction, ReAct combines both: the agent explicitly reasons about what to do, takes actions using tools, observes the results, and continues reasoning based on those observations.

## Why Use This Pattern?

Traditional approaches have limitations:

- **Pure reasoning**: Limited to internal knowledge, cannot access external data or take actions
- **Pure action**: May execute tools without proper planning, leading to inefficient or incorrect sequences
- **Chain-of-Thought alone**: Helps reasoning but can't interact with the world or update beliefs with real data

ReAct solves these by:
- **Explicit reasoning traces**: Shows the agent's thinking process before each action
- **Grounded action execution**: Uses tools to gather information and validate hypotheses
- **Dynamic adaptation**: Adjusts strategy based on observation results
- **Interpretability**: Makes decision-making transparent and debuggable
- **Error recovery**: Can recognize mistakes and course-correct

### Example: Question Answering with ReAct

```
Without ReAct (Single-shot):
User: "What was the population of the city where the 2024 Olympics were held?"
Agent: "The 2024 Olympics were held in Paris, which has a population of about 2.2 million."
→ May be incorrect, outdated, or hallucinated

With ReAct (Reasoning + Acting):
User: "What was the population of the city where the 2024 Olympics were held?"

Thought: I need to first find out which city hosted the 2024 Olympics.
Action: search("2024 Olympics host city")
Observation: The 2024 Summer Olympics were held in Paris, France.

Thought: Now I know it's Paris. I need to find the current population of Paris.
Action: search("Paris France population 2024")
Observation: As of 2024, Paris has a population of approximately 2.1 million in the city proper,
            and 12.4 million in the metropolitan area.

Thought: I have the information I need. I can now provide a complete answer.
Answer: The 2024 Olympics were held in Paris, France. The city proper has a population of
        approximately 2.1 million, while the metropolitan area has about 12.4 million people.
```

## How It Works

The ReAct loop consists of three key steps repeated until task completion:

1. **Thought (Reasoning)**: The agent reasons about the current state, what it knows, and what it needs to do next
2. **Action**: The agent executes a tool/function to gather information or perform an operation
3. **Observation**: The agent receives and processes the result of the action

```
┌─────────────────────────────────────────────────────┐
│                    User Query                        │
└───────────────────┬─────────────────────────────────┘
                    ↓
            ┌───────────────┐
            │   Thought 1   │ "What information do I need?"
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   Action 1    │ Call search_tool()
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ Observation 1 │ Receive search results
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   Thought 2   │ "Do I have enough info?"
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   Action 2    │ Call calculator_tool()
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ Observation 2 │ Receive calculation result
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   Thought 3   │ "I have everything needed"
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ Final Answer  │
            └───────────────┘
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Research and fact-checking**: Questions requiring multiple information lookups and verification
- **Multi-step problem solving**: Math problems, logic puzzles, planning tasks
- **Exploratory tasks**: When the path to solution isn't clear upfront
- **Complex reasoning with external data**: Combining analysis with real-time information
- **Debugging and troubleshooting**: Iterative investigation of issues
- **Data analysis workflows**: Query → analyze → query → synthesize
- **Decision-making under uncertainty**: Gather evidence before deciding

### ❌ When NOT to Use

- **Simple queries**: Single-step questions answerable without tools
- **Pure generation tasks**: Creative writing, brainstorming (no actions needed)
- **Well-defined workflows**: When prompt chaining or planning patterns are more appropriate
- **Real-time requirements**: ReAct adds overhead through reasoning steps
- **Limited tool access**: When tools are unavailable or restricted

## Rule of Thumb

**Use ReAct when:**
1. Task requires **multiple tool interactions** to solve
2. The **sequence of actions isn't predetermined** (exploratory)
3. You need **transparent reasoning** for debugging or compliance
4. Agent must **adapt based on intermediate results**
5. Task involves **verification or fact-checking**

**Don't use ReAct when:**
1. Single tool call suffices (use simple tool use)
2. Action sequence is fixed and known (use prompt chaining)
3. Pure reasoning without actions is enough (use CoT)
4. Latency is critical (reasoning adds overhead)

## Core Components

### 1. Reasoning Traces (Thoughts)

The agent explicitly generates reasoning about:
- What information it currently has
- What information it needs
- Which action to take next
- How to interpret observations
- Whether it has enough to answer

### 2. Action Space (Tools)

Available tools the agent can invoke:
- Search engines (web search, knowledge bases)
- Calculators (mathematical operations)
- APIs (weather, stocks, databases)
- Code execution (Python, SQL)
- Information retrieval (documents, databases)

### 3. Observation Processing

After each action, the agent:
- Receives the tool output
- Integrates it into its knowledge state
- Decides the next step based on the result

## Implementation Approaches

### Approach 1: LangGraph PreBuilt ReAct Agent

The simplest way to implement ReAct using LangGraph's built-in agent:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """Search for information on the web."""
    # Implementation
    return search_results

@tool
def calculator(expression: str) -> float:
    """Perform mathematical calculations."""
    return eval(expression)

# Create ReAct agent
llm = ChatOpenAI(model="gpt-4")
tools = [search, calculator]

agent = create_react_agent(llm, tools)

# Use the agent
result = agent.invoke({
    "messages": [("user", "What's the square root of the population of Tokyo?")]
})
```

### Approach 2: Custom ReAct Loop with LangGraph

For more control, build a custom graph:

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

def agent_node(state: MessagesState):
    """Agent decides to reason and/or act"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    """Check if agent wants to use tools or finish"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")  # Loop back after tool execution

app = workflow.compile()
```

### Approach 3: Explicit Reasoning Prompts

Structure prompts to encourage explicit reasoning:

```python
react_prompt = """You are a helpful assistant. For each task, follow this process:

1. Thought: Reason about what you need to do
2. Action: Choose a tool and provide input
3. Observation: See the tool result
4. Repeat steps 1-3 as needed
5. Answer: When you have enough information, provide the final answer

Available tools:
{tools}

Current task: {task}

Begin!"""
```

## Key Benefits

### 🧠 Enhanced Reasoning
- **Transparent thinking**: See the agent's reasoning process
- **Better decisions**: Thoughtful consideration before acting
- **Error recognition**: Agent can identify and correct mistakes

### 🎯 Improved Accuracy
- **Grounded in facts**: Actions provide real data, not hallucinations
- **Verification loops**: Can double-check and validate information
- **Adaptive strategies**: Change approach based on observations

### 🔍 Debuggability
- **Reasoning traces**: Understand why the agent made each decision
- **Step-by-step inspection**: Debug at each thought-action-observation
- **Failure analysis**: See where reasoning or actions went wrong

### 🔄 Dynamic Adaptation
- **Context-aware**: Adjusts based on what it learns
- **Flexible sequences**: Not locked into predetermined steps
- **Recovers from errors**: Can try alternative approaches

## Trade-offs

### ⚠️ Increased Latency

**Issue**: Multiple LLM calls for reasoning + action cycles add time

**Impact**: 3-10x slower than single-shot responses

**Mitigation**:
- Use faster models for reasoning steps (GPT-4o-mini, Claude Haiku)
- Cache common reasoning patterns
- Parallelize independent tool calls
- Set maximum iteration limits

### 💰 Higher Token Costs

**Issue**: Reasoning traces consume tokens without direct output value

**Impact**: 2-5x more tokens than direct tool use

**Mitigation**:
- Use cheaper models for reasoning when possible
- Implement reasoning compression techniques
- Set iteration limits to prevent runaway loops
- Cache and reuse common reasoning patterns

### 🔁 Potential for Loops

**Issue**: Agent might get stuck in unproductive reasoning-action cycles

**Mitigation**:
- Set maximum iteration counts (typically 5-10)
- Implement loop detection logic
- Add "stuck detection" heuristics
- Provide escape hatches for time limits

## Best Practices

### 1. Tool Design

```python
@tool
def search(query: str) -> str:
    """Search the web for information.

    Use this when you need current information, facts, or data
    not in your training data.

    Args:
        query: Clear, specific search query

    Returns:
        Relevant search results and excerpts
    """
    # Clear, focused functionality
    # Good error handling
    # Structured output
```

### 2. Prompt Engineering

```python
system_prompt = """You are a research assistant using the ReAct framework.

For each task:
1. THINK: What do I know? What do I need?
2. ACT: Use tools to gather information
3. OBSERVE: Integrate the results
4. REPEAT: Until you have sufficient information
5. ANSWER: Provide a complete, accurate response

Be systematic and thorough. Show your reasoning."""
```

### 3. Iteration Limits

```python
agent = create_react_agent(
    llm,
    tools,
    state_modifier="You have {remaining} iterations left."
)

# Or in custom implementation
max_iterations = 10
current_iteration = 0

while current_iteration < max_iterations:
    # ReAct loop
    current_iteration += 1
```

### 4. Observation Formatting

```python
@tool
def get_weather(location: str) -> str:
    """Get weather information."""
    data = weather_api.get(location)

    # Return structured, parseable observations
    return f"""Weather in {location}:
    Temperature: {data.temp}°F
    Conditions: {data.conditions}
    Humidity: {data.humidity}%
    Wind: {data.wind_speed} mph
    """
```

## Performance Metrics

Track these metrics for ReAct agents:

- **Task success rate**: % of tasks completed correctly
- **Average iterations**: How many thought-action-observation cycles
- **Tool utilization**: Which tools used, how often
- **Reasoning quality**: Are thoughts relevant and logical?
- **Efficiency**: Ratio of necessary vs. unnecessary actions
- **Latency per task**: Total time to completion
- **Token usage**: Reasoning overhead vs. action tokens
- **Error recovery rate**: % of self-corrections after mistakes

## Example Scenarios

### Scenario 1: Research Question

```
User: "How old was the author of 'The Old Man and the Sea' when it was published?"

Thought: I need to find out who wrote 'The Old Man and the Sea'.
Action: search("The Old Man and the Sea author")
Observation: Ernest Hemingway wrote 'The Old Man and the Sea'.

Thought: Now I need to know when it was published.
Action: search("The Old Man and the Sea publication date")
Observation: It was published in 1952.

Thought: I need Hemingway's birth year to calculate his age.
Action: search("Ernest Hemingway birth year")
Observation: Ernest Hemingway was born on July 21, 1899.

Thought: Now I can calculate: 1952 - 1899 = 53 years old.
Action: calculator("1952 - 1899")
Observation: 53

Answer: Ernest Hemingway was 53 years old when "The Old Man and the Sea"
        was published in 1952.
```

### Scenario 2: Math Word Problem

```
User: "A store has 15% off all items. If I buy 3 shirts at $25 each and
       2 pairs of pants at $40 each, what's my final price?"

Thought: I need to calculate the subtotal first.
Action: calculator("3 * 25 + 2 * 40")
Observation: 155

Thought: The subtotal is $155. Now I need to apply the 15% discount.
Action: calculator("155 * 0.15")
Observation: 23.25

Thought: The discount is $23.25. Final price = subtotal - discount.
Action: calculator("155 - 23.25")
Observation: 131.75

Answer: Your final price after the 15% discount would be $131.75.
        (Original: $155, Discount: $23.25, Final: $131.75)
```

### Scenario 3: Data Analysis

```
User: "Compare the GDP growth rates of USA and China in 2023"

Thought: I need current GDP growth data for USA in 2023.
Action: search("USA GDP growth rate 2023")
Observation: The US economy grew at 2.5% in 2023.

Thought: Now I need China's GDP growth for the same year.
Action: search("China GDP growth rate 2023")
Observation: China's GDP grew at 5.2% in 2023.

Thought: I have both figures. Let me calculate the difference.
Action: calculator("5.2 - 2.5")
Observation: 2.7

Answer: In 2023, China's GDP grew at 5.2% compared to the USA's 2.5%.
        China's growth rate was 2.7 percentage points higher.
```

## Advanced Patterns

### 1. Self-Correction

```
Thought: I need to find the capital of France.
Action: search("capital France")
Observation: Paris is the capital of France.

Thought: Wait, let me verify this is still current.
Action: search("France capital 2024")
Observation: Confirmed, Paris is the capital of France.
```

### 2. Multi-Tool Composition

```
Thought: I need weather and stock data together.
Action: get_weather("New York")
Observation: Sunny, 72°F

Thought: Good weather might affect retail stocks positively.
Action: get_stock_price("WMT")
Observation: Walmart stock at $165.30, up 1.2%
```

### 3. Reasoning About Tool Limitations

```
Thought: I should search for this, but the query might be too specific.
Action: search("very specific rare topic")
Observation: No relevant results found.

Thought: Let me broaden my search and then filter.
Action: search("broader related topic")
Observation: Found relevant context...
```

## Comparison with Related Patterns

| Pattern | Reasoning | Actions | When to Use |
|---------|-----------|---------|-------------|
| **ReAct** | Explicit, interleaved | Yes, dynamic | Multi-step exploration |
| **Tool Use** | Implicit | Yes, single/parallel | Known tools needed |
| **Prompt Chain** | Implicit | No | Fixed sequence |
| **Planning** | Upfront plan | Follows plan | Complex, predictable |
| **Reflection** | Post-hoc critique | Optional refinement | Quality critical |

## Common Pitfalls

### 1. Excessive Reasoning

**Problem**: Agent overthinks simple tasks

**Solution**: Prompt engineering to be concise; use simpler models for reasoning

### 2. Tool Overuse

**Problem**: Calls tools unnecessarily when it already has the answer

**Solution**: Include "based on what I know..." reasoning; encourage using existing information

### 3. Circular Reasoning

**Problem**: Repeats the same actions without progress

**Solution**: Track action history; detect loops; limit iterations

### 4. Poor Observation Integration

**Problem**: Doesn't properly use tool results in next reasoning step

**Solution**: Prompt to explicitly reference observations; structured observation format

## Conclusion

The ReAct pattern represents a significant advance in agentic AI by combining the strengths of reasoning and acting. It enables transparent, adaptive, and accurate problem-solving for complex tasks requiring external information and dynamic decision-making.

**Use ReAct when:**
- Tasks require exploratory, multi-step tool use
- Transparency and debuggability are important
- The solution path isn't predetermined
- You need verification and error correction
- Combining reasoning with real-world data

**Implementation checklist:**
- ✅ Define clear, well-documented tools
- ✅ Use prompts that encourage explicit reasoning
- ✅ Set iteration limits to prevent loops
- ✅ Format observations for easy parsing
- ✅ Log reasoning traces for debugging
- ✅ Monitor token usage and latency
- ✅ Implement error handling and recovery

**Key Takeaways:**
- 🔄 ReAct interleaves Thought, Action, and Observation
- 🧠 Explicit reasoning improves decision quality
- 🎯 Grounded actions reduce hallucinations
- 🔍 Reasoning traces enable debugging
- ⚡ Trade-off: Better accuracy vs. higher latency/cost
- 🛠️ LangGraph provides excellent ReAct support

---

*ReAct transforms LLMs from passive responders into active problem-solvers that think, act, and learn—making them capable of tackling complex, real-world tasks with transparency and adaptability.*
