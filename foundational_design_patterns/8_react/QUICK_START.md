# ReAct Pattern - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the ReAct Directory
```bash
cd foundational_design_patterns/8_react
```

### Step 2: Install Dependencies (if not already installed)
```bash
uv sync
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic ReAct Agent (simpler, using prebuilt)
- **Option 2**: Advanced ReAct Agent (custom, with explicit traces)
- **Option 3**: Run all examples

---

## 📖 Understanding ReAct in 30 Seconds

**ReAct** = **Rea**soning + **Act**ing

The agent follows this loop:
1. **Thought**: "What do I need to know?"
2. **Action**: Use a tool to get information
3. **Observation**: See what the tool returned
4. **Repeat** until it has enough information
5. **Answer**: Provide the final response

---

## 🛠️ Available Tools

### Basic Implementation
- `search(query)` - Search knowledge base for facts
- `calculator(expression)` - Perform math calculations
- `get_word_count(text)` - Analyze text statistics

### Advanced Implementation
- `wikipedia_search(query)` - Search Wikipedia articles
- `scientific_calculator(operation, a, b)` - Scientific calculations
- `text_analyzer(text, type)` - Full text analysis

---

## 💡 Example Queries to Try

### Simple Research
```
"What is photosynthesis?"
```

### Multi-Step Problem
```
"If Tokyo has 14 million people and the Eiffel Tower is 330 meters tall,
what is their sum?"
```

### Research + Analysis
```
"Tell me about Marie Curie and count the words in her description."
```

### Complex Query
```
"Look up Albert Einstein, find what year he won the Nobel Prize,
then calculate 20% of that year."
```

---

## 🎯 Key Concepts

### Thought (Reasoning)
The agent explicitly states what it's thinking:
- "I need to find X first"
- "Now I have Y, I should calculate Z"
- "I have enough information to answer"

### Action (Tool Use)
The agent calls a tool with specific parameters:
- `search("Albert Einstein")`
- `calculator("2024 - 1879")`
- `text_analyzer("Some text...", "words")`

### Observation (Result Integration)
The agent receives and processes the tool output:
- "The search returned: ..."
- "The calculation result is: ..."
- "Based on this information, ..."

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| Implementation | `create_react_agent()` | Custom `StateGraph` |
| Reasoning Traces | Implicit | Explicit |
| Iteration Tracking | Automatic | Manual with limits |
| Output Format | Simple | Formatted with boxes |
| Customization | Limited | Fully customizable |
| Complexity | Low | Medium |

**Recommendation**: Start with Basic, move to Advanced for production.

---

## 🔧 Customization Tips

### Add Your Own Tool

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(param: str) -> str:
    """Description of what this tool does.

    Args:
        param: Description of the parameter

    Returns:
        Description of the return value
    """
    # Your implementation here
    return f"Result: {param}"

# Add to tools list
tools = [search, calculator, my_custom_tool]
```

### Adjust Iteration Limits

In `react_agent_advanced.py`, change:
```python
max_iterations: int = 10  # Change this value
```

### Modify System Prompt

Edit the `REACT_SYSTEM_PROMPT` in `react_agent_advanced.py` to customize reasoning behavior.

---

## ⚡ Common Issues & Solutions

### Issue: "Maximum iterations reached"
**Solution**: Increase `max_iterations` or simplify the query.

### Issue: "Tool not found" error
**Solution**: Check tool is in the `tools` list and properly decorated with `@tool`.

### Issue: Agent doesn't use tools
**Solution**: Make the query more specific about needing external information.

### Issue: Circular reasoning
**Solution**: Iteration limits prevent this, but you can add loop detection logic.

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Implementation Details**: See [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ Start: Run the basic example
2. ✅ Understand: Read the console output, see Thought → Action → Observation
3. ✅ Explore: Run the advanced example to see explicit traces
4. ✅ Experiment: Modify queries and see how the agent adapts
5. ✅ Customize: Add your own tools and try custom queries
6. ✅ Integrate: Use ReAct in your own applications

---

## 🌟 Pro Tips

1. **Clear Queries**: Specific questions get better results
2. **Tool Design**: Make tools do ONE thing well
3. **Error Handling**: Tools should return useful error messages
4. **Iteration Limits**: Always set a maximum to prevent infinite loops
5. **Observation Format**: Structure tool outputs for easy parsing
6. **System Prompts**: Guide reasoning with clear instructions

---

**Happy ReActing! 🚀**

For questions or issues, refer to the full [README.md](./README.md).
