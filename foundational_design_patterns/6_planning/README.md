# Planning Pattern

## Overview

The **Planning Pattern** enables agentic systems to decompose complex, multi-step goals into structured sequences of actionable sub-tasks, creating a strategic roadmap before execution. This transforms reactive agents into proactive, strategic executors capable of handling sophisticated workflows.

## Why Use This Pattern?

Many real-world tasks are too complex to be solved with a single action or prompt. They require:
- Multiple interdependent steps executed in a specific order
- Coordination across different tools and resources
- Adaptive decision-making based on intermediate results
- Management of dependencies and prerequisites

The Planning pattern solves this by:
- **Decomposing complex goals** into manageable sub-tasks
- **Creating structured execution plans** with clear sequences
- **Identifying dependencies** between steps
- **Enabling strategic thinking** rather than reactive responses
- **Facilitating adaptation** when plans need to change

### Example: Without vs. With Planning
```
Without Planning (Reactive):
User: "Create a comprehensive market analysis report"
Agent: [Attempts to generate entire report in one shot]
Result: Incomplete, lacks depth, misses key aspects

With Planning (Strategic):
User: "Create a comprehensive market analysis report"
Agent: 
  Step 1: Identify key competitors and market segments
  Step 2: Gather data on each competitor (tools: web search, databases)
  Step 3: Analyze market trends and growth patterns
  Step 4: Collect customer sentiment data
  Step 5: Synthesize findings into structured report
  Step 6: Generate visualizations for key metrics
  Step 7: Write executive summary
→ Executes each step systematically
Result: Comprehensive, well-structured, data-driven report
```

## How It Works

1. **Goal Analysis**: Agent analyzes the high-level objective and requirements
2. **Task Decomposition**: Break down the goal into smaller, actionable sub-tasks
3. **Dependency Mapping**: Identify which tasks depend on others (sequencing)
4. **Resource Allocation**: Determine which tools/APIs are needed for each step
5. **Plan Generation**: Create structured execution plan with ordered steps
6. **Execution**: Execute steps sequentially (or in parallel when possible)
7. **Monitoring & Adaptation**: Track progress and adjust plan if needed

### Typical Architecture
```
User Goal/Objective
    ↓
Planning Agent
    ↓
[Analyze Goal]
    ↓
[Decompose into Sub-tasks]
    ↓
Generate Execution Plan:
  - Step 1: Research competitors
  - Step 2: Analyze market data  
  - Step 3: Gather customer feedback
  - Step 4: Synthesize findings
  - Step 5: Generate report
    ↓
Execute Step 1
    ↓
Execute Step 2
    ↓
Execute Step 3
    ↓
[Monitor Progress & Adapt if Needed]
    ↓
Execute Step 4
    ↓
Execute Step 5
    ↓
Final Output
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Research & Analysis**: Competitive analysis, market research, literature reviews
- **Content Creation**: Multi-section reports, comprehensive guides, documentation
- **Workflow Automation**: Employee onboarding, procurement processes, project setup
- **Data Pipeline**: Extract → Transform → Analyze → Visualize → Report
- **Complex Problem Solving**: Debugging systems, optimizing processes, strategic planning
- **Multi-tool Orchestration**: Tasks requiring coordinated use of multiple APIs/tools
- **Project Management**: Breaking down projects into phases and deliverables

### ❌ When NOT to Use

- **Simple, single-step tasks**: Direct questions, basic lookups, simple calculations
- **Time-critical responses**: Real-time chat, immediate answers
- **Well-defined, atomic operations**: Tasks that are already clearly scoped
- **Exploratory queries**: Open-ended browsing without clear objectives
- **When existing patterns suffice**: Use Prompt Chaining for pre-defined sequences

## Rule of Thumb

**Use Planning when:**
1. User's request is **too complex for a single action or tool**
2. Task requires **multiple interdependent operations** in sequence
3. Success depends on **executing steps in the right order**
4. You need to **orchestrate multiple tools/resources** strategically
5. The goal requires **synthesizing information from multiple sources**

**Don't use Planning when:**
1. Task is simple and direct (single tool call or prompt)
2. Sequence of steps is already pre-defined (use Prompt Chaining)
3. Real-time response is critical (planning adds latency)
4. Goal is exploratory without clear success criteria

## Framework Support

### LangChain with Plan-and-Execute
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0, model="gpt-4")

# Planning prompt
planning_prompt = ChatPromptTemplate.from_template(
    """Given this objective: {objective}
    
    Create a detailed step-by-step plan to accomplish it.
    
    For each step:
    1. Describe what needs to be done
    2. Identify required tools or resources
    3. Note any dependencies on previous steps
    
    Plan:"""
)

# Execution prompt for each step
execution_prompt = ChatPromptTemplate.from_template(
    """Execute this step: {step}
    
    Context from previous steps: {context}
    
    Available tools: {tools}
    
    Result:"""
)

# Create chains
planner = planning_prompt | llm
executor = execution_prompt | llm

# Generate plan
plan = planner.invoke({"objective": "Create a market analysis report"})

# Execute each step
results = []
for step in plan.split('\n'):
    if step.strip():
        result = executor.invoke({
            "step": step,
            "context": results,
            "tools": available_tools
        })
        results.append(result)
```

### LangGraph (Stateful Planning)
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class PlanningState(TypedDict):
    objective: str
    plan: List[str]
    current_step: int
    results: List[dict]
    complete: bool

def plan_node(state: PlanningState):
    """Generate execution plan"""
    plan = generate_plan(state["objective"])
    return {"plan": plan, "current_step": 0}

def execute_step_node(state: PlanningState):
    """Execute current step in plan"""
    step = state["plan"][state["current_step"]]
    result = execute_step(step, state["results"])
    
    return {
        "results": state["results"] + [result],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: PlanningState):
    if state["current_step"] >= len(state["plan"]):
        return "end"
    return "execute"

# Build graph
workflow = StateGraph(PlanningState)
workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_step_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue, {
    "execute": "executor",
    "end": END
})

app = workflow.compile()
```

### Google ADK (Deep Research)

Google's Deep Research demonstrates planning in action:
```python
# Google Deep Research uses Planning pattern internally:
# 1. Analyzes research question
# 2. Generates search strategy (plan)
# 3. Executes searches sequentially
# 4. Reflects on findings
# 5. Adapts plan based on results
# 6. Synthesizes comprehensive report
```

### ReAct Pattern Integration

Combine Planning with ReAct (Reasoning + Acting):
```python
# ReAct loop within planned steps
for step in plan:
    # Reason: What should I do for this step?
    reasoning = reason_about_step(step)
    
    # Act: Execute the action
    action = determine_action(reasoning)
    result = execute_action(action)
    
    # Observe: What happened?
    observation = observe_result(result)
    
    # Update plan if needed
    if observation.requires_replan:
        plan = replan(objective, completed_steps, observation)
```

## Key Benefits

### 🎯 Strategic Execution
- **Structured approach**: Clear roadmap from start to finish
- **Logical sequencing**: Tasks executed in optimal order
- **Dependency management**: Prerequisites handled automatically
- **Resource optimization**: Right tool for each step

### 🧠 Complex Problem Solving
- **Decomposition**: Break overwhelming tasks into manageable pieces
- **Systematic coverage**: Ensure all aspects are addressed
- **Quality improvement**: Each step can be optimized independently
- **Transparency**: Clear visibility into what's being done and why

### 🔄 Adaptability
- **Dynamic replanning**: Adjust strategy based on intermediate results
- **Error recovery**: Identify and retry failed steps
- **Incremental progress**: Partial completion still provides value
- **Learning**: Improve future plans based on execution outcomes

## Important Considerations

### ⚠️ Planning Overhead

**Initial Latency:**
- Planning phase adds upfront delay (5-15 seconds)
- More complex than direct execution
- Requires additional LLM calls for plan generation

**Token Costs:**
- Plan generation: 200-500 tokens
- Plan storage in context: Carried through execution
- Total: +20-40% token usage vs. direct execution

**When Overhead is Worth It:**
- Complex tasks where planning saves overall time
- Multi-step workflows that would fail without structure
- Tasks requiring tool coordination

### 🔍 Plan Quality Matters

**Good Plans:**
- Clear, specific steps with measurable outcomes
- Realistic resource requirements
- Proper dependency ordering
- Adaptable to changing conditions

**Poor Plans:**
- Vague steps ("do research", "analyze data")
- Missing dependencies (step 3 needs output from step 2)
- Unrealistic assumptions about available tools
- Too rigid (can't adapt to failures)

### 🛠️ Implementation Complexity

**Simple Planning (LangChain):**
- Generate plan → Execute steps sequentially
- Good for predictable workflows
- Limited adaptability

**Advanced Planning (LangGraph):**
- Stateful plan execution
- Dynamic replanning
- Parallel step execution when possible
- More complex but more powerful

## Best Practices

1. **Make plans concrete and actionable**: Each step should be executable
2. **Include success criteria**: How to know if a step succeeded
3. **Anticipate failure modes**: What to do if steps fail
4. **Enable plan visibility**: Show users the plan before execution
5. **Support human-in-the-loop**: Allow plan approval/modification
6. **Log plan execution**: Track which steps completed, which failed
7. **Implement replanning**: Adapt when circumstances change
8. **Balance detail and flexibility**: Detailed enough to execute, flexible enough to adapt

## Performance Metrics

Track these metrics for planning effectiveness:

- **Plan quality**: Completeness, accuracy, feasibility
- **Execution success rate**: % of plans that complete successfully
- **Time to completion**: Total time from goal to result
- **Step failure rate**: % of individual steps that fail
- **Replanning frequency**: How often plans need adjustment
- **Resource efficiency**: Tools/API calls used vs. optimal
- **User satisfaction**: Did the result meet the objective?

## Example Scenarios

### Scenario 1: Competitive Analysis Report
```
Objective: "Create a competitive analysis comparing our product to top 3 competitors"

Generated Plan:
1. Identify top 3 competitors (Tool: Web Search)
2. For each competitor:
   a. Gather product features (Tool: Web Scraping)
   b. Collect pricing information (Tool: Web Search)
   c. Analyze customer reviews (Tool: Review API)
3. Compare features side-by-side (Tool: Spreadsheet)
4. Analyze pricing strategies (Tool: Data Analysis)
5. Synthesize strengths/weaknesses (Tool: LLM)
6. Generate visual comparisons (Tool: Charts)
7. Write executive summary (Tool: LLM)
8. Compile into report (Tool: Document Generator)

Execution: 8 steps, 45 seconds, 12 tool calls
Result: Comprehensive 10-page report with data, visuals, insights
```

### Scenario 2: Employee Onboarding Automation
```
Objective: "Onboard new software engineer Sarah Chen"

Generated Plan:
1. Create email account (Tool: Google Workspace API)
2. Add to Slack channels (Tool: Slack API)
3. Grant GitHub access (Tool: GitHub API)
4. Assign onboarding buddy (Tool: HR Database)
5. Schedule orientation meetings (Tool: Calendar API)
6. Send welcome email with credentials (Tool: Email)
7. Create Jira account (Tool: Jira API)
8. Add to team distribution lists (Tool: Email API)
9. Order equipment (Tool: Procurement System)
10. Generate welcome packet (Tool: Document Generator)

Execution: 10 steps, 30 seconds, 10 tool calls
Result: Fully onboarded employee with all access and resources
```

### Scenario 3: Data Pipeline
```
Objective: "Analyze Q4 sales data and create executive dashboard"

Generated Plan:
1. Extract sales data from database (Tool: SQL Query)
2. Clean and validate data (Tool: Pandas)
3. Calculate key metrics (Tool: Data Analysis)
   - Revenue by region
   - Growth rates
   - Top products
4. Generate trend visualizations (Tool: Matplotlib/Seaborn)
5. Create comparison charts (Tool: Visualization)
6. Build interactive dashboard (Tool: Dashboard Framework)
7. Write insights summary (Tool: LLM)
8. Deploy dashboard (Tool: Deployment)

Execution: 8 steps, 60 seconds, 8 tool calls
Result: Live dashboard with visualizations and insights
```

## Planning Strategies

### Top-Down Decomposition
```
Goal: Create marketing campaign
├─ Research target audience
├─ Develop messaging strategy
├─ Create content assets
│  ├─ Write copy
│  ├─ Design visuals
│  └─ Produce videos
├─ Set up distribution channels
└─ Launch and monitor
```

### Dependency-Aware Planning
```
Step 1: Gather requirements (no dependencies)
Step 2: Design architecture (depends on Step 1)
Step 3a: Build frontend (depends on Step 2)
Step 3b: Build backend (depends on Step 2)  [Can run in parallel]
Step 4: Integration testing (depends on 3a, 3b)
Step 5: Deployment (depends on Step 4)
```

### Iterative Planning (Agile)
```
Sprint 1 Plan:
- User research
- MVP design
- Core features

[Execute Sprint 1]

Sprint 2 Plan (adapted based on Sprint 1 results):
- Implement feedback
- Add requested features
- Performance optimization
```

## Related Patterns

- **Prompt Chaining**: Fixed sequence vs. dynamic planning
- **Tool Use**: Plans orchestrate tool usage
- **Reflection**: Can reflect on plan quality before execution
- **Routing**: Different plans for different goal types
- **Parallelization**: Execute independent plan steps concurrently

## Conclusion

The Planning pattern is essential for building agents that can handle complex, multi-step objectives with strategic thinking. By decomposing goals into actionable plans before execution, agents become proactive problem-solvers rather than reactive responders.

**Use Planning when:**
- Tasks are too complex for single-step execution
- Multiple tools need orchestration
- Steps have dependencies requiring specific ordering
- You need transparency into the execution strategy
- Adaptation based on intermediate results is valuable

**Implementation guidance:**
- Start with **simple linear plans** for predictable workflows
- Add **replanning capabilities** for dynamic environments
- Enable **human oversight** for high-stakes plans
- **Log execution thoroughly** for debugging and improvement
- **Measure plan quality** through execution success rates

**Key Takeaways:**
- 🎯 Planning transforms complex goals into structured, executable steps
- 🧠 Essential for multi-step tasks, workflow automation, and tool orchestration
- 📊 LLMs excel at generating contextually appropriate plans
- ⚡ Adds overhead but enables solving otherwise intractable problems
- 🔄 Dynamic replanning enables adaptation to changing conditions
- 🛠️ Well-supported by frameworks like LangGraph and AutoGen

---

*Planning elevates agents from reactive executors to strategic problem-solvers capable of tackling complex, real-world objectives.*