# Planning - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd foundational_design_patterns/6_planning
```

### Step 2: Run the Example
```bash
bash run.sh
```

---

## 📖 Understanding Planning in 30 Seconds

**Planning** decomposes complex goals into structured, executable action plans:

```
Without Planning:                With Planning:
Complex Goal                     Complex Goal
    ↓                               ↓
React + Execute                  Analyze → Plan
(chaotic, incomplete)                ↓
                                 Step 1 → Step 2 → Step 3
                                 (systematic, complete)
```

Think before acting!

---

## 🎯 What This Example Does

The example demonstrates **strategic planning**:

1. **Analyze** - Understand the goal and requirements
2. **Decompose** - Break into manageable subtasks
3. **Sequence** - Order tasks logically
4. **Execute** - Follow the plan systematically
5. **Adapt** - Replan if needed based on results

---

## 💡 Example Flow

```
Goal: "Create a market research report"
    ↓
Planning Phase:
    Step 1: Identify target market and competitors
    Step 2: Gather data from multiple sources
    Step 3: Analyze trends and patterns
    Step 4: Create visualizations
    Step 5: Write executive summary
    Step 6: Compile final report
    ↓
Execution Phase:
    Execute Step 1 → Results → ✓
    Execute Step 2 → Results → ✓
    Execute Step 3 → Results → ✓
    ...
    ↓
Final Output: Complete market research report
```

---

## 🔧 Key Concepts

### Upfront Planning
Create a strategy before executing tasks.

### Task Decomposition
Break complex goals into smaller, manageable steps.

### Dependency Management
Understand which tasks must happen before others.

### Dynamic Replanning
Adjust the plan based on intermediate results.

---

## 🎨 When to Use Planning

✅ **Good For:**
- Multi-step workflows requiring orchestration
- Complex goals with interdependent tasks
- Research reports, data pipelines
- Project setup and onboarding
- Strategic problem-solving

❌ **Not Ideal For:**
- Simple, single-step tasks
- Well-known workflows (use prompt chaining)
- Real-time reactive systems
- Tasks where exploration > planning

---

## 🛠️ Planning Approaches

### 1. Simple Plan-Then-Execute
```python
# Phase 1: Planning
plan = planner_llm.invoke(f"Create a plan for: {goal}")

# Phase 2: Execution
for step in plan.steps:
    result = execute_step(step)

return final_result
```

### 2. Plan-Execute-Replan Loop
```python
plan = create_plan(goal)

while not goal_achieved:
    # Execute next step
    result = execute(plan.next_step)

    # Check if replanning needed
    if needs_replanning(result):
        plan = update_plan(plan, result)

return result
```

### 3. LangGraph Planning Agent
```python
from langchain.agents import create_agent

# Define planning and execution nodes
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_node)
workflow.add_node("replanner", replan_node)

# Add edges with conditional logic
workflow.add_conditional_edges(...)

app = workflow.compile()
```

---

## 📊 Planning vs Reactive

| Aspect | Reactive (No Planning) | With Planning |
|--------|----------------------|---------------|
| Completeness | 60-70% | 90-95% |
| Efficiency | Lower (redundant work) | Higher (optimized path) |
| Transparency | Low (ad-hoc decisions) | High (clear strategy) |
| Overhead | Low | +20-40% time/tokens |
| Quality | Variable | Consistent |

---

## 💡 Planning Patterns

### 1. Linear Planning
```
Goal → [Step 1 → Step 2 → Step 3] → Result
```

### 2. Hierarchical Planning
```
Goal → [
    SubGoal 1 → [Task 1.1, Task 1.2]
    SubGoal 2 → [Task 2.1, Task 2.2]
] → Result
```

### 3. Dependency-Aware Planning
```
          Task A
         ↙      ↘
    Task B      Task C
         ↘      ↙
          Task D
```

### 4. Adaptive Planning
```
Plan v1 → Execute → Results → Replan v2 → Execute → Done
```

---

## 🔧 Customization Tips

### Define Planning Prompt
```python
planning_prompt = """
Given this goal: {goal}

Create a detailed plan with:
1. Numbered steps in logical order
2. Dependencies between steps
3. Expected outcome for each step
4. Estimated effort/time
5. Success criteria

Plan:
"""
```

### Execute with Validation
```python
for step in plan.steps:
    result = execute_step(step)

    # Validate result
    if not validate(result):
        # Replan or retry
        plan = update_plan(plan, result)
```

### Add Progress Tracking
```python
class Plan:
    def __init__(self, steps):
        self.steps = steps
        self.completed = []
        self.current = 0

    def progress(self):
        return len(self.completed) / len(self.steps)
```

---

## 🐛 Common Issues & Solutions

### Issue: Plan Too Generic
**Solution**: Use more specific planning prompts with examples.

### Issue: Plan Never Completes
**Solution**: Set maximum steps and time limits.

### Issue: Execution Deviates from Plan
**Solution**: Implement plan validation and replanning.

### Issue: Over-Planning
**Solution**: Balance planning depth with execution needs.

---

## 📚 Real-World Applications

### Research Reports
```
Plan: Define scope → Research → Analyze → Visualize → Write → Review
```

### Software Development
```
Plan: Requirements → Architecture → Implementation → Testing → Deployment
```

### Data Pipelines
```
Plan: Extract → Validate → Transform → Enrich → Load → Monitor
```

### Customer Onboarding
```
Plan: Setup account → Configure → Train → Verify → Go-live
```

---

## 🎓 Advanced Techniques

### Multi-Level Planning
Create high-level plans, then detailed sub-plans for each step.

### Conditional Planning
Include if-then logic in plans for different scenarios.

### Resource-Aware Planning
Consider available tools, time, and budget in planning.

### Collaborative Planning
Multiple agents contribute to plan creation.

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)
- **Related Patterns**:
  - Pattern 1 (Prompt Chaining) - Execute planned steps
  - Pattern 7 (Multi-Agent) - Distribute plan execution

---

## 🎓 Next Steps

1. ✅ Run the planning example
2. ✅ Observe the plan structure
3. ✅ Modify the goal
4. ✅ Implement custom validation
5. ✅ Try replanning on failures

---

**Pattern Type**: Strategic Orchestration

**Complexity**: ⭐⭐⭐⭐ (Advanced)

**Best For**: Complex multi-step goals, orchestration

**Overhead**: +20-40% time/tokens for planning

**Benefit**: 90%+ task completeness vs 60-70% reactive
