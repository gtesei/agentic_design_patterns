# Goal Management Pattern

## Overview

The **Goal Management Pattern** is an orchestration approach that enables AI agents to break down complex, multi-step objectives into manageable subgoals, track progress, manage dependencies, and dynamically adapt execution strategies. Unlike simple task execution or fixed planning, Goal Management provides a flexible framework for handling uncertainty, resource constraints, and changing conditions while maintaining clear visibility into progress toward long-term objectives.

This pattern transforms monolithic tasks into hierarchical goal structures where each goal can be decomposed, prioritized, executed in parallel or sequence, monitored continuously, and replanned when necessary. It's particularly valuable for complex projects, research initiatives, software development, and any scenario where success requires coordinating multiple interdependent activities over time.

## Why Use This Pattern?

Traditional approaches have significant limitations:

- **Single-step execution**: Cannot handle complex objectives requiring multiple coordinated actions
- **Fixed planning**: Fails when conditions change or unexpected obstacles arise
- **No progress visibility**: Difficult to track what's done, what's in progress, and what remains
- **No dependency management**: Cannot handle prerequisites or parallelizable work
- **No adaptation**: Stuck with initial plan even when better approaches emerge

Goal Management solves these by:
- **Hierarchical decomposition**: Break complex goals into manageable subgoals and tasks
- **Dependency tracking**: Understand and respect prerequisites and relationships
- **Progress monitoring**: Real-time visibility into completion status and blockers
- **Dynamic replanning**: Adapt strategy when goals fail or conditions change
- **Resource optimization**: Execute tasks in parallel when possible, respect constraints
- **Continuous learning**: Improve strategies based on successes and failures

### Example: Research Report Without Goal Management

```
User: "Write a comprehensive research report on renewable energy trends"

Agent: "Here's a report on renewable energy..."
→ Rushed, incomplete analysis
→ Missing key data sources
→ No systematic coverage
→ Cannot track progress
→ No ability to handle obstacles
```

### Example: Research Report With Goal Management

```
User: "Write a comprehensive research report on renewable energy trends"

Main Goal: Complete Renewable Energy Research Report
├─ Subgoal 1: Define Research Scope [COMPLETED]
│  ├─ Task: Identify key topics (solar, wind, hydro, storage)
│  └─ Task: Define time period (2020-2024)
├─ Subgoal 2: Gather Data [IN PROGRESS - 60%]
│  ├─ Task: Search academic papers [COMPLETED]
│  ├─ Task: Analyze market reports [IN PROGRESS]
│  └─ Task: Collect statistics [PENDING - blocked by market analysis]
├─ Subgoal 3: Analyze Trends [PENDING - waiting for data]
│  ├─ Task: Identify growth patterns
│  ├─ Task: Compare technologies
│  └─ Task: Forecast future developments
└─ Subgoal 4: Write Report [PENDING]
   ├─ Task: Draft sections
   ├─ Task: Create visualizations
   └─ Task: Review and refine

Progress: 35% complete, 2 blockers identified
Next: Complete market report analysis, then statistics collection
```

## How It Works

The Goal Management pattern operates through five interconnected phases:

```
┌────────────────────────────────────────────────────────────────┐
│                     1. GOAL DECOMPOSITION                       │
│  Complex objective → Hierarchical structure of subgoals/tasks  │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     2. DEPENDENCY MAPPING                       │
│  Identify prerequisites, blockers, and parallelizable work     │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     3. PRIORITY ASSIGNMENT                      │
│  Determine execution order based on dependencies & importance  │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     4. EXECUTION & MONITORING                   │
│  Execute tasks, track progress, detect blockers & failures     │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     5. ADAPTATION & REPLANNING                  │
│  Adjust strategy, replan failed goals, optimize execution      │
└────────────────────────────────────────────────────────────────┘
          ↓                                              ↑
          └──────────────────────────────────────────────┘
                    (Continuous loop)
```

### Phase Breakdown

1. **Decomposition**: Break the main goal into a hierarchy of subgoals and executable tasks
2. **Dependency Mapping**: Identify which tasks must complete before others can start
3. **Priority Assignment**: Determine execution order considering urgency, dependencies, and resources
4. **Execution & Monitoring**: Execute ready tasks, track status, collect results
5. **Adaptation**: When failures occur or conditions change, replan and adjust strategy

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Long-running projects**: Tasks spanning hours, days, or longer with multiple phases
- **Complex objectives**: Goals requiring 5+ interdependent activities
- **Research and analysis**: Systematic investigation with multiple data sources
- **Software development**: Feature implementation with design, coding, testing, deployment phases
- **Event planning**: Coordinate venue, catering, invitations, logistics, etc.
- **Multi-agent coordination**: Multiple agents working toward shared goals
- **Uncertain environments**: When initial plan may need adjustment based on discoveries
- **Resource-constrained execution**: Need to optimize parallel work and sequencing
- **Progress visibility required**: Stakeholders need to track status and completion

### ❌ When NOT to Use

- **Simple single-step tasks**: "What's the weather?" doesn't need goal decomposition
- **Well-defined linear workflows**: Fixed sequences better served by prompt chaining
- **Real-time interactive tasks**: Goal management overhead too high for chat/conversation
- **Creative brainstorming**: Open-ended exploration doesn't fit goal structures
- **Stateless operations**: When each request is independent

## Rule of Thumb

**Use Goal Management when:**
1. Task has **3+ major phases** or components
2. Some work can be done **in parallel**
3. **Progress tracking** is valuable
4. **Adaptation** may be needed during execution
5. **Dependencies** exist between subtasks
6. Project spans **multiple sessions** or considerable time

**Don't use Goal Management when:**
1. Task completes in single step or fixed sequence
2. No benefit from parallel execution
3. Progress visibility not needed
4. Plan won't need adjustment
5. Real-time responsiveness critical

## Core Components

### 1. Goal Hierarchy

Goals are organized in a tree structure:

```python
Goal:
  - id: Unique identifier
  - title: What to accomplish
  - description: Detailed explanation
  - type: "main" | "subgoal" | "task"
  - status: "pending" | "in_progress" | "completed" | "failed" | "blocked"
  - parent_id: Link to parent goal
  - children: List of child goals/tasks
  - dependencies: Goals that must complete first
  - priority: Urgency and importance score
  - estimated_effort: Time/resource estimate
  - actual_effort: Actual time/resources used
  - result: Outcome when completed
  - metadata: Custom attributes
```

### 2. Dependency Graph

Represents relationships between goals:

- **Prerequisites**: Goals that must complete before this one starts
- **Blockers**: Current obstacles preventing progress
- **Enables**: Goals that become available after this completes
- **Parallel groups**: Goals that can execute simultaneously

### 3. Progress Tracker

Monitors execution state:

- **Completion percentage**: Overall and per-goal progress
- **Time tracking**: Started, duration, estimated completion
- **Resource usage**: Compute, API calls, tool usage
- **Bottleneck detection**: Identify critical path and blockers
- **Success metrics**: Quality, accuracy, completeness

### 4. Execution Scheduler

Determines what to execute next:

- **Ready queue**: Tasks with dependencies satisfied
- **Waiting queue**: Tasks blocked by dependencies
- **In-progress**: Currently executing tasks
- **Priority scoring**: Combine urgency, importance, effort
- **Resource allocation**: Assign available resources optimally

### 5. Replanning Engine

Handles failures and changes:

- **Failure analysis**: Why did goal fail?
- **Alternative strategies**: Different approaches to same goal
- **Partial success**: What was accomplished, what remains
- **Context updates**: New information affecting plan
- **Optimization**: Better ways discovered during execution

## Implementation Approaches

### Approach 1: Hierarchical Goal Tree

Simple parent-child relationships:

```python
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Goal:
    id: str
    title: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    children: List['Goal'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10 scale
    result: Optional[str] = None

    def is_ready(self, completed_goals: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_goals for dep in self.dependencies)

    def progress(self) -> float:
        """Calculate completion percentage"""
        if not self.children:
            return 1.0 if self.status == GoalStatus.COMPLETED else 0.0
        completed = sum(child.status == GoalStatus.COMPLETED for child in self.children)
        return completed / len(self.children)

class GoalManager:
    def __init__(self):
        self.goals: dict[str, Goal] = {}
        self.completed: set[str] = set()

    def decompose(self, main_goal: str) -> Goal:
        """Decompose main goal into subgoals and tasks"""
        # Use LLM to analyze goal and create hierarchy
        pass

    def get_ready_goals(self) -> List[Goal]:
        """Get goals ready to execute"""
        return [g for g in self.goals.values()
                if g.status == GoalStatus.PENDING
                and g.is_ready(self.completed)]

    def execute_goal(self, goal: Goal):
        """Execute a single goal"""
        goal.status = GoalStatus.IN_PROGRESS
        # Execute with LLM/tools
        # Update status based on result
        pass

    def replan_failed(self, goal: Goal):
        """Create new plan for failed goal"""
        # Analyze failure and create alternative approach
        pass
```

### Approach 2: Goal Graph with LangGraph

Use LangGraph's state management for goal orchestration:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

class GoalState(TypedDict):
    main_goal: str
    goal_tree: dict
    completed_goals: Annotated[List[str], operator.add]
    current_goal: str
    progress: float
    status: str
    results: dict

def decompose_goal_node(state: GoalState):
    """Decompose main goal into subgoals"""
    # LLM analyzes goal and creates hierarchy
    goal_tree = llm_decompose(state["main_goal"])
    return {"goal_tree": goal_tree, "status": "planning"}

def select_next_goal_node(state: GoalState):
    """Choose next goal to execute based on dependencies and priority"""
    ready_goals = get_ready_goals(state["goal_tree"], state["completed_goals"])
    if not ready_goals:
        return {"status": "complete"}

    next_goal = max(ready_goals, key=lambda g: g.priority)
    return {"current_goal": next_goal.id, "status": "executing"}

def execute_goal_node(state: GoalState):
    """Execute the current goal"""
    goal = state["goal_tree"][state["current_goal"]]
    result = llm_execute(goal)

    return {
        "completed_goals": [goal.id],
        "results": {goal.id: result},
        "progress": calculate_progress(state["goal_tree"], state["completed_goals"])
    }

def should_continue(state: GoalState):
    """Decide whether to continue or finish"""
    if state["progress"] >= 1.0:
        return "end"
    return "continue"

# Build the graph
workflow = StateGraph(GoalState)
workflow.add_node("decompose", decompose_goal_node)
workflow.add_node("select_next", select_next_goal_node)
workflow.add_node("execute", execute_goal_node)

workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "select_next")
workflow.add_conditional_edges(
    "select_next",
    should_continue,
    {"continue": "execute", "end": END}
)
workflow.add_edge("execute", "select_next")

goal_manager = workflow.compile()
```

### Approach 3: Dynamic Replanning with Reflection

Incorporate reflection for continuous improvement:

```python
def monitor_and_replan_node(state: GoalState):
    """Monitor progress and replan if needed"""
    current_goal = state["goal_tree"][state["current_goal"]]

    # Check if replanning needed
    if current_goal.status == "failed":
        analysis = analyze_failure(current_goal)
        alternative_plan = create_alternative(current_goal, analysis)
        return {"goal_tree": update_tree(state["goal_tree"], alternative_plan)}

    # Check if better approach discovered
    if should_optimize(state):
        optimized_plan = optimize_remaining_goals(state["goal_tree"])
        return {"goal_tree": optimized_plan}

    return {}

workflow.add_node("monitor", monitor_and_replan_node)
workflow.add_edge("execute", "monitor")
workflow.add_edge("monitor", "select_next")
```

## Key Benefits

### 🎯 Clarity and Structure

- **Organized approach**: Complex goals broken into clear, manageable pieces
- **Visible plan**: Everyone understands what needs to happen
- **Reduced overwhelm**: Tackle one piece at a time
- **Clear ownership**: Each goal/task has defined responsibility

### 📊 Progress Visibility

- **Real-time tracking**: Always know where you are
- **Bottleneck detection**: Identify what's slowing progress
- **Completion estimates**: Predict when goals will finish
- **Stakeholder communication**: Easy status reporting

### 🔄 Adaptive Execution

- **Handle failures**: Automatically replan when obstacles arise
- **Optimize dynamically**: Adjust strategy based on learnings
- **Respond to changes**: Adapt when requirements or conditions shift
- **Learn from experience**: Improve approach over time

### ⚡ Efficiency Gains

- **Parallel execution**: Do independent work simultaneously
- **Resource optimization**: Allocate resources where most valuable
- **Avoid redundancy**: Don't repeat completed work
- **Critical path focus**: Prioritize work that unblocks others

### 🤝 Coordination

- **Multi-agent support**: Multiple agents working toward shared goals
- **Dependency respect**: Ensure prerequisites before starting work
- **Conflict resolution**: Manage competing priorities
- **Synchronization**: Coordinate concurrent activities

## Trade-offs

### ⚠️ Planning Overhead

**Issue**: Initial decomposition and planning takes time

**Impact**: Simple tasks become slower with goal management

**Mitigation**:
- Quick assessment: Is goal complex enough to warrant decomposition?
- Templated plans: Reuse structures for similar goal types
- Lazy decomposition: Break down only as needed
- Simple goals: Use direct execution for straightforward tasks

### 🧠 Complexity Burden

**Issue**: Managing goal trees, dependencies, and state is complex

**Impact**: More code, more potential bugs, harder to debug

**Mitigation**:
- Use proven libraries: LangGraph, TaskWeaver, AutoGen
- Start simple: Begin with basic hierarchy, add features incrementally
- Good abstractions: Hide complexity behind clean interfaces
- Comprehensive testing: Test goal management logic thoroughly

### 💰 Resource Usage

**Issue**: Tracking and monitoring consumes compute, memory, and API calls

**Impact**: Higher costs, especially for many small goals

**Mitigation**:
- Efficient data structures: Optimize goal storage and lookups
- Batch operations: Group similar goals for efficiency
- Smart monitoring: Don't check status unnecessarily often
- Prune completed: Archive finished goals to reduce active set

### 📏 Granularity Challenges

**Issue**: Too fine-grained → overhead, too coarse → loses benefits

**Impact**: Suboptimal decomposition reduces effectiveness

**Mitigation**:
- Rules of thumb: Goals should be 10-60 minute tasks
- Adaptive granularity: Decompose further if goal takes too long
- Context-aware: Adjust based on domain and requirements
- Human-in-loop: Allow manual refinement of decomposition

## Best Practices

### 1. Smart Decomposition

```python
def decompose_goal(goal: str) -> List[Goal]:
    """
    Good decomposition characteristics:
    - Each subgoal is independently achievable
    - Subgoals are testable/verifiable
    - Clear success criteria
    - Reasonable granularity (not too fine, not too coarse)
    """

    prompt = f"""Decompose this goal into subgoals:
    Goal: {goal}

    Requirements:
    1. Each subgoal should be completable in 10-60 minutes
    2. Clear success criteria for each
    3. Identify dependencies between subgoals
    4. Aim for 3-7 main subgoals
    5. Each subgoal may have 2-5 tasks

    Return structured hierarchy with dependencies."""

    return llm_analyze(prompt)
```

### 2. Dependency Management

```python
def build_dependency_graph(goals: List[Goal]) -> dict:
    """
    Best practices:
    - Minimize dependencies (reduces critical path)
    - Identify truly parallel work
    - Avoid circular dependencies
    - Make implicit dependencies explicit
    """

    graph = {}
    for goal in goals:
        # Explicit dependencies from goal definition
        graph[goal.id] = set(goal.dependencies)

        # Implicit dependencies (e.g., data requirements)
        implicit_deps = detect_implicit_dependencies(goal, goals)
        graph[goal.id].update(implicit_deps)

    # Validate no cycles
    if has_cycle(graph):
        raise ValueError("Circular dependencies detected")

    return graph
```

### 3. Progress Monitoring

```python
def monitor_progress(goal_tree: dict) -> ProgressReport:
    """
    Effective monitoring includes:
    - Overall completion percentage
    - Per-goal status
    - Blocked goals and reasons
    - Critical path analysis
    - Estimated time remaining
    """

    report = ProgressReport()

    # Calculate completion
    total_goals = len(goal_tree)
    completed = sum(1 for g in goal_tree.values() if g.completed)
    report.completion_pct = completed / total_goals

    # Identify blockers
    report.blocked_goals = [
        g for g in goal_tree.values()
        if g.status == "blocked"
    ]

    # Critical path
    report.critical_path = find_critical_path(goal_tree)

    # Time estimates
    report.estimated_completion = estimate_completion_time(goal_tree)

    return report
```

### 4. Failure Handling

```python
def handle_goal_failure(goal: Goal, error: Exception) -> Goal:
    """
    Effective failure handling:
    - Analyze root cause
    - Determine if recoverable
    - Create alternative approach
    - Preserve partial progress
    - Update dependencies
    """

    analysis = analyze_failure(goal, error)

    if analysis.is_recoverable:
        # Try alternative approach
        alternative = create_alternative_plan(goal, analysis)
        return alternative
    else:
        # Mark failed, update dependents
        goal.status = "failed"
        propagate_failure(goal)

        # Suggest manual intervention
        notify_human(f"Goal {goal.title} failed: {analysis.reason}")

        return goal
```

### 5. Optimization Strategies

```python
def optimize_execution(goal_tree: dict) -> dict:
    """
    Optimization techniques:
    - Identify parallelizable goals
    - Reorder for efficiency
    - Batch similar operations
    - Cache reusable results
    - Prune unnecessary goals
    """

    # Find independent goals for parallel execution
    parallel_groups = find_parallel_goals(goal_tree)

    # Reorder to minimize blocking
    optimized_order = topological_sort_with_priority(goal_tree)

    # Identify cacheable operations
    cache_opportunities = find_cacheable_goals(goal_tree)

    # Remove redundant goals
    deduplicated = remove_duplicate_goals(goal_tree)

    return apply_optimizations(deduplicated, parallel_groups, optimized_order)
```

## Performance Metrics

Track these metrics to evaluate Goal Management effectiveness:

### Completion Metrics
- **Goal completion rate**: % of goals successfully completed
- **Time to completion**: Actual vs. estimated time for each goal
- **Overall project duration**: Start to finish time
- **Success rate by goal type**: Which types succeed/fail most

### Efficiency Metrics
- **Parallel execution ratio**: % of time multiple goals ran simultaneously
- **Resource utilization**: How efficiently resources were used
- **Blocking time**: Time goals spent waiting for dependencies
- **Replanning frequency**: How often plans needed adjustment

### Quality Metrics
- **Goal quality scores**: How well completed goals met criteria
- **Accuracy of decomposition**: How well initial plan matched execution
- **Dependency prediction**: How accurate were dependency estimates
- **Adaptation effectiveness**: How well replanning recovered from failures

### Cost Metrics
- **API calls per goal**: Efficiency of execution
- **Tokens used**: Language model costs
- **Compute time**: Processing resource usage
- **Human intervention**: How often manual help needed

## Example Scenarios

### Scenario 1: Research Report Project

```
Main Goal: Write comprehensive AI safety research report

Initial Decomposition:
├─ Define Research Scope
│  ├─ Identify key AI safety areas (alignment, robustness, interpretability)
│  ├─ Define target audience (researchers, policymakers, practitioners)
│  └─ Set report structure and length requirements
├─ Literature Review (depends on: Scope)
│  ├─ Search academic databases (parallel)
│  ├─ Review recent papers (parallel)
│  ├─ Identify key researchers and institutions (parallel)
│  └─ Synthesize findings
├─ Data Collection (depends on: Scope)
│  ├─ Gather incident reports (parallel)
│  ├─ Collect industry statistics (parallel)
│  ├─ Survey expert opinions (parallel)
│  └─ Organize data
├─ Analysis (depends on: Literature Review, Data Collection)
│  ├─ Identify trends and patterns
│  ├─ Compare approaches across organizations
│  ├─ Evaluate effectiveness of interventions
│  └─ Develop insights and recommendations
└─ Report Writing (depends on: Analysis)
   ├─ Draft executive summary
   ├─ Write main sections
   ├─ Create visualizations
   ├─ Review and refine
   └─ Format and finalize

Execution:
[Phase 1] Define scope: COMPLETED (15 min)
[Phase 2] Parallel execution:
  - Literature search: IN PROGRESS
  - Data collection: IN PROGRESS
[Phase 2] Literature review: BLOCKED (waiting for search)
[Phase 2] Data organization: COMPLETED

Status: 40% complete, 2 parallel tasks running, 0 blockers
```

### Scenario 2: Software Feature Development

```
Main Goal: Implement user authentication system

Decomposition:
├─ Requirements & Design
│  ├─ Define authentication methods (email/password, OAuth, 2FA)
│  ├─ Design database schema
│  ├─ Plan API endpoints
│  └─ Security requirements analysis
├─ Backend Implementation (depends on: Design)
│  ├─ Set up database models
│  ├─ Implement password hashing and validation
│  ├─ Create JWT token system
│  ├─ Build authentication endpoints (login, logout, refresh)
│  └─ Add OAuth integration
├─ Frontend Implementation (depends on: Design)
│  ├─ Design login/signup UI
│  ├─ Implement form validation
│  ├─ Handle token storage and refresh
│  └─ Add OAuth buttons
├─ Testing (depends on: Backend, Frontend)
│  ├─ Unit tests for auth logic
│  ├─ Integration tests for API
│  ├─ E2E tests for user flows
│  └─ Security testing (penetration, vulnerability scan)
└─ Deployment (depends on: Testing)
   ├─ Configure production environment
   ├─ Set up monitoring and logging
   ├─ Deploy to staging
   ├─ Validation testing
   └─ Production deployment

Mid-Execution Update:
[Requirements] COMPLETED
[Backend - JWT] COMPLETED
[Backend - OAuth] FAILED (API key issue)
  → Replanning: Use alternative OAuth provider
  → New subtasks: Setup new provider, migrate config
[Frontend - UI] IN PROGRESS
[Frontend - Validation] PENDING (unblocked, can start)

Decision: Replan OAuth integration, start frontend validation in parallel
```

### Scenario 3: Event Planning

```
Main Goal: Organize company annual conference (250 attendees)

Goal Hierarchy:
├─ Planning Phase
│  ├─ Set date and duration
│  ├─ Define budget
│  ├─ Create attendee list
│  └─ Outline agenda and themes
├─ Venue & Logistics (depends on: Planning)
│  ├─ Research and book venue
│  ├─ Arrange catering
│  ├─ Plan AV and tech setup
│  ├─ Book accommodations
│  └─ Organize transportation
├─ Content & Speakers (depends on: Planning)
│  ├─ Identify speaker candidates
│  ├─ Confirm keynote speakers
│  ├─ Schedule breakout sessions
│  ├─ Prepare presentation materials
│  └─ Arrange panel discussions
├─ Marketing & Registration (depends on: Planning)
│  ├─ Design event website
│  ├─ Create promotional materials
│  ├─ Launch registration system
│  ├─ Email campaigns
│  └─ Social media promotion
└─ Execution & Follow-up (depends on: All above)
   ├─ On-site coordination
   ├─ Manage registration desk
   ├─ Technical support during event
   ├─ Collect feedback
   └─ Post-event report

Progress Dashboard:
Overall: 65% complete
├─ Planning: 100% ✓
├─ Venue: 90% (catering pending confirmation)
├─ Speakers: 75% (2 speakers yet to confirm)
├─ Marketing: 80% (registrations at 85% capacity)
└─ Execution: 0% (scheduled for 2 weeks)

Risks Identified:
- 2 speakers unconfirmed (backup options prepared)
- Catering capacity at limit (monitoring registrations)

Next Actions:
1. Follow up with pending speakers [HIGH PRIORITY]
2. Finalize catering numbers [MEDIUM]
3. Complete social media campaign [LOW]
```

## Advanced Patterns

### 1. Goal Learning and Optimization

```python
class LearningGoalManager:
    def __init__(self):
        self.goal_history = []
        self.success_patterns = {}

    def learn_from_execution(self, goal: Goal):
        """Learn which strategies work for goal types"""
        self.goal_history.append({
            'type': goal.type,
            'strategy': goal.execution_strategy,
            'success': goal.status == 'completed',
            'duration': goal.actual_duration,
            'efficiency': goal.estimated_duration / goal.actual_duration
        })

        # Identify successful patterns
        if goal.status == 'completed':
            goal_type = goal.type
            if goal_type not in self.success_patterns:
                self.success_patterns[goal_type] = []
            self.success_patterns[goal_type].append(goal.execution_strategy)

    def recommend_strategy(self, new_goal: Goal) -> str:
        """Recommend execution strategy based on past successes"""
        similar_goals = [g for g in self.goal_history
                        if g['type'] == new_goal.type and g['success']]

        if not similar_goals:
            return "default_strategy"

        # Return most successful strategy
        best = max(similar_goals, key=lambda g: g['efficiency'])
        return best['strategy']
```

### 2. Dynamic Priority Adjustment

```python
def adjust_priorities(goal_tree: dict, context: dict) -> dict:
    """Dynamically adjust priorities based on context changes"""

    for goal in goal_tree.values():
        # Boost priority if blocking many other goals
        blocked_count = count_goals_blocked_by(goal, goal_tree)
        if blocked_count > 3:
            goal.priority += 2

        # Increase priority as deadline approaches
        if goal.deadline:
            days_remaining = (goal.deadline - datetime.now()).days
            if days_remaining < 7:
                goal.priority += 3

        # Adjust based on resource availability
        if goal.required_resources in context['available_resources']:
            goal.priority += 1

        # Lower priority if dependencies far from completion
        if not dependencies_near_complete(goal, goal_tree):
            goal.priority -= 1

    return goal_tree
```

### 3. Constraint Satisfaction

```python
class ConstraintAwareGoalManager:
    def __init__(self, constraints: dict):
        self.constraints = constraints

    def validate_goal_plan(self, goal_tree: dict) -> bool:
        """Ensure plan satisfies all constraints"""

        # Budget constraint
        total_cost = sum(g.estimated_cost for g in goal_tree.values())
        if total_cost > self.constraints['max_budget']:
            return False

        # Time constraint
        critical_path_duration = calculate_critical_path(goal_tree)
        if critical_path_duration > self.constraints['max_duration']:
            return False

        # Resource constraint
        peak_resources = calculate_peak_resource_usage(goal_tree)
        if peak_resources > self.constraints['max_resources']:
            return False

        return True

    def optimize_under_constraints(self, goal_tree: dict) -> dict:
        """Adjust plan to meet constraints"""

        while not self.validate_goal_plan(goal_tree):
            # Try various optimization strategies
            goal_tree = reduce_scope(goal_tree)
            goal_tree = increase_parallelism(goal_tree)
            goal_tree = optimize_resource_allocation(goal_tree)

        return goal_tree
```

### 4. Multi-Agent Goal Coordination

```python
class MultiAgentGoalCoordinator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.goal_assignments = {}

    def assign_goals(self, goal_tree: dict) -> dict:
        """Assign goals to agents based on capabilities and availability"""

        ready_goals = get_ready_goals(goal_tree)

        for goal in ready_goals:
            # Find agent best suited for this goal
            best_agent = self.select_agent_for_goal(goal)

            if best_agent and best_agent.is_available():
                self.goal_assignments[goal.id] = best_agent
                best_agent.assign_goal(goal)

        return self.goal_assignments

    def select_agent_for_goal(self, goal: Goal) -> Optional[Agent]:
        """Select best agent based on skills and current load"""

        suitable_agents = [a for a in self.agents
                          if goal.required_skills.issubset(a.skills)]

        if not suitable_agents:
            return None

        # Choose agent with lowest current workload
        return min(suitable_agents, key=lambda a: a.current_workload())

    def synchronize_agents(self):
        """Ensure agents coordinate on shared dependencies"""

        for goal_id, agent in self.goal_assignments.items():
            goal = self.get_goal(goal_id)

            # Share results of completed goals with dependent agents
            if goal.status == 'completed':
                dependent_goals = get_dependent_goals(goal)
                for dep_goal in dependent_goals:
                    assigned_agent = self.goal_assignments.get(dep_goal.id)
                    if assigned_agent:
                        assigned_agent.receive_dependency_result(goal)
```

## Comparison with Related Patterns

| Pattern | Planning | Execution | Adaptation | Use Case |
|---------|----------|-----------|------------|----------|
| **Goal Management** | Hierarchical decomposition | Parallel + Sequential | Dynamic replanning | Complex multi-phase projects |
| **Planning** | Upfront detailed plan | Follows plan | Limited | Predictable workflows |
| **Multi-Agent** | Distributed | Parallel collaboration | Coordination | Multiple specialized agents |
| **Prioritization** | Priority-based queue | Sequential by priority | Re-prioritization | Task triage and ordering |
| **ReAct** | Per-step reasoning | Sequential exploration | Per-step adaptation | Exploratory problem-solving |

**Goal Management** vs **Planning**:
- Planning creates detailed upfront plan, Goal Management adapts continuously
- Goal Management handles uncertainty better
- Planning is simpler for predictable workflows

**Goal Management** vs **Multi-Agent**:
- Multi-Agent focuses on agent coordination, Goal Management on goal structure
- Can be combined: use Goal Management to organize work, Multi-Agent to execute
- Goal Management works with single or multiple agents

**Goal Management** vs **Prioritization**:
- Prioritization orders tasks, Goal Management structures and tracks them
- Goal Management includes dependency handling, Prioritization doesn't
- Prioritization is simpler for flat task lists

## Common Pitfalls

### 1. Over-Decomposition

**Problem**: Breaking goals into too many tiny pieces creates overhead

**Example**:
```
Bad: "Type the letter 'H'", "Type the letter 'e'", ...
Good: "Write introduction paragraph"
```

**Solution**:
- Goals should be 10-60 minute tasks
- Stop decomposing when further breakdown adds no value
- Balance granularity with overhead

### 2. Missing Dependencies

**Problem**: Not identifying prerequisites leads to failures and wasted work

**Example**:
```
Problem: Start "Write analysis" before "Collect data" completes
Result: Analysis based on incomplete data, needs redoing
```

**Solution**:
- Explicit dependency declaration
- Automated dependency detection where possible
- Review dependency graph before execution
- Test with dry-run

### 3. Ignoring Progress Signals

**Problem**: Not monitoring leads to late detection of issues

**Example**:
```
Unnoticed: Subgoal stuck for 2 hours
Result: Entire project delayed, could have replanned earlier
```

**Solution**:
- Regular progress checks
- Timeout detection
- Proactive blocker identification
- Real-time dashboards

### 4. Rigid Planning

**Problem**: Treating initial plan as unchangeable despite new information

**Example**:
```
Situation: Better approach discovered mid-execution
Rigid response: Continue with inferior plan
Adaptive response: Replan to use better approach
```

**Solution**:
- Embrace replanning as normal
- Continuous reflection on effectiveness
- Encourage strategy updates
- Balance stability with adaptability

### 5. Poor Goal Granularity

**Problem**: Goals too large (untrackable) or too small (overhead)

**Solution**:
```python
def assess_goal_granularity(goal: Goal) -> str:
    """Check if goal is appropriately sized"""

    estimated_minutes = goal.estimated_effort

    if estimated_minutes < 5:
        return "too_small"  # Consider merging
    elif estimated_minutes > 120:
        return "too_large"  # Consider decomposing
    else:
        return "appropriate"
```

## Conclusion

The Goal Management pattern provides a powerful framework for orchestrating complex, multi-step objectives through hierarchical decomposition, dependency tracking, progress monitoring, and dynamic adaptation. It transforms overwhelming projects into manageable, trackable, and adaptable execution plans.

**Use Goal Management when:**
- Complex objectives with multiple interdependent activities
- Progress visibility and tracking are valuable
- Adaptation and replanning may be necessary
- Opportunities for parallel execution exist
- Coordination across multiple agents or phases required
- Project spans considerable time or multiple sessions

**Implementation checklist:**
- ✅ Define clear goal hierarchy with proper granularity
- ✅ Identify and map all dependencies accurately
- ✅ Implement progress tracking and monitoring
- ✅ Set up parallel execution for independent goals
- ✅ Build replanning capability for failures
- ✅ Create visualization for goal status and progress
- ✅ Establish priority scoring and dynamic adjustment
- ✅ Track metrics for continuous improvement
- ✅ Handle constraint satisfaction (budget, time, resources)
- ✅ Test with various goal types and complexities

**Key Takeaways:**
- 🎯 Decompose complex goals into hierarchical structures
- 📊 Track progress continuously for visibility
- 🔄 Adapt plans dynamically based on execution realities
- ⚡ Execute independent goals in parallel for efficiency
- 🤝 Coordinate multiple agents toward shared objectives
- 🧠 Learn from execution to improve future planning
- 🎨 Balance structure with flexibility
- 📈 Measure effectiveness with completion rates and efficiency metrics

---

*Goal Management transforms complex, overwhelming objectives into structured, trackable, and adaptable execution plans—enabling agents to tackle long-running projects with clarity, efficiency, and resilience.*

## Corporate SSL proxy note

If you're behind a corporate SSL-inspecting proxy, run examples with:

```bash
AGENTIC_DISABLE_SSL=1 bash run.sh
```

