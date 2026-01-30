"""
Goal Management Pattern: Advanced Implementation

This example demonstrates advanced goal management with:
- Complex goal graphs with dependencies
- Parallel execution where possible
- Dynamic priority adjustment
- Continuous monitoring and replanning
- Resource allocation tracking
- Detailed progress dashboard

Problem: Complex project with uncertainties (e.g., software development, business launch)
Solution: Adaptive goal management with parallel execution and continuous optimization
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=False)


# --- Advanced Goal Data Structures ---

class GoalStatus(Enum):
    """Status of a goal in the execution lifecycle"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class GoalType(Enum):
    """Type of goal for categorization"""
    MAIN = "main"
    SUBGOAL = "subgoal"
    TASK = "task"


@dataclass
class Goal:
    """Advanced goal with additional tracking and optimization features"""
    id: str
    title: str
    description: str
    goal_type: GoalType = GoalType.TASK
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    children: List['Goal'] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)  # Must complete before this
    enables: Set[str] = field(default_factory=set)  # This unlocks these goals
    priority: float = 5.0  # Dynamic priority (1-10)
    base_priority: int = 5  # Original priority
    result: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_minutes: int = 30
    actual_minutes: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 2
    parallel_group: Optional[str] = None  # Goals with same group can run in parallel
    resource_requirements: Set[str] = field(default_factory=set)
    success_criteria: str = ""

    def is_ready(self, completed_goals: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return self.dependencies.issubset(completed_goals)

    def is_leaf(self) -> bool:
        """Check if this is a leaf task (no children)"""
        return len(self.children) == 0

    def progress(self) -> float:
        """Calculate completion percentage (0.0 to 1.0)"""
        if not self.children:
            return 1.0 if self.status == GoalStatus.COMPLETED else 0.0

        if not self.children:
            return 0.0

        completed = sum(1 for child in self.children if child.status == GoalStatus.COMPLETED)
        return completed / len(self.children)

    def duration_minutes(self) -> Optional[int]:
        """Calculate actual duration in minutes"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() / 60)
        return None


@dataclass
class ResourceUsage:
    """Track resource usage across goal execution"""
    api_calls: int = 0
    total_tokens: int = 0
    execution_time_seconds: float = 0.0
    parallel_tasks_peak: int = 0


@dataclass
class AdvancedGoalState:
    """State for advanced goal management workflow"""
    main_goal: str
    goal_tree: Optional[Goal] = None
    completed_goals: Set[str] = field(default_factory=set)
    failed_goals: Set[str] = field(default_factory=set)
    in_progress_goals: Set[str] = field(default_factory=set)
    all_goals: Dict[str, Goal] = field(default_factory=dict)
    parallel_groups: Dict[str, List[str]] = field(default_factory=dict)
    status: str = "initializing"
    cycle_count: int = 0
    max_cycles: int = 50
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    final_report: str = ""


# --- Advanced Goal Management Functions ---

def decompose_goal_advanced(state: AdvancedGoalState) -> AdvancedGoalState:
    """Decompose goal with parallel group identification"""
    print("\n" + "="*80)
    print("PHASE 1: ADVANCED GOAL DECOMPOSITION")
    print("="*80)
    print(f"\nMain Goal: {state.main_goal}\n")

    decomposition_prompt = f"""You are an expert project planner. Create a detailed breakdown of this goal:

Main Goal: {state.main_goal}

Create a structured breakdown with:
1. 3-5 main subgoals (major phases)
2. For each subgoal, 2-4 specific tasks
3. Identify dependencies clearly
4. Identify which tasks can run in PARALLEL (mark with same group ID)
5. Estimate time for each task (in minutes)
6. Priority (1-10, higher = more important)

Format:
SUBGOAL: [Title]
DESCRIPTION: [What to accomplish]
PRIORITY: [1-10]
TASKS:
- TASK: [Title] | DESC: [Details] | DEPENDS: [goal IDs or "none"] | TIME: [minutes] | PRIORITY: [1-10] | PARALLEL_GROUP: [group_id or "sequential"]

Example:
SUBGOAL: Requirements Analysis
DESCRIPTION: Define all system requirements
PRIORITY: 10
TASKS:
- TASK: User research | DESC: Interview stakeholders | DEPENDS: none | TIME: 45 | PRIORITY: 9 | PARALLEL_GROUP: research
- TASK: Competitor analysis | DESC: Study existing solutions | DEPENDS: none | TIME: 30 | PRIORITY: 8 | PARALLEL_GROUP: research
- TASK: Requirements documentation | DESC: Write formal requirements | DEPENDS: sg1_t1,sg1_t2 | TIME: 60 | PRIORITY: 9 | PARALLEL_GROUP: sequential

Be specific and identify all parallelization opportunities."""

    response = llm.invoke([
        SystemMessage(content="You are a strategic planning assistant."),
        HumanMessage(content=decomposition_prompt)
    ])

    state.resource_usage.api_calls += 1

    # Parse into goal graph
    goal_tree, parallel_groups = parse_advanced_goal_hierarchy(response.content, state.main_goal)

    print("\n📋 Advanced Goal Graph Created:")
    print_goal_tree_advanced(goal_tree)

    print("\n⚡ Parallel Execution Groups Identified:")
    for group_id, task_ids in parallel_groups.items():
        if group_id != "sequential" and len(task_ids) > 1:
            print(f"   Group '{group_id}': {len(task_ids)} tasks can run in parallel")

    return {
        "goal_tree": goal_tree,
        "all_goals": collect_all_goals_advanced(goal_tree),
        "parallel_groups": parallel_groups,
        "status": "planning_complete",
        "resource_usage": state.resource_usage
    }


def parse_advanced_goal_hierarchy(content: str, main_goal: str) -> tuple[Goal, Dict[str, List[str]]]:
    """Parse LLM response into advanced Goal hierarchy with parallel groups"""
    root = Goal(
        id="0",
        title=main_goal,
        description="Main objective",
        goal_type=GoalType.MAIN,
        priority=10.0,
        base_priority=10
    )

    lines = content.strip().split('\n')
    current_subgoal = None
    subgoal_counter = 1
    task_counter = 1
    parallel_groups = {"sequential": []}

    for line in lines:
        line = line.strip()

        if line.startswith("SUBGOAL:"):
            title = line.replace("SUBGOAL:", "").strip()
            current_subgoal = Goal(
                id=f"sg{subgoal_counter}",
                title=title,
                description="",
                goal_type=GoalType.SUBGOAL,
                parent_id="0",
                priority=8.0,
                base_priority=8
            )
            root.children.append(current_subgoal)
            subgoal_counter += 1

        elif line.startswith("DESCRIPTION:") and current_subgoal:
            current_subgoal.description = line.replace("DESCRIPTION:", "").strip()

        elif line.startswith("PRIORITY:") and current_subgoal:
            try:
                priority = int(line.replace("PRIORITY:", "").strip())
                current_subgoal.priority = float(priority)
                current_subgoal.base_priority = priority
            except ValueError:
                pass

        elif line.startswith("- TASK:") and current_subgoal:
            parts = line.replace("- TASK:", "").split("|")
            task_title = parts[0].strip()
            task_desc = ""
            dependencies = set()
            time_est = 30
            priority = 5
            parallel_group = "sequential"

            for part in parts[1:]:
                part = part.strip()
                if part.startswith("DESC:"):
                    task_desc = part.replace("DESC:", "").strip()
                elif part.startswith("DEPENDS:"):
                    dep_str = part.replace("DEPENDS:", "").strip()
                    if dep_str.lower() not in ["none", "n/a", ""]:
                        dependencies = {current_subgoal.id}
                elif part.startswith("TIME:"):
                    try:
                        time_est = int(part.replace("TIME:", "").strip())
                    except ValueError:
                        pass
                elif part.startswith("PRIORITY:"):
                    try:
                        priority = int(part.replace("PRIORITY:", "").strip())
                    except ValueError:
                        pass
                elif part.startswith("PARALLEL_GROUP:"):
                    parallel_group = part.replace("PARALLEL_GROUP:", "").strip()

            task = Goal(
                id=f"t{task_counter}",
                title=task_title,
                description=task_desc,
                goal_type=GoalType.TASK,
                parent_id=current_subgoal.id,
                dependencies=dependencies,
                priority=float(priority),
                base_priority=priority,
                estimated_minutes=time_est,
                parallel_group=parallel_group
            )
            current_subgoal.children.append(task)

            # Track parallel groups
            if parallel_group not in parallel_groups:
                parallel_groups[parallel_group] = []
            parallel_groups[parallel_group].append(task.id)

            task_counter += 1

    return root, parallel_groups


def collect_all_goals_advanced(root: Goal) -> Dict[str, Goal]:
    """Collect all goals into flat dictionary"""
    goals = {root.id: root}

    def collect_recursive(goal: Goal):
        for child in goal.children:
            goals[child.id] = child
            collect_recursive(child)

    collect_recursive(root)
    return goals


def adjust_priorities(state: AdvancedGoalState) -> AdvancedGoalState:
    """Dynamically adjust priorities based on context"""
    print("\n🎯 Adjusting Priorities...")

    for goal_id, goal in state.all_goals.items():
        if goal.status != GoalStatus.PENDING:
            continue

        # Start with base priority
        adjusted_priority = float(goal.base_priority)

        # Boost if blocking many other goals
        enabled_count = len(goal.enables)
        if enabled_count > 0:
            adjusted_priority += enabled_count * 0.5

        # Boost if part of critical path
        if is_on_critical_path(goal, state.all_goals):
            adjusted_priority += 2.0

        # Reduce if dependencies far from complete
        if not are_dependencies_near_complete(goal, state.all_goals, state.completed_goals):
            adjusted_priority -= 1.0

        goal.priority = max(1.0, min(10.0, adjusted_priority))

    return {"all_goals": state.all_goals}


def is_on_critical_path(goal: Goal, all_goals: Dict[str, Goal]) -> bool:
    """Check if goal is on critical path (longest path to completion)"""
    # Simplified: check if goal has many dependents
    return len(goal.enables) >= 2


def are_dependencies_near_complete(goal: Goal, all_goals: Dict[str, Goal], completed: Set[str]) -> bool:
    """Check if dependencies are close to completion"""
    if not goal.dependencies:
        return True

    for dep_id in goal.dependencies:
        if dep_id not in completed:
            dep_goal = all_goals.get(dep_id)
            if dep_goal and dep_goal.progress() < 0.5:
                return False

    return True


def select_parallel_goals(state: AdvancedGoalState) -> AdvancedGoalState:
    """Select multiple goals for parallel execution"""
    if state.status == "complete":
        return {"status": "complete"}

    # Get all leaf goals ready to execute
    ready_goals = []

    for goal_id, goal in state.all_goals.items():
        if (goal.is_leaf() and
            goal.status == GoalStatus.PENDING and
            goal.is_ready(state.completed_goals)):
            ready_goals.append(goal)

    if not ready_goals:
        # Check if complete
        all_complete = all(
            g.status == GoalStatus.COMPLETED or not g.is_leaf()
            for g in state.all_goals.values()
        )
        if all_complete:
            return {"status": "complete"}

        # Check for blocked state
        if state.in_progress_goals:
            return {"status": "waiting"}  # Wait for in-progress goals
        else:
            print("\n⚠️  No ready goals and none in progress. Checking for issues...")
            return {"status": "blocked"}

    # Sort by priority
    ready_goals.sort(key=lambda g: g.priority, reverse=True)

    # Select goals for parallel execution (up to 3 at once)
    max_parallel = 3
    selected_goals = ready_goals[:min(max_parallel, len(ready_goals))]

    # Update resource tracking
    peak_parallel = len(selected_goals)
    if peak_parallel > state.resource_usage.parallel_tasks_peak:
        state.resource_usage.parallel_tasks_peak = peak_parallel

    print(f"\n▶️  Selected {len(selected_goals)} goals for parallel execution:")
    for goal in selected_goals:
        print(f"   • {goal.title} (Priority: {goal.priority:.1f})")
        goal.status = GoalStatus.IN_PROGRESS
        goal.start_time = datetime.now()

    return {
        "in_progress_goals": state.in_progress_goals | {g.id for g in selected_goals},
        "status": "executing_parallel",
        "resource_usage": state.resource_usage
    }


def execute_parallel_goals(state: AdvancedGoalState) -> AdvancedGoalState:
    """Execute multiple goals in parallel using thread pool"""
    in_progress_ids = list(state.in_progress_goals)
    in_progress_goals = [state.all_goals[gid] for gid in in_progress_ids]

    print(f"\n⚙️  Executing {len(in_progress_goals)} goals in parallel...")

    completed_in_cycle = set()
    failed_in_cycle = set()

    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(in_progress_goals)) as executor:
        future_to_goal = {
            executor.submit(execute_single_goal, goal, state.main_goal): goal
            for goal in in_progress_goals
        }

        for future in as_completed(future_to_goal):
            goal = future_to_goal[future]
            try:
                result = future.result()
                if result["success"]:
                    goal.result = result["result"]
                    goal.status = GoalStatus.COMPLETED
                    goal.end_time = datetime.now()
                    goal.actual_minutes = goal.duration_minutes()
                    completed_in_cycle.add(goal.id)
                    print(f"   ✓ {goal.title} - Completed")

                    # Update parent status
                    update_parent_status_advanced(goal, state.all_goals)
                else:
                    goal.status = GoalStatus.FAILED
                    goal.retry_count += 1
                    failed_in_cycle.add(goal.id)
                    print(f"   ✗ {goal.title} - Failed")

            except Exception as e:
                print(f"   ✗ {goal.title} - Error: {str(e)}")
                goal.status = GoalStatus.FAILED
                goal.retry_count += 1
                failed_in_cycle.add(goal.id)

    state.resource_usage.api_calls += len(in_progress_goals)

    return {
        "completed_goals": state.completed_goals | completed_in_cycle,
        "failed_goals": state.failed_goals | failed_in_cycle,
        "in_progress_goals": set(),
        "status": "cycle_complete",
        "resource_usage": state.resource_usage
    }


def execute_single_goal(goal: Goal, main_goal_context: str) -> Dict:
    """Execute a single goal (used in parallel execution)"""
    execution_prompt = f"""Execute this task and provide the result:

Task: {goal.title}
Description: {goal.description}
Context: Part of larger goal: {main_goal_context}
Success Criteria: {goal.success_criteria or 'Complete the task effectively'}

Provide a concrete, specific result. Be realistic and detailed.
Focus on what was accomplished and any key findings or outputs."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a task execution assistant."),
            HumanMessage(content=execution_prompt)
        ])

        return {
            "success": True,
            "result": response.content
        }
    except Exception as e:
        return {
            "success": False,
            "result": str(e)
        }


def update_parent_status_advanced(goal: Goal, all_goals: Dict[str, Goal]):
    """Update parent goal status if all children complete"""
    if not goal.parent_id:
        return

    parent = all_goals[goal.parent_id]
    if all(child.status == GoalStatus.COMPLETED for child in parent.children):
        parent.status = GoalStatus.COMPLETED
        update_parent_status_advanced(parent, all_goals)


def replan_failed_goals(state: AdvancedGoalState) -> AdvancedGoalState:
    """Replan all failed goals with alternative approaches"""
    failed_goal_ids = [gid for gid in state.failed_goals if gid not in state.completed_goals]

    if not failed_goal_ids:
        return {"status": "cycle_complete"}

    print(f"\n🔄 Replanning {len(failed_goal_ids)} failed goals...")

    for goal_id in failed_goal_ids:
        goal = state.all_goals[goal_id]

        if goal.retry_count >= goal.max_retries:
            print(f"   ✗ {goal.title} - Max retries reached, marking as permanently failed")
            continue

        print(f"   🔄 Creating alternative approach for: {goal.title}")

        replan_prompt = f"""A goal failed. Create an alternative, simpler approach.

Failed Goal: {goal.title}
Description: {goal.description}
Attempt: {goal.retry_count + 1} of {goal.max_retries}

Provide:
1. Brief analysis of potential failure reason
2. Alternative, simpler approach
3. Specific actionable steps

Be concise and practical."""

        response = llm.invoke([
            SystemMessage(content="You are a problem-solving assistant."),
            HumanMessage(content=replan_prompt)
        ])

        # Create alternative task
        new_task = Goal(
            id=f"{goal.id}_v{goal.retry_count + 1}",
            title=f"{goal.title} (Alternative)",
            description=response.content,
            goal_type=goal.goal_type,
            parent_id=goal.parent_id,
            dependencies=goal.dependencies.copy(),
            priority=goal.priority + 1.0,
            base_priority=goal.base_priority,
            estimated_minutes=goal.estimated_minutes,
            status=GoalStatus.PENDING
        )

        # Add to parent and all_goals
        if goal.parent_id:
            parent = state.all_goals[goal.parent_id]
            parent.children.append(new_task)

        state.all_goals[new_task.id] = new_task
        state.resource_usage.api_calls += 1

    # Clear failed set (they've been replanned)
    return {
        "failed_goals": set(),
        "all_goals": state.all_goals,
        "status": "cycle_complete",
        "resource_usage": state.resource_usage
    }


def monitor_and_continue(state: AdvancedGoalState) -> AdvancedGoalState:
    """Monitor progress and decide next action"""
    state.cycle_count += 1

    if state.cycle_count >= state.max_cycles:
        print(f"\n⚠️  Maximum cycles ({state.max_cycles}) reached")
        return {"status": "complete"}

    # Display progress
    display_progress_dashboard(state)

    # Adjust priorities for next cycle
    state = adjust_priorities(state)

    # Determine next action
    if state.failed_goals:
        return {"status": "needs_replanning"}
    else:
        return {"status": "continue"}


def display_progress_dashboard(state: AdvancedGoalState):
    """Display detailed progress dashboard"""
    root = state.goal_tree
    overall_progress = root.progress() * 100

    total_tasks = sum(1 for g in state.all_goals.values() if g.is_leaf())
    completed_tasks = len([g for g in state.all_goals.values() if g.is_leaf() and g.status == GoalStatus.COMPLETED])

    print("\n" + "="*80)
    print("📊 PROGRESS DASHBOARD")
    print("="*80)

    # Progress bar
    bar_length = 40
    filled = int(bar_length * overall_progress / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\nOverall: {overall_progress:.0f}% [{bar}]")

    print(f"\nTasks: {completed_tasks}/{total_tasks} completed")
    print(f"Cycle: {state.cycle_count}")
    print(f"API Calls: {state.resource_usage.api_calls}")
    print(f"Peak Parallel Tasks: {state.resource_usage.parallel_tasks_peak}")

    # Status breakdown
    print("\nStatus Breakdown:")
    for status in GoalStatus:
        count = sum(1 for g in state.all_goals.values() if g.is_leaf() and g.status == status)
        if count > 0:
            print(f"  {status.value.upper()}: {count}")

    print("="*80)


def generate_final_report_advanced(state: AdvancedGoalState) -> AdvancedGoalState:
    """Generate comprehensive final report"""
    print("\n" + "="*80)
    print("FINAL REPORT - ADVANCED GOAL MANAGEMENT")
    print("="*80)

    root = state.goal_tree

    # Calculate metrics
    total_time = sum(
        g.actual_minutes for g in state.all_goals.values()
        if g.actual_minutes is not None
    )

    estimated_time = sum(
        g.estimated_minutes for g in state.all_goals.values()
        if g.is_leaf()
    )

    efficiency = (estimated_time / total_time * 100) if total_time > 0 else 0

    report_parts = [
        f"\n🎯 Goal Management Summary",
        f"\nMain Goal: {root.title}",
        f"Status: {'✓ COMPLETED' if root.status == GoalStatus.COMPLETED else '⚠ INCOMPLETE'}",
        f"\n📊 Metrics:",
        f"  • Overall Progress: {root.progress() * 100:.0f}%",
        f"  • Completed Tasks: {len(state.completed_goals)}/{len([g for g in state.all_goals.values() if g.is_leaf()])}",
        f"  • Execution Cycles: {state.cycle_count}",
        f"  • Total Time: {total_time} minutes",
        f"  • Estimated Time: {estimated_time} minutes",
        f"  • Efficiency: {efficiency:.0f}%",
        f"\n⚡ Resource Usage:",
        f"  • API Calls: {state.resource_usage.api_calls}",
        f"  • Peak Parallel Tasks: {state.resource_usage.parallel_tasks_peak}",
        f"\n📋 Goal Hierarchy with Results:",
        "\n" + get_results_tree_advanced(root)
    ]

    report = "\n".join(report_parts)
    print(report)

    return {
        "final_report": report,
        "status": "finished"
    }


def get_results_tree_advanced(goal: Goal, indent: int = 0) -> str:
    """Generate detailed results tree"""
    prefix = "  " * indent

    status_icon = {
        GoalStatus.COMPLETED: "✓",
        GoalStatus.IN_PROGRESS: "⟳",
        GoalStatus.PENDING: "○",
        GoalStatus.FAILED: "✗",
        GoalStatus.BLOCKED: "⏸"
    }

    icon = status_icon.get(goal.status, "?")
    time_str = f" ({goal.actual_minutes}min)" if goal.actual_minutes else ""

    lines = [f"{prefix}{icon} {goal.title}{time_str}"]

    if goal.result and goal.is_leaf():
        result_preview = goal.result[:100] + "..." if len(goal.result) > 100 else goal.result
        lines.append(f"{prefix}  → {result_preview}")

    for child in goal.children:
        lines.append(get_results_tree_advanced(child, indent + 1))

    return "\n".join(lines)


def print_goal_tree_advanced(goal: Goal, indent: int = 0):
    """Print advanced goal tree with metadata"""
    prefix = "  " * indent
    symbol = "├─" if indent > 0 else ""

    deps_str = f", Deps: {list(goal.dependencies)}" if goal.dependencies else ""
    parallel_str = f", Group: {goal.parallel_group}" if goal.parallel_group and goal.parallel_group != "sequential" else ""

    print(f"{prefix}{symbol} {goal.title}")
    print(f"{prefix}   (ID: {goal.id}, Priority: {goal.priority:.1f}, Est: {goal.estimated_minutes}min{deps_str}{parallel_str})")

    for child in goal.children:
        print_goal_tree_advanced(child, indent + 1)


def should_continue_advanced(state: AdvancedGoalState) -> str:
    """Determine next node in workflow"""
    if state.status == "planning_complete":
        return "select"
    elif state.status == "continue":
        return "select"
    elif state.status == "executing_parallel":
        return "execute"
    elif state.status == "needs_replanning":
        return "replan"
    elif state.status == "cycle_complete":
        return "monitor"
    elif state.status == "waiting":
        return "monitor"
    elif state.status == "complete":
        return "report"
    elif state.status == "blocked":
        return "report"
    elif state.status == "finished":
        return "end"
    else:
        return "end"


# --- Build Advanced Workflow ---

def create_advanced_goal_workflow():
    """Create advanced LangGraph workflow"""
    workflow = StateGraph(AdvancedGoalState)

    # Add nodes
    workflow.add_node("decompose", decompose_goal_advanced)
    workflow.add_node("select", select_parallel_goals)
    workflow.add_node("execute", execute_parallel_goals)
    workflow.add_node("replan", replan_failed_goals)
    workflow.add_node("monitor", monitor_and_continue)
    workflow.add_node("report", generate_final_report_advanced)

    # Set entry point
    workflow.set_entry_point("decompose")

    # Add conditional edges
    workflow.add_conditional_edges(
        "decompose",
        should_continue_advanced,
        {"select": "select", "end": END}
    )

    workflow.add_conditional_edges(
        "select",
        should_continue_advanced,
        {"execute": "execute", "report": "report", "monitor": "monitor", "end": END}
    )

    workflow.add_conditional_edges(
        "execute",
        should_continue_advanced,
        {"monitor": "monitor", "replan": "replan", "end": END}
    )

    workflow.add_conditional_edges(
        "replan",
        should_continue_advanced,
        {"monitor": "monitor", "end": END}
    )

    workflow.add_conditional_edges(
        "monitor",
        should_continue_advanced,
        {"select": "select", "replan": "replan", "report": "report", "end": END}
    )

    workflow.add_conditional_edges(
        "report",
        should_continue_advanced,
        {"end": END}
    )

    return workflow.compile()


# --- Example Usage ---

def run_advanced_goal_management(main_goal: str):
    """Run advanced goal management example"""
    print(f"\n{'='*80}")
    print(f"Goal Management Pattern - Advanced Implementation")
    print(f"{'='*80}")

    app = create_advanced_goal_workflow()

    initial_state = AdvancedGoalState(
        main_goal=main_goal,
        status="initializing"
    )

    result = app.invoke(initial_state)

    return result


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║       Goal Management Pattern - Advanced Implementation        ║
    ║                                                                ║
    ║  Demonstrates parallel execution, dynamic replanning,          ║
    ║  priority adjustment, and comprehensive progress tracking      ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Example: Software Project
    example_goal = "Implement a user authentication system with OAuth, JWT, and 2FA support"

    result = run_advanced_goal_management(example_goal)

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    Example Complete!                           ║
    ║                                                                ║
    ║  The Advanced Goal Management pattern demonstrated:            ║
    ║  ✓ Complex goal graph with dependencies                        ║
    ║  ✓ Parallel execution of independent tasks                     ║
    ║  ✓ Dynamic priority adjustment                                 ║
    ║  ✓ Continuous monitoring and progress tracking                 ║
    ║  ✓ Automatic replanning on failures                            ║
    ║  ✓ Resource usage tracking and optimization                    ║
    ║  ✓ Comprehensive final report with metrics                     ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
