"""
Goal Management Pattern: Basic Implementation

This example demonstrates the Goal Management pattern with hierarchical decomposition,
dependency tracking, sequential execution, and simple replanning on failure.

Problem: Plan and execute a complex task like writing a research report or organizing an event
Solution: Break down into subgoals, track dependencies, execute sequentially, monitor progress
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=False)


# --- Goal Data Structures ---

class GoalStatus(Enum):
    """Status of a goal in the execution lifecycle"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Goal:
    """Represents a goal or subgoal in the goal hierarchy"""
    id: str
    title: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    children: List['Goal'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10 scale
    result: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_minutes: int = 30

    def is_ready(self, completed_goals: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep_id in completed_goals for dep_id in self.dependencies)

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


@dataclass
class GoalManagementState:
    """State for the goal management workflow"""
    main_goal: str
    goal_tree: Optional[Goal] = None
    completed_goals: set = field(default_factory=set)
    current_goal_id: Optional[str] = None
    all_goals: dict = field(default_factory=dict)  # id -> Goal mapping
    status: str = "initializing"
    final_report: str = ""


# --- Goal Management Functions ---

def decompose_goal(state: GoalManagementState) -> GoalManagementState:
    """Decompose the main goal into a hierarchy of subgoals and tasks"""
    print("\n" + "="*80)
    print("PHASE 1: GOAL DECOMPOSITION")
    print("="*80)
    print(f"\nMain Goal: {state.main_goal}\n")

    decomposition_prompt = f"""You are a goal planning expert. Break down this main goal into a hierarchical structure.

Main Goal: {state.main_goal}

Create a hierarchical breakdown with:
1. 3-5 main subgoals (major phases)
2. For each subgoal, 2-4 specific tasks
3. Identify dependencies (which goals must complete before others)

Format your response as:
SUBGOAL: [Title]
DESCRIPTION: [What needs to be accomplished]
TASKS:
- TASK: [Task title] | DESCRIPTION: [Task details] | DEPENDS_ON: [goal IDs if any, otherwise "none"]
- TASK: [Task title] | DESCRIPTION: [Task details] | DEPENDS_ON: [goal IDs if any]

Example:
SUBGOAL: Research Phase
DESCRIPTION: Gather all necessary information and data
TASKS:
- TASK: Define research scope | DESCRIPTION: Identify key topics and boundaries | DEPENDS_ON: none
- TASK: Search academic sources | DESCRIPTION: Find relevant papers and articles | DEPENDS_ON: 1
- TASK: Collect statistics | DESCRIPTION: Gather numerical data | DEPENDS_ON: 1

Be specific and concrete. Make tasks actionable."""

    response = llm.invoke([
        SystemMessage(content="You are a strategic planning assistant."),
        HumanMessage(content=decomposition_prompt)
    ])

    # Parse the response into goal hierarchy
    goal_tree = parse_goal_hierarchy(response.content, state.main_goal)

    print("\n📋 Goal Hierarchy Created:")
    print_goal_tree(goal_tree, indent=0)

    return {
        "goal_tree": goal_tree,
        "all_goals": collect_all_goals(goal_tree),
        "status": "planning_complete"
    }


def parse_goal_hierarchy(content: str, main_goal: str) -> Goal:
    """Parse LLM response into Goal hierarchy"""
    root = Goal(
        id="0",
        title=main_goal,
        description="Main objective",
        priority=10
    )

    lines = content.strip().split('\n')
    current_subgoal = None
    subgoal_counter = 1
    task_counter = 1

    for line in lines:
        line = line.strip()

        if line.startswith("SUBGOAL:"):
            title = line.replace("SUBGOAL:", "").strip()
            current_subgoal = Goal(
                id=f"sg{subgoal_counter}",
                title=title,
                description="",
                parent_id="0",
                priority=8
            )
            root.children.append(current_subgoal)
            subgoal_counter += 1

        elif line.startswith("DESCRIPTION:") and current_subgoal:
            current_subgoal.description = line.replace("DESCRIPTION:", "").strip()

        elif line.startswith("- TASK:") and current_subgoal:
            parts = line.replace("- TASK:", "").split("|")
            task_title = parts[0].strip()
            task_desc = ""
            dependencies = []

            for part in parts[1:]:
                if "DESCRIPTION:" in part:
                    task_desc = part.replace("DESCRIPTION:", "").strip()
                elif "DEPENDS_ON:" in part:
                    dep_str = part.replace("DEPENDS_ON:", "").strip()
                    if dep_str.lower() not in ["none", "n/a", ""]:
                        # Map dependency descriptions to actual IDs
                        dependencies = [current_subgoal.id]  # Depend on parent subgoal by default

            task = Goal(
                id=f"t{task_counter}",
                title=task_title,
                description=task_desc,
                parent_id=current_subgoal.id,
                dependencies=dependencies if dependencies else [],
                priority=5,
                estimated_minutes=15
            )
            current_subgoal.children.append(task)
            task_counter += 1

    return root


def collect_all_goals(root: Goal) -> dict:
    """Collect all goals into a flat dictionary for easy lookup"""
    goals = {root.id: root}

    def collect_recursive(goal: Goal):
        for child in goal.children:
            goals[child.id] = child
            collect_recursive(child)

    collect_recursive(root)
    return goals


def select_next_goal(state: GoalManagementState) -> GoalManagementState:
    """Select the next goal to execute based on dependencies and priority"""
    if state.status == "complete":
        return {"status": "complete"}

    # Get all leaf goals (tasks) that are ready to execute
    ready_goals = []

    for goal_id, goal in state.all_goals.items():
        if goal.is_leaf() and goal.status == GoalStatus.PENDING:
            if goal.is_ready(state.completed_goals):
                ready_goals.append(goal)

    if not ready_goals:
        # Check if everything is complete
        all_complete = all(
            g.status == GoalStatus.COMPLETED or not g.is_leaf()
            for g in state.all_goals.values()
        )
        if all_complete:
            return {"status": "complete"}

        # Check for blocked goals
        print("\n⚠️  No ready goals available. Checking for blockers...")
        return {"status": "blocked"}

    # Select highest priority ready goal
    next_goal = max(ready_goals, key=lambda g: g.priority)
    next_goal.status = GoalStatus.IN_PROGRESS
    next_goal.start_time = datetime.now()

    print(f"\n▶️  Selected Next Goal: {next_goal.title} (ID: {next_goal.id})")

    return {
        "current_goal_id": next_goal.id,
        "status": "executing"
    }


def execute_goal(state: GoalManagementState) -> GoalManagementState:
    """Execute the current goal"""
    current_goal = state.all_goals[state.current_goal_id]

    print(f"\n⚙️  Executing: {current_goal.title}")
    print(f"   Description: {current_goal.description}")

    execution_prompt = f"""Execute this task and provide the result:

Task: {current_goal.title}
Description: {current_goal.description}
Context: This is part of the larger goal: {state.main_goal}

Provide a concrete, specific result of completing this task. Be realistic and detailed.
If this is a planning/research task, provide specific findings or plans.
If this is an execution task, describe what was created or accomplished."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a task execution assistant."),
            HumanMessage(content=execution_prompt)
        ])

        result = response.content
        current_goal.result = result
        current_goal.status = GoalStatus.COMPLETED
        current_goal.end_time = datetime.now()

        print(f"   ✓ Completed!")
        print(f"   Result: {result[:150]}..." if len(result) > 150 else f"   Result: {result}")

        # Update parent status if all siblings complete
        update_parent_status(current_goal, state.all_goals)

        return {
            "completed_goals": state.completed_goals | {current_goal.id},
            "status": "selecting_next"
        }

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        current_goal.status = GoalStatus.FAILED
        return {
            "status": "replanning"
        }


def update_parent_status(goal: Goal, all_goals: dict):
    """Update parent goal status if all children are complete"""
    if not goal.parent_id:
        return

    parent = all_goals[goal.parent_id]
    if all(child.status == GoalStatus.COMPLETED for child in parent.children):
        parent.status = GoalStatus.COMPLETED
        print(f"   ✓ Parent goal completed: {parent.title}")
        # Recursively update grandparent
        update_parent_status(parent, all_goals)


def replan_failed_goal(state: GoalManagementState) -> GoalManagementState:
    """Create an alternative plan for a failed goal"""
    current_goal = state.all_goals[state.current_goal_id]

    print(f"\n🔄 Replanning failed goal: {current_goal.title}")

    replan_prompt = f"""A goal has failed. Create an alternative approach.

Failed Goal: {current_goal.title}
Description: {current_goal.description}
Main Goal Context: {state.main_goal}

Provide:
1. Analysis of why it might have failed
2. An alternative, simpler approach to achieve the same outcome
3. Specific steps for the new approach

Be concise and practical."""

    response = llm.invoke([
        SystemMessage(content="You are a problem-solving assistant."),
        HumanMessage(content=replan_prompt)
    ])

    # Create new task with alternative approach
    new_task = Goal(
        id=f"{current_goal.id}_retry",
        title=f"{current_goal.title} (Alternative Approach)",
        description=response.content,
        parent_id=current_goal.parent_id,
        status=GoalStatus.PENDING,
        priority=current_goal.priority + 1  # Higher priority
    )

    # Add to parent's children and all_goals
    if current_goal.parent_id:
        parent = state.all_goals[current_goal.parent_id]
        parent.children.append(new_task)

    state.all_goals[new_task.id] = new_task

    print(f"   ✓ New approach created: {new_task.title}")

    return {
        "all_goals": state.all_goals,
        "status": "selecting_next"
    }


def generate_final_report(state: GoalManagementState) -> GoalManagementState:
    """Generate a final summary report of goal completion"""
    print("\n" + "="*80)
    print("PHASE 5: FINAL REPORT")
    print("="*80)

    root_goal = state.goal_tree

    report_parts = [
        f"\n🎯 Goal Management Summary",
        f"\nMain Goal: {root_goal.title}",
        f"Status: {'✓ COMPLETED' if root_goal.status == GoalStatus.COMPLETED else '⚠ INCOMPLETE'}",
        f"\nProgress: {root_goal.progress() * 100:.0f}%",
        f"\nCompleted Goals: {len(state.completed_goals)}/{len(state.all_goals) - 1}",  # -1 for root
        "\n\nGoal Hierarchy with Results:",
        "\n" + get_results_tree(root_goal)
    ]

    report = "\n".join(report_parts)
    print(report)

    return {
        "final_report": report,
        "status": "finished"
    }


def get_results_tree(goal: Goal, indent: int = 0) -> str:
    """Generate a tree view of goals with results"""
    prefix = "  " * indent

    status_icon = {
        GoalStatus.COMPLETED: "✓",
        GoalStatus.IN_PROGRESS: "⟳",
        GoalStatus.PENDING: "○",
        GoalStatus.FAILED: "✗",
        GoalStatus.BLOCKED: "⏸"
    }

    icon = status_icon.get(goal.status, "?")

    lines = [f"{prefix}{icon} {goal.title}"]

    if goal.result and goal.is_leaf():
        result_preview = goal.result[:100] + "..." if len(goal.result) > 100 else goal.result
        lines.append(f"{prefix}  → {result_preview}")

    for child in goal.children:
        lines.append(get_results_tree(child, indent + 1))

    return "\n".join(lines)


def print_goal_tree(goal: Goal, indent: int = 0):
    """Print goal hierarchy in tree format"""
    prefix = "  " * indent
    symbol = "├─" if indent > 0 else ""

    print(f"{prefix}{symbol} {goal.title} (ID: {goal.id}, Priority: {goal.priority})")

    if goal.dependencies:
        print(f"{prefix}   Dependencies: {goal.dependencies}")

    for i, child in enumerate(goal.children):
        print_goal_tree(child, indent + 1)


def should_continue(state: GoalManagementState) -> str:
    """Determine next node in the workflow"""
    if state.status == "planning_complete":
        return "select_next"
    elif state.status == "selecting_next":
        return "select_next"
    elif state.status == "executing":
        return "execute"
    elif state.status == "replanning":
        return "replan"
    elif state.status == "blocked":
        return "end"  # End if blocked
    elif state.status == "complete":
        return "report"
    elif state.status == "finished":
        return "end"
    else:
        return "end"


# --- Build the Goal Management Workflow ---

def create_goal_management_workflow():
    """Create the LangGraph workflow for goal management"""
    workflow = StateGraph(GoalManagementState)

    # Add nodes
    workflow.add_node("decompose", decompose_goal)
    workflow.add_node("select_next", select_next_goal)
    workflow.add_node("execute", execute_goal)
    workflow.add_node("replan", replan_failed_goal)
    workflow.add_node("report", generate_final_report)

    # Set entry point
    workflow.set_entry_point("decompose")

    # Add edges
    workflow.add_conditional_edges(
        "decompose",
        should_continue,
        {
            "select_next": "select_next",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "select_next",
        should_continue,
        {
            "execute": "execute",
            "report": "report",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "select_next": "select_next",
            "replan": "replan",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "replan",
        should_continue,
        {
            "select_next": "select_next",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "report",
        should_continue,
        {
            "end": END
        }
    )

    return workflow.compile()


# --- Example Usage ---

def run_goal_management_example(main_goal: str):
    """Run a goal management example"""
    print(f"\n{'='*80}")
    print(f"Goal Management Pattern - Basic Implementation")
    print(f"{'='*80}")

    app = create_goal_management_workflow()

    initial_state = GoalManagementState(
        main_goal=main_goal,
        status="initializing"
    )

    result = app.invoke(initial_state)

    return result


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         Goal Management Pattern - Basic Implementation         ║
    ║                                                                ║
    ║  Demonstrates hierarchical goal decomposition, dependency      ║
    ║  tracking, sequential execution, and progress monitoring       ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Example: Research Report
    example_goal = "Write a comprehensive research report on renewable energy trends in 2024"

    result = run_goal_management_example(example_goal)

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    Example Complete!                           ║
    ║                                                                ║
    ║  The Goal Management pattern demonstrated:                     ║
    ║  ✓ Hierarchical decomposition of complex goal                  ║
    ║  ✓ Dependency tracking between subgoals                        ║
    ║  ✓ Sequential execution respecting dependencies                ║
    ║  ✓ Progress monitoring and status updates                      ║
    ║  ✓ Simple replanning on failure                                ║
    ║  ✓ Final comprehensive report generation                       ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
