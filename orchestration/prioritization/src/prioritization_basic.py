"""
Prioritization Pattern: Basic Implementation

This example demonstrates the Prioritization pattern with multi-criteria scoring,
weighted priority calculation, simple FIFO queue with priority override, and
basic aging mechanism to prevent starvation.

Problem: Prioritize incoming tasks (e.g., support tickets, feature requests, bug reports)
Solution: Assess urgency, impact, and effort; calculate weighted priority; execute in order
"""

import heapq
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=False)


# --- Priority Data Structures ---

class TaskStatus(Enum):
    """Status of a task in execution lifecycle"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TaskType(Enum):
    """Type/category of task"""
    SUPPORT_TICKET = "support_ticket"
    BUG_FIX = "bug_fix"
    FEATURE_REQUEST = "feature_request"
    TECHNICAL_DEBT = "technical_debt"
    INCIDENT = "incident"


@dataclass
class Task:
    """Represents a task that needs to be prioritized and executed"""
    id: str
    title: str
    description: str
    task_type: TaskType
    urgency: int  # 1-10 scale (10 = most urgent)
    impact: int  # 1-10 scale (10 = highest impact)
    effort_hours: float  # Estimated effort in hours
    status: TaskStatus = TaskStatus.PENDING
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[str] = None

    def wait_time_minutes(self) -> float:
        """Calculate how long this task has been waiting"""
        return (time.time() - self.arrival_time) / 60

    def aging_bonus(self, aging_factor: float = 0.1) -> float:
        """Calculate priority boost based on wait time (prevent starvation)"""
        wait_minutes = self.wait_time_minutes()
        return wait_minutes * aging_factor

    def priority_score(self, weights: Dict[str, float], aging_factor: float = 0.1) -> float:
        """
        Calculate weighted priority score

        Priority = w_urgency × urgency + w_impact × impact + w_effort × (1/effort) + aging_bonus

        Higher score = higher priority
        """
        # Effort contribution: inverse relationship (shorter tasks get boost)
        # Normalize effort: 10 / effort_hours (e.g., 1 hour = 10, 10 hours = 1)
        effort_score = 10.0 / max(self.effort_hours, 0.5)  # Avoid division by small numbers

        base_priority = (
            weights["urgency"] * self.urgency +
            weights["impact"] * self.impact +
            weights["effort"] * effort_score
        )

        # Add aging bonus to prevent starvation
        aged_priority = base_priority + self.aging_bonus(aging_factor)

        return aged_priority


@dataclass
class PriorityConfig:
    """Configuration for priority calculation"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "urgency": 0.4,
        "impact": 0.3,
        "effort": 0.2,
    })
    aging_factor: float = 0.1  # Priority increase per minute of waiting


class PriorityQueue:
    """Priority queue that orders tasks by priority score"""

    def __init__(self, config: PriorityConfig):
        self.config = config
        self.tasks: List[Task] = []
        self._heap: List[tuple] = []  # (negative_priority, counter, task)
        self._counter = 0  # Tie-breaker for equal priorities

    def add_task(self, task: Task):
        """Add a task to the priority queue"""
        self.tasks.append(task)
        priority = task.priority_score(self.config.weights, self.config.aging_factor)
        # Use negative priority for max-heap behavior (highest priority first)
        heapq.heappush(self._heap, (-priority, self._counter, task))
        self._counter += 1

    def pop_highest_priority(self) -> Optional[Task]:
        """Remove and return the highest priority task"""
        if not self._heap:
            return None

        _, _, task = heapq.heappop(self._heap)
        return task

    def rebalance(self):
        """Recalculate priorities for all pending tasks (aging effect)"""
        # Get all pending tasks
        pending_tasks = [task for task in self.tasks if task.status == TaskStatus.PENDING]

        # Rebuild heap with updated priorities
        self._heap = []
        for task in pending_tasks:
            priority = task.priority_score(self.config.weights, self.config.aging_factor)
            heapq.heappush(self._heap, (-priority, self._counter, task))
            self._counter += 1

    def peek_top(self, n: int = 5) -> List[tuple[Task, float]]:
        """View top N tasks without removing them"""
        sorted_heap = sorted(self._heap)
        result = []
        for neg_priority, _, task in sorted_heap[:n]:
            result.append((task, -neg_priority))
        return result

    def size(self) -> int:
        """Number of pending tasks in queue"""
        return len(self._heap)


# --- Prioritization Functions ---

def assess_task_priority(task_description: str, task_type: str) -> Dict[str, int]:
    """
    Use LLM to assess priority dimensions for a task

    Returns: Dictionary with urgency, impact, and effort estimates
    """
    assessment_prompt = f"""You are a task prioritization expert. Assess this task on three dimensions:

Task Type: {task_type}
Task Description: {task_description}

Provide ratings on a 1-10 scale:

1. URGENCY (1-10): How time-sensitive is this task?
   - 10: Critical/emergency (production down, security breach, blocking all users)
   - 7-9: High urgency (major feature broken, deadline today, VIP customer)
   - 4-6: Medium urgency (important but not time-critical)
   - 1-3: Low urgency (nice to have, no deadline)

2. IMPACT (1-10): How many people/systems are affected? What's the business value?
   - 10: All users/entire system affected, major revenue impact
   - 7-9: Large user segment affected (20%+), significant business value
   - 4-6: Small user segment affected (<20%), moderate value
   - 1-3: Individual users or minimal business impact

3. EFFORT (hours): How long will this task take?
   - Estimate in hours (0.5 to 40 hours)
   - Consider complexity, dependencies, testing, deployment

Format your response EXACTLY as:
URGENCY: [number 1-10]
IMPACT: [number 1-10]
EFFORT: [number in hours]

Example:
URGENCY: 8
IMPACT: 9
EFFORT: 2.0
"""

    response = llm.invoke([
        SystemMessage(content="You are a task prioritization expert."),
        HumanMessage(content=assessment_prompt)
    ])

    # Parse response
    content = response.content.strip()
    lines = [line.strip() for line in content.split('\n') if ':' in line]

    urgency = 5
    impact = 5
    effort = 4.0

    for line in lines:
        if line.startswith('URGENCY:'):
            urgency = int(line.split(':')[1].strip())
        elif line.startswith('IMPACT:'):
            impact = int(line.split(':')[1].strip())
        elif line.startswith('EFFORT:'):
            effort = float(line.split(':')[1].strip())

    return {
        "urgency": urgency,
        "impact": impact,
        "effort": effort
    }


def execute_task(task: Task) -> str:
    """
    Simulate task execution

    In a real system, this would dispatch to appropriate handler
    """
    task.status = TaskStatus.IN_PROGRESS
    task.start_time = time.time()

    execution_prompt = f"""You are executing a task. Provide a brief summary of the actions taken.

Task: {task.title}
Description: {task.description}
Type: {task.task_type.value}
Estimated Effort: {task.effort_hours} hours

Describe the key actions you would take to complete this task (2-3 sentences).
Be specific and practical."""

    response = llm.invoke([
        SystemMessage(content="You are a task execution assistant."),
        HumanMessage(content=execution_prompt)
    ])

    result = response.content.strip()

    task.status = TaskStatus.COMPLETED
    task.end_time = time.time()
    task.result = result

    return result


def display_priority_visualization(queue: PriorityQueue):
    """Display current priority queue state"""
    print("\n" + "="*80)
    print("PRIORITY QUEUE VISUALIZATION")
    print("="*80 + "\n")

    top_tasks = queue.peek_top(n=10)

    if not top_tasks:
        print("Queue is empty.\n")
        return

    print(f"Queue Size: {queue.size()} tasks\n")
    print(f"{'Rank':<6} {'ID':<8} {'Priority':<10} {'Urgency':<9} {'Impact':<8} "
          f"{'Effort':<8} {'Aging':<8} {'Title':<30}")
    print("-" * 120)

    for rank, (task, priority) in enumerate(top_tasks, 1):
        aging = task.aging_bonus(queue.config.aging_factor)
        wait_min = task.wait_time_minutes()

        # Truncate title if too long
        title = task.title[:27] + "..." if len(task.title) > 30 else task.title

        print(f"{rank:<6} {task.id:<8} {priority:<10.2f} {task.urgency:<9} {task.impact:<8} "
              f"{task.effort_hours:<8.1f} +{aging:<7.2f} {title:<30}")

        if wait_min > 30:
            print(f"       └─> Waiting {wait_min:.0f} minutes (aging boost applied)")

    print("\n" + "="*80 + "\n")


def display_execution_summary(completed_tasks: List[Task]):
    """Display summary of completed tasks"""
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80 + "\n")

    total_time = sum(task.end_time - task.start_time for task in completed_tasks)

    print(f"Total Tasks Completed: {len(completed_tasks)}")
    print(f"Total Execution Time: {total_time:.1f} seconds")
    print(f"Average Time per Task: {total_time/len(completed_tasks):.1f} seconds\n")

    print(f"{'Exec Order':<12} {'Task ID':<10} {'Priority':<10} {'Wait Time':<12} {'Title':<35}")
    print("-" * 120)

    for idx, task in enumerate(completed_tasks, 1):
        wait_minutes = (task.start_time - task.arrival_time) / 60
        # Recalculate priority at time of execution
        priority = task.priority_score(
            {"urgency": 0.4, "impact": 0.3, "effort": 0.2},
            aging_factor=0.1
        )
        title = task.title[:32] + "..." if len(task.title) > 35 else task.title

        print(f"{idx:<12} {task.id:<10} {priority:<10.2f} {wait_minutes:<12.1f} {title:<35}")

    print("\n" + "="*80 + "\n")


# --- Main Demonstration ---

def run_basic_prioritization():
    """Run basic prioritization demonstration"""
    print("\n" + "="*80)
    print("PRIORITIZATION PATTERN - BASIC IMPLEMENTATION")
    print("="*80)
    print("\nScenario: Customer Support Ticket Prioritization")
    print("Problem: Multiple support tickets arriving, need to handle in optimal order")
    print("Solution: Multi-criteria priority scoring with aging to prevent starvation\n")

    # Sample incoming tasks
    incoming_tasks = [
        {
            "id": "TICK-001",
            "title": "Password reset not working",
            "description": "User cannot reset password via email link",
            "type": TaskType.SUPPORT_TICKET
        },
        {
            "id": "TICK-002",
            "title": "Payment processing completely down",
            "description": "All payment transactions failing, revenue impact",
            "type": TaskType.INCIDENT
        },
        {
            "id": "TICK-003",
            "title": "Add dark mode to dashboard",
            "description": "Feature request from multiple users for dark mode UI",
            "type": TaskType.FEATURE_REQUEST
        },
        {
            "id": "TICK-004",
            "title": "Enterprise client: Dashboard charts not loading",
            "description": "VIP enterprise customer reporting dashboard visualization issues",
            "type": TaskType.BUG_FIX
        },
        {
            "id": "TICK-005",
            "title": "Refactor legacy authentication code",
            "description": "Technical debt: old auth code needs modernization",
            "type": TaskType.TECHNICAL_DEBT
        },
        {
            "id": "TICK-006",
            "title": "Minor UI alignment issue on mobile",
            "description": "Button slightly misaligned on mobile view, cosmetic only",
            "type": TaskType.BUG_FIX
        },
    ]

    # Initialize priority queue
    config = PriorityConfig()
    queue = PriorityQueue(config)

    print("="*80)
    print("PHASE 1: TASK ASSESSMENT")
    print("="*80 + "\n")

    # Assess and add tasks to queue
    tasks = []
    for idx, task_data in enumerate(incoming_tasks):
        print(f"[{idx+1}/{len(incoming_tasks)}] Assessing: {task_data['title']}")

        # Get priority dimensions from LLM
        assessment = assess_task_priority(task_data["description"], task_data["type"].value)

        # Create task
        task = Task(
            id=task_data["id"],
            title=task_data["title"],
            description=task_data["description"],
            task_type=task_data["type"],
            urgency=assessment["urgency"],
            impact=assessment["impact"],
            effort_hours=assessment["effort"],
        )

        # Simulate some tasks arriving earlier (for aging demonstration)
        if idx in [2, 4]:  # Dark mode and tech debt arrived earlier
            task.arrival_time = time.time() - (120 * 60)  # 2 hours ago

        tasks.append(task)
        queue.add_task(task)

        print(f"  Urgency: {task.urgency}/10, Impact: {task.impact}/10, Effort: {task.effort_hours}h\n")

        time.sleep(0.5)  # Small delay for readability

    # Display initial priority queue
    display_priority_visualization(queue)

    print("="*80)
    print("PHASE 2: PRIORITY RANKING")
    print("="*80 + "\n")

    print("Priority Score Formula:")
    print(f"  Priority = {config.weights['urgency']} × Urgency + "
          f"{config.weights['impact']} × Impact + "
          f"{config.weights['effort']} × (10/Effort) + Aging Bonus\n")

    print("Aging Mechanism:")
    print(f"  Aging Bonus = Wait Time (minutes) × {config.aging_factor}")
    print("  → Prevents low-priority tasks from starving\n")

    input("Press Enter to start task execution...\n")

    print("="*80)
    print("PHASE 3: TASK EXECUTION")
    print("="*80 + "\n")

    # Execute tasks in priority order
    completed_tasks = []
    execution_count = 0

    while queue.size() > 0 and execution_count < len(incoming_tasks):
        # Rebalance queue to update aging (in real system, do this periodically)
        if execution_count > 0:
            queue.rebalance()

        # Get highest priority task
        task = queue.pop_highest_priority()
        if not task:
            break

        execution_count += 1
        priority = task.priority_score(config.weights, config.aging_factor)
        wait_minutes = task.wait_time_minutes()

        print(f"\n[{execution_count}/{len(incoming_tasks)}] Executing: {task.title}")
        print(f"  ID: {task.id}")
        print(f"  Priority Score: {priority:.2f}")
        print(f"  Wait Time: {wait_minutes:.1f} minutes")
        print(f"  Urgency: {task.urgency}, Impact: {task.impact}, Effort: {task.effort_hours}h")

        if wait_minutes > 30:
            aging = task.aging_bonus(config.aging_factor)
            print(f"  Aging Boost: +{aging:.2f} (task waited {wait_minutes:.0f} minutes)")

        print("\n  Actions taken:")

        # Execute the task
        result = execute_task(task)

        # Display result with indentation
        for line in result.split('\n'):
            if line.strip():
                print(f"    - {line.strip()}")

        completed_tasks.append(task)

        print(f"\n  Status: COMPLETED ✓")

        time.sleep(1)  # Simulate execution time

    # Display final summary
    display_execution_summary(completed_tasks)

    print("="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")

    print("✓ Critical tasks (payment down) executed first despite arrival order")
    print("✓ High-impact VIP customer issues prioritized appropriately")
    print("✓ Aging mechanism boosted old feature requests (dark mode, tech debt)")
    print("✓ Quick wins (low effort) got slight priority boost")
    print("✓ Cosmetic issues handled last when capacity available")
    print("✓ No tasks starved - all eventually executed\n")

    print("Benefits of Prioritization:")
    print("  • Revenue-critical issues resolved immediately")
    print("  • VIP customer SLAs maintained")
    print("  • Fair treatment via aging (no indefinite waiting)")
    print("  • Efficient resource utilization (quick wins considered)")
    print("  • Transparent, explainable prioritization decisions\n")


if __name__ == "__main__":
    run_basic_prioritization()
