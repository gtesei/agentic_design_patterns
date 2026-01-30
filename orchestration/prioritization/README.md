# Prioritization Pattern

## Overview

The **Prioritization Pattern** is an orchestration approach that enables AI agents to intelligently rank tasks, allocate limited resources, and determine optimal execution order based on multiple criteria such as urgency, impact, effort, deadlines, and strategic importance. Unlike simple first-in-first-out (FIFO) queues or static scheduling, Prioritization provides a dynamic framework for making informed decisions about what to work on next when faced with competing demands and resource constraints.

This pattern transforms chaotic task management into structured, goal-aligned execution by continuously assessing priority based on changing conditions, preventing resource starvation, ensuring deadline adherence, and optimizing for business value. It's particularly valuable for support systems, project management, incident response, resource allocation, and any scenario where multiple tasks compete for limited capacity.

## Why Use This Pattern?

Traditional approaches have significant limitations:

- **FIFO queues**: First task in gets worked on first, regardless of urgency or importance
- **Manual prioritization**: Subjective, inconsistent, doesn't scale, can't adapt quickly
- **Static priorities**: Set once and never change, even as conditions evolve
- **No resource awareness**: Overcommit or underutilize available capacity
- **Starvation risk**: Low-priority tasks may wait indefinitely

Prioritization solves these by:
- **Multi-criteria scoring**: Consider urgency, impact, effort, strategic alignment, and deadlines
- **Dynamic rebalancing**: Priorities adjust as tasks age, deadlines approach, or conditions change
- **Resource optimization**: Allocate capacity to maximize throughput and value delivery
- **Deadline awareness**: Ensure time-sensitive tasks complete on schedule
- **Starvation prevention**: Gradually increase priority of waiting tasks (aging)
- **Transparent ranking**: Clear, explainable prioritization decisions

### Example: Support Tickets Without Prioritization

```
Queue: [Ticket A, Ticket B, Ticket C, Ticket D, Ticket E]
→ Process in arrival order
→ Critical production outage (Ticket D) waits behind password reset (Ticket A)
→ Important feature request (Ticket C) with deadline missed
→ VIP customer (Ticket E) frustrated by long wait
→ Simple 5-minute fix (Ticket B) blocks 2-hour investigation
```

### Example: Support Tickets With Prioritization

```
Incoming: [A: Password reset, B: UI bug, C: Feature request (deadline tomorrow),
           D: Production outage, E: VIP customer issue]

Priority Assessment:
┌────┬─────────────────────┬─────────┬────────┬────────┬────────┬───────────┐
│ ID │ Task                │ Urgency │ Impact │ Effort │ Aging  │ Priority  │
├────┼─────────────────────┼─────────┼────────┼────────┼────────┼───────────┤
│ D  │ Production outage   │   10    │   10   │   8    │   0    │   9.5     │
│ E  │ VIP customer issue  │    9    │    8   │   5    │   1    │   8.2     │
│ C  │ Feature (deadline)  │    8    │    7   │   7    │   2    │   7.8     │
│ B  │ UI bug              │    6    │    5   │   3    │   2    │   5.5     │
│ A  │ Password reset      │    5    │    2   │   2    │   1    │   3.8     │
└────┴─────────────────────┴─────────┴────────┴────────┴────────┴───────────┘

Execution Order: D → E → C → B → A
✓ Critical issues addressed first
✓ Deadline-sensitive tasks prioritized
✓ VIP customers served quickly
✓ Resource allocation optimized
```

## How It Works

The Prioritization pattern operates through five interconnected phases:

```
┌────────────────────────────────────────────────────────────────┐
│                     1. TASK ASSESSMENT                          │
│  Evaluate each task across multiple dimensions (urgency,       │
│  impact, effort, strategic value, deadline, customer tier)     │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     2. PRIORITY SCORING                         │
│  Calculate weighted priority scores combining all criteria     │
│  Priority = w1×urgency + w2×impact + w3×(1/effort) + aging    │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     3. QUEUE RANKING                            │
│  Order tasks by priority, considering deadlines and            │
│  dependencies (Earliest Deadline First, Critical Path)         │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     4. RESOURCE ALLOCATION                      │
│  Assign available resources to highest-priority ready tasks    │
│  considering capacity constraints and skill requirements       │
└─────────────────────────┬──────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                     5. DYNAMIC REBALANCING                      │
│  Continuously reassess priorities as tasks age, deadlines      │
│  approach, new tasks arrive, or conditions change              │
└────────────────────────────────────────────────────────────────┘
          ↓                                              ↑
          └──────────────────────────────────────────────┘
                    (Continuous loop)
```

### Phase Breakdown

1. **Assessment**: Evaluate each task's urgency, business impact, effort required, deadline proximity, and strategic importance
2. **Scoring**: Calculate weighted priority scores using configurable weights for each criterion
3. **Ranking**: Order the queue by priority, applying scheduling algorithms (EDF, SJF, weighted fair queuing)
4. **Allocation**: Assign work to available resources, respecting capacity limits and skill requirements
5. **Rebalancing**: Recalculate priorities as time passes, conditions change, or new information emerges

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Customer support systems**: Prioritize tickets by urgency, customer tier, SLA requirements
- **Incident response**: Triage and address critical production issues before minor bugs
- **Project task management**: Balance urgent deadlines with important strategic work
- **Resource allocation**: Distribute limited compute, API quota, or human attention optimally
- **Request queuing**: Handle API rate limits, job scheduling, batch processing priorities
- **Development workflows**: Prioritize bug fixes, features, technical debt, and research
- **Emergency services**: Medical triage, disaster response, resource deployment
- **Content moderation**: Flag and review high-risk content before low-risk items
- **Research agenda**: Decide which experiments or studies to pursue given time/budget constraints

### ❌ When NOT to Use

- **Single task at a time**: No need to prioritize if only doing one thing
- **All tasks equal priority**: If everything truly has the same importance (rare)
- **Strict FIFO requirements**: Some systems require first-in-first-out by design
- **Instantaneous decisions**: When overhead of prioritization exceeds task duration
- **No resource constraints**: Unlimited capacity means everything can run in parallel

## Rule of Thumb

**Use Prioritization when:**
1. You have **more tasks than capacity** to handle simultaneously
2. Tasks have **different levels of urgency or importance**
3. There are **deadlines or SLAs** to meet
4. Some tasks provide **higher business value** than others
5. **Resource optimization** matters (time, compute, people, budget)
6. You need to **prevent low-priority tasks from starving**

**Don't use Prioritization when:**
1. Processing one task at a time in order received
2. All tasks genuinely equal in every dimension
3. No resource constraints or competing demands
4. Prioritization overhead > task execution time
5. System design requires strict ordering guarantees

## Core Components

### 1. Priority Scoring Function

Calculates task priority based on multiple weighted criteria:

```python
Priority = (
    w_urgency × urgency_score +
    w_impact × impact_score +
    w_effort × (1 / effort_estimate) +  # Prefer quick wins
    w_deadline × deadline_pressure +
    w_strategic × strategic_value +
    aging_bonus  # Prevent starvation
)
```

**Common criteria:**
- **Urgency**: How time-sensitive is the task? (1-10)
- **Impact**: How many users/systems affected? Business value? (1-10)
- **Effort**: Time/resources required (lower effort = higher priority for quick wins)
- **Deadline**: Days/hours until due (approaching deadlines increase priority)
- **Strategic alignment**: Fits company goals, OKRs, roadmap? (1-10)
- **Customer tier**: Free, paid, enterprise, VIP? (affects weight)
- **Aging**: How long has task been waiting? (increases over time)

### 2. Scheduling Algorithms

Different strategies for ordering the queue:

**Simple Priority Queue**:
- Order by priority score (highest first)
- Static weights, recalculated periodically

**Earliest Deadline First (EDF)**:
- Prioritize tasks with nearest deadlines
- Optimal for meeting all deadlines when possible
- Risk: May sacrifice important tasks for urgent unimportant ones

**Shortest Job First (SJF)**:
- Execute quick tasks first to maximize throughput
- Improves average completion time
- Risk: Starvation of long tasks

**Weighted Fair Queuing**:
- Allocate capacity proportionally to priority weights
- Ensures all priority levels get some attention
- Prevents complete starvation

**Multi-Level Feedback Queue**:
- Tasks start at high priority
- Decrease priority if not completed quickly
- Prevents monopolization by long-running tasks

### 3. Resource Manager

Tracks and allocates available capacity:

```python
ResourceManager:
  - available_capacity: Current free resources
  - reserved_capacity: Allocated to in-progress tasks
  - total_capacity: Maximum system capacity
  - allocation_strategy: How to distribute resources
  - constraints: Skill requirements, dependencies, quotas
```

**Functions:**
- Check resource availability before starting tasks
- Reserve resources for assigned work
- Release resources when tasks complete
- Prevent overallocation and thrashing

### 4. Aging Mechanism

Prevents low-priority tasks from waiting indefinitely:

```python
aging_bonus = wait_time × aging_factor

# Example: Task waiting 10 minutes gets +2.0 priority boost
aging_bonus = 10 minutes × 0.2 = +2.0

# Eventually, even low-priority tasks rise to the top
```

**Strategies:**
- **Linear aging**: Constant boost per time unit
- **Exponential aging**: Accelerating boost as wait time increases
- **Threshold aging**: Big boost after waiting X time
- **Adaptive aging**: Adjust rate based on queue depth

### 5. Preemption Controller

Decides if higher-priority tasks should interrupt running work:

```python
PreemptionPolicy:
  - enabled: bool  # Allow preemption?
  - threshold: float  # Priority gap required to preempt
  - checkpoints: bool  # Can tasks save state and resume?
  - cost: float  # Penalty for context switching
```

**Use preemption when:**
- Critical task arrives (production outage)
- Significant priority gap (critical vs. low)
- Running task is interruptible (has checkpoints)
- Benefit > context switch cost

## Implementation Approaches

### Approach 1: Simple Multi-Criteria Scoring

Basic weighted priority calculation:

```python
from dataclasses import dataclass
from typing import List
import heapq

@dataclass
class Task:
    id: str
    urgency: int  # 1-10
    impact: int   # 1-10
    effort: int   # hours
    wait_time: float = 0.0

    def priority(self) -> float:
        """Calculate priority score"""
        # Weighted scoring
        score = (
            0.4 * self.urgency +
            0.3 * self.impact +
            0.2 * (10 / self.effort) +  # Inverse effort (quick wins)
            0.1 * self.wait_time  # Aging bonus
        )
        return score

class PriorityQueue:
    def __init__(self):
        self.tasks = []

    def add(self, task: Task):
        # Use negative priority for max-heap behavior
        heapq.heappush(self.tasks, (-task.priority(), task))

    def pop_highest(self) -> Task:
        _, task = heapq.heappop(self.tasks)
        return task

# Usage
queue = PriorityQueue()
queue.add(Task(id="A", urgency=9, impact=8, effort=2))
queue.add(Task(id="B", urgency=5, impact=10, effort=8))

next_task = queue.pop_highest()  # Returns highest priority
```

### Approach 2: Deadline-Aware Scheduling

Earliest Deadline First (EDF) algorithm:

```python
from datetime import datetime, timedelta
from dataclasses import dataclass
import heapq

@dataclass
class DeadlineTask:
    id: str
    deadline: datetime
    effort_hours: float
    priority_base: float

    def deadline_pressure(self) -> float:
        """How urgent based on deadline proximity"""
        time_remaining = (self.deadline - datetime.now()).total_seconds() / 3600
        if time_remaining <= 0:
            return 1000.0  # Overdue - critical
        elif time_remaining < self.effort_hours:
            return 100.0  # Deadline at risk
        elif time_remaining < 2 * self.effort_hours:
            return 50.0  # Getting close
        else:
            return 10.0 / time_remaining  # Normal priority

    def priority(self) -> float:
        return self.priority_base + self.deadline_pressure()

class EDFScheduler:
    def __init__(self):
        self.queue = []

    def schedule(self, task: DeadlineTask):
        # EDF: prioritize by deadline, with base priority modifier
        heapq.heappush(self.queue, (task.deadline, -task.priority(), task))

    def next_task(self) -> DeadlineTask:
        _, _, task = heapq.heappop(self.queue)
        return task
```

### Approach 3: Resource-Aware Prioritization

Consider resource availability and constraints:

```python
@dataclass
class ResourceTask:
    id: str
    priority: float
    required_capacity: float  # 0.0 - 1.0
    estimated_duration: float  # minutes
    skills_needed: List[str]

class ResourceScheduler:
    def __init__(self, total_capacity: float):
        self.total_capacity = total_capacity
        self.available_capacity = total_capacity
        self.running_tasks = []
        self.queue = []

    def can_start(self, task: ResourceTask) -> bool:
        """Check if resources available"""
        return self.available_capacity >= task.required_capacity

    def allocate(self, task: ResourceTask):
        """Reserve resources for task"""
        self.available_capacity -= task.required_capacity
        self.running_tasks.append(task)

    def release(self, task: ResourceTask):
        """Free resources after completion"""
        self.available_capacity += task.required_capacity
        self.running_tasks.remove(task)

    def schedule_next(self):
        """Start highest-priority task that fits"""
        # Sort queue by priority
        self.queue.sort(key=lambda t: t.priority, reverse=True)

        for task in self.queue:
            if self.can_start(task):
                self.queue.remove(task)
                self.allocate(task)
                return task

        return None  # No task fits current capacity
```

### Approach 4: Dynamic Rebalancing with Aging

Prevent starvation with priority aging:

```python
import time
from typing import Dict

class AgingPriorityQueue:
    def __init__(self, aging_factor: float = 0.1):
        self.tasks: Dict[str, Task] = {}
        self.arrival_times: Dict[str, float] = {}
        self.aging_factor = aging_factor

    def add(self, task: Task):
        """Add task and record arrival time"""
        self.tasks[task.id] = task
        self.arrival_times[task.id] = time.time()

    def get_priority(self, task_id: str) -> float:
        """Calculate current priority with aging"""
        task = self.tasks[task_id]
        wait_time = time.time() - self.arrival_times[task_id]

        # Base priority + aging bonus
        aged_priority = task.priority() + (wait_time * self.aging_factor)
        return aged_priority

    def next_task(self) -> Task:
        """Get highest priority task considering aging"""
        if not self.tasks:
            return None

        # Find task with highest aged priority
        best_id = max(self.tasks.keys(), key=self.get_priority)
        task = self.tasks.pop(best_id)
        del self.arrival_times[best_id]

        return task

    def rebalance(self):
        """Recalculate all priorities with current aging"""
        # Called periodically or when new tasks arrive
        pass
```

## Key Benefits

### 🎯 Optimal Resource Utilization

- **Maximize throughput**: Complete more high-value work in less time
- **Minimize waste**: Avoid spending resources on low-impact tasks
- **Balance workload**: Distribute capacity across priority levels
- **Improve efficiency**: Quick wins and high-impact work get attention

### ⏰ Deadline Adherence

- **Meet SLAs**: Ensure time-sensitive commitments are honored
- **Early warning**: Detect deadline risks before they become failures
- **Proactive scheduling**: Allocate time for approaching deadlines
- **Prevent escalations**: Address urgent issues before they become crises

### 📊 Goal Alignment

- **Strategic focus**: Prioritize work aligned with business objectives
- **Value maximization**: Deliver highest-impact outcomes first
- **Clear trade-offs**: Explicit decisions about what to defer
- **Measurable impact**: Track value delivered per resource spent

### ⚖️ Fairness and Starvation Prevention

- **Aging mechanism**: Low-priority tasks eventually rise to the top
- **Guaranteed progress**: All tasks complete eventually
- **Balanced attention**: Weighted fair queuing ensures all levels served
- **Predictable wait times**: Estimate when tasks will be addressed

### 🔍 Transparency and Explainability

- **Clear rankings**: Understand why one task prioritized over another
- **Auditable decisions**: Log priority scores and criteria
- **Stakeholder visibility**: Show what's being worked on and why
- **Continuous optimization**: Data-driven tuning of weights and algorithms

## Trade-offs

### ⚠️ Prioritization Overhead

**Issue**: Calculating priorities, rebalancing, and managing queues adds computational cost

**Impact**: Reduced throughput if prioritization time > task execution time

**Mitigation**:
- Use simple scoring for fast decisions
- Cache priority calculations
- Rebalance periodically rather than continuously
- Use approximate algorithms for large queues

### 🔄 Risk of Starvation

**Issue**: Low-priority tasks may wait indefinitely if high-priority work keeps arriving

**Impact**: Some tasks never complete, SLAs missed, customer frustration

**Mitigation**:
- Implement aging/priority boost over time
- Set maximum wait time thresholds
- Use weighted fair queuing for guaranteed progress
- Reserve capacity for low-priority work

### 📈 Complexity in Multi-Criteria Decisions

**Issue**: Balancing urgency, impact, effort, deadlines, and other factors is non-trivial

**Impact**: Difficult to tune weights, unexpected prioritization, gaming the system

**Mitigation**:
- Start with simple scoring (2-3 criteria)
- Use machine learning to learn optimal weights from historical data
- A/B test different prioritization strategies
- Collect feedback on prioritization quality

### ⚡ Priority Inversion

**Issue**: Low-priority task blocks high-priority task (e.g., holds shared resource)

**Impact**: High-priority work delayed despite capacity availability

**Mitigation**:
- Priority inheritance: Boost blocking task's priority temporarily
- Minimize resource contention
- Use preemption when safe and beneficial
- Explicit dependency tracking

### 🎲 Gaming and Manipulation

**Issue**: Users may artificially inflate urgency/impact to jump the queue

**Impact**: Prioritization loses effectiveness, important work deprioritized

**Mitigation**:
- Require justification for high urgency/impact
- Manager approval for priority escalations
- Monitor for priority inflation patterns
- Separate user-reported vs. system-calculated priority

## Best Practices

### 1. Define Clear Priority Criteria

```python
# Explicit, measurable criteria
PRIORITY_CRITERIA = {
    "urgency": {
        "critical": 10,    # Production down, security breach
        "high": 8,         # Major feature broken, deadline today
        "medium": 5,       # Important but not time-critical
        "low": 2,          # Nice to have, no deadline
    },
    "impact": {
        "all_users": 10,   # Affects entire user base
        "large_segment": 7, # 20%+ of users
        "small_segment": 4, # < 20% of users
        "individual": 1,    # Single user
    },
    "customer_tier": {
        "enterprise": 10,  # Paying customers, SLAs
        "paid": 7,         # Standard paid tier
        "trial": 4,        # Evaluation users
        "free": 2,         # Free tier
    }
}
```

### 2. Tune Weights Based on Goals

```python
# Different profiles for different objectives
WEIGHT_PROFILES = {
    "maximize_throughput": {
        # Favor quick wins
        "urgency": 0.2,
        "impact": 0.3,
        "effort": 0.4,  # High weight on low effort
        "aging": 0.1,
    },
    "deadline_focused": {
        # Ensure SLAs met
        "urgency": 0.4,
        "deadline_pressure": 0.4,
        "impact": 0.1,
        "aging": 0.1,
    },
    "value_maximization": {
        # Highest business impact
        "impact": 0.5,
        "strategic_value": 0.3,
        "urgency": 0.1,
        "aging": 0.1,
    }
}
```

### 3. Implement Aging to Prevent Starvation

```python
def calculate_aging_bonus(wait_time_minutes: float, base_priority: float) -> float:
    """
    Gradually increase priority as tasks wait

    Strategy: Linear aging with acceleration
    - First 30 min: slow growth
    - 30-120 min: linear growth
    - 120+ min: exponential growth
    """
    if wait_time_minutes < 30:
        return wait_time_minutes * 0.01  # Minimal boost
    elif wait_time_minutes < 120:
        return 0.3 + (wait_time_minutes - 30) * 0.05  # Linear
    else:
        # Exponential boost after 2 hours
        excess = wait_time_minutes - 120
        return 4.8 + (excess ** 1.2) * 0.01

# Ensure even lowest-priority task eventually reaches top
# Example: Low priority (2.0) after 3 hours wait → ~10.0 priority
```

### 4. Monitor and Alert on Queue Health

```python
class QueueHealthMonitor:
    def check_health(self, queue: PriorityQueue) -> Dict[str, Any]:
        """Monitor for issues"""
        metrics = {
            "queue_depth": len(queue.tasks),
            "oldest_task_age": self.get_oldest_age(queue),
            "average_wait_time": self.get_average_wait(queue),
            "starvation_risk": self.count_long_waiting(queue, threshold_minutes=180),
            "deadline_violations": self.count_overdue(queue),
        }

        alerts = []

        # Alert if tasks waiting too long
        if metrics["oldest_task_age"] > 240:  # 4 hours
            alerts.append(f"Task waiting {metrics['oldest_task_age']} minutes - starvation risk!")

        # Alert if queue growing unbounded
        if metrics["queue_depth"] > 1000:
            alerts.append(f"Queue depth {metrics['queue_depth']} - need more capacity!")

        # Alert on deadline violations
        if metrics["deadline_violations"] > 0:
            alerts.append(f"{metrics['deadline_violations']} tasks overdue!")

        return {"metrics": metrics, "alerts": alerts}
```

### 5. Separate Queues for Different Classes

```python
class MultiQueuePrioritizer:
    """Separate queues for different priority classes"""

    def __init__(self):
        self.critical_queue = PriorityQueue()  # P0: Production issues
        self.high_queue = PriorityQueue()      # P1: Important deadlines
        self.medium_queue = PriorityQueue()    # P2: Normal work
        self.low_queue = PriorityQueue()       # P3: Backlog

    def add_task(self, task: Task):
        """Route to appropriate queue"""
        if task.is_critical():
            self.critical_queue.add(task)
        elif task.priority() >= 8:
            self.high_queue.add(task)
        elif task.priority() >= 5:
            self.medium_queue.add(task)
        else:
            self.low_queue.add(task)

    def next_task(self) -> Task:
        """Weighted fair selection across queues"""
        # 60% from critical, 25% high, 10% medium, 5% low
        # Ensures all queues get attention

        rand = random.random()

        if rand < 0.60 and not self.critical_queue.empty():
            return self.critical_queue.pop()
        elif rand < 0.85 and not self.high_queue.empty():
            return self.high_queue.pop()
        elif rand < 0.95 and not self.medium_queue.empty():
            return self.medium_queue.pop()
        else:
            return self.low_queue.pop()
```

### 6. Log Priority Decisions for Analysis

```python
@dataclass
class PriorityDecision:
    timestamp: datetime
    task_id: str
    priority_score: float
    criteria_breakdown: Dict[str, float]
    position_in_queue: int
    estimated_wait_time: float

def log_priority_decision(task: Task, queue: PriorityQueue):
    """Record prioritization for later analysis"""
    decision = PriorityDecision(
        timestamp=datetime.now(),
        task_id=task.id,
        priority_score=task.priority(),
        criteria_breakdown={
            "urgency": task.urgency * WEIGHTS["urgency"],
            "impact": task.impact * WEIGHTS["impact"],
            "effort_factor": (10 / task.effort) * WEIGHTS["effort"],
            "aging_bonus": task.aging_bonus(),
        },
        position_in_queue=queue.get_position(task),
        estimated_wait_time=queue.estimate_wait_time(task),
    )

    # Store in database or log file for analysis
    logger.info(f"Priority decision: {decision}")
```

## Performance Metrics

Track these metrics to evaluate prioritization effectiveness:

### Queue Metrics
- **Queue depth**: Number of waiting tasks (trend over time)
- **Average wait time**: Time from submission to start
- **Max wait time**: Longest any task has waited
- **Throughput**: Tasks completed per hour/day
- **Starvation count**: Tasks waiting > threshold time

### Priority Metrics
- **Priority distribution**: Histogram of priority scores
- **Priority accuracy**: Do high-priority tasks actually matter more?
- **Priority drift**: How often do priorities change significantly?
- **Gaming attempts**: Suspicious priority inflation patterns

### Deadline Metrics
- **SLA compliance**: % of tasks completed within deadline
- **Average slack time**: Time buffer between completion and deadline
- **Deadline violations**: Count and severity of missed deadlines
- **At-risk tasks**: Count of tasks unlikely to meet deadline

### Resource Metrics
- **Utilization rate**: % of capacity actively used
- **Allocation efficiency**: Time spent on high-value vs. low-value work
- **Context switch cost**: Overhead from preemption/rebalancing
- **Idle time**: Capacity available but no ready tasks

### Business Metrics
- **Value delivered**: Sum of business value for completed tasks
- **Customer satisfaction**: Related to priority handling
- **Cost per task**: Resource cost divided by tasks completed
- **ROI on prioritization**: Value gain vs. prioritization overhead

## Example Scenarios

### Scenario 1: Customer Support Prioritization

```
Incoming Support Tickets:

Ticket A: "Can't reset password"
- Urgency: 6 (blocking user but not critical)
- Impact: 1 (single user)
- Effort: 5 min
- Customer: Free tier
- Priority Score: 4.2

Ticket B: "Payment processing down"
- Urgency: 10 (revenue-impacting)
- Impact: 10 (all paid users affected)
- Effort: 30 min
- Customer: All paid
- Priority Score: 9.8

Ticket C: "Feature request: Dark mode"
- Urgency: 2 (nice to have)
- Impact: 5 (many users want it)
- Effort: 4 hours
- Customer: Mixed
- Priority Score: 3.5
- Wait time: 2 days → Aging bonus +2.4
- Adjusted Priority: 5.9

Ticket D: "Dashboard charts not loading for enterprise client"
- Urgency: 8 (SLA at risk)
- Impact: 8 (affects major customer)
- Effort: 15 min
- Customer: Enterprise (SLA)
- Priority Score: 8.9

Prioritized Queue:
1. Ticket B (9.8) - Payment processing down [START IMMEDIATELY]
2. Ticket D (8.9) - Enterprise dashboard issue [NEXT]
3. Ticket C (5.9) - Dark mode feature [After aging boost]
4. Ticket A (4.2) - Password reset [When capacity available]

Outcome:
✓ Revenue-critical issue resolved first
✓ Enterprise SLA maintained
✓ Old feature request not starved (aging helped)
✓ Simple password reset handled when convenient
```

### Scenario 2: Software Development Task Prioritization

```
Development Backlog:

Task A: "Critical security vulnerability fix"
- Urgency: 10 (active exploit in the wild)
- Impact: 10 (all users at risk)
- Effort: 2 hours
- Deadline: Today
- Strategic value: 10
- Priority: 10.0

Task B: "Implement new payment gateway"
- Urgency: 7 (deadline next week)
- Impact: 8 (enables new revenue stream)
- Effort: 16 hours (2 days)
- Deadline: 7 days
- Strategic value: 9
- Priority: 7.8

Task C: "Refactor legacy authentication code"
- Urgency: 3 (technical debt, not urgent)
- Impact: 4 (improves maintainability)
- Effort: 8 hours (1 day)
- Deadline: None
- Strategic value: 5
- Wait time: 45 days → Aging +4.5
- Priority: 7.5

Task D: "Fix minor UI alignment issue"
- Urgency: 2 (cosmetic)
- Impact: 2 (low visibility)
- Effort: 30 min
- Deadline: None
- Strategic value: 1
- Priority: 2.5

Resource Constraints:
- 2 developers available
- 8 hours/day capacity

Schedule:
Day 1:
  Dev 1: Task A (2h) → Task C (6h) [High priority + aging boost]
  Dev 2: Task B (8h) [Parallel execution]

Day 2:
  Dev 1: Task C (2h remaining) → Task D (0.5h)
  Dev 2: Task B (8h remaining)

Outcome:
✓ Security vulnerability patched same day
✓ Payment gateway on track for deadline
✓ Technical debt addressed (didn't starve due to aging)
✓ Minor UI fix completed when time allowed
✓ Parallel execution optimized throughput
```

### Scenario 3: Incident Response Prioritization

```
Active Incidents:

Incident 1: "Database connection pool exhausted"
- Severity: P1 (Critical)
- Affected: 80% of users experiencing errors
- Impact: Revenue loss $5K/hour
- Estimated fix time: 30 min
- Deadline pressure: HIGH (ongoing)
- Priority: 9.5

Incident 2: "Slow page load on marketing site"
- Severity: P3 (Low)
- Affected: 5% of visitors notice
- Impact: Potential lead loss
- Estimated fix time: 2 hours
- Deadline pressure: LOW
- Priority: 3.2

Incident 3: "Email notifications delayed"
- Severity: P2 (Medium)
- Affected: All users (non-blocking)
- Impact: User experience degradation
- Estimated fix time: 1 hour
- Deadline pressure: MEDIUM (SLA: 4 hours)
- Priority: 6.5

Incident 4: "API rate limiting too aggressive"
- Severity: P2 (Medium)
- Affected: 10 enterprise API customers
- Impact: Customer escalations, contract risk
- Estimated fix time: 15 min (config change)
- Deadline pressure: MEDIUM (SLA: 2 hours)
- Priority: 7.8

Dynamic Rebalancing:

T=0: Prioritize Incident 1 (9.5) - Critical production issue
T=30min: Incident 1 resolved
         Next: Incident 4 (7.8) - Quick win, high customer impact
T=45min: Incident 4 resolved
         Next: Incident 3 (6.5) - SLA deadline approaching
T=1h45min: Incident 3 resolved
           Next: Incident 2 (3.2 → 4.7 with aging)
T=3h45min: Incident 2 resolved

Outcome:
✓ Production restored in 30 min
✓ Enterprise customers unblocked quickly
✓ All SLAs met
✓ No incidents starved
✓ Efficient use of on-call engineer time
```

## Advanced Patterns

### 1. Predictive Prioritization

Use machine learning to predict task priority and adjust proactively:

```python
class PredictivePrioritizer:
    """ML-based priority prediction"""

    def __init__(self):
        self.model = self.train_priority_model()

    def train_priority_model(self):
        """Train on historical data: features → actual priority"""
        # Features: task metadata, time, context
        # Labels: final priority score after completion
        # Learn patterns like "similar tasks escalated" or "VIP customers"
        pass

    def predict_priority(self, task: Task) -> float:
        """Predict priority using learned patterns"""
        features = self.extract_features(task)
        predicted_priority = self.model.predict(features)
        return predicted_priority

    def adjust_for_trends(self, task: Task) -> float:
        """Factor in current trends and anomalies"""
        base_priority = task.priority()
        predicted = self.predict_priority(task)

        # Blend rule-based and ML-based priorities
        return 0.7 * base_priority + 0.3 * predicted
```

### 2. Constraint Satisfaction Scheduling

Optimize across multiple constraints simultaneously:

```python
from ortools.sat.python import cp_model

class ConstraintBasedScheduler:
    """Use constraint programming for optimal scheduling"""

    def schedule_tasks(self, tasks: List[Task], resources: List[Resource],
                      constraints: List[Constraint]) -> Schedule:
        """
        Find optimal schedule satisfying all constraints:
        - Resource capacity limits
        - Task dependencies
        - Deadline requirements
        - Priority optimization
        """
        model = cp_model.CpModel()

        # Variables: start time for each task
        starts = {t.id: model.NewIntVar(0, HORIZON, f'start_{t.id}')
                  for t in tasks}

        # Constraints
        for task in tasks:
            # Resource capacity
            model.Add(resource_usage <= capacity)

            # Dependencies
            for dep in task.dependencies:
                model.Add(starts[task.id] >= starts[dep] + duration[dep])

            # Deadlines
            if task.deadline:
                model.Add(starts[task.id] + task.duration <= task.deadline)

        # Objective: Maximize weighted priority completion
        model.Maximize(sum(priority[t] * completed[t] for t in tasks))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        return self.extract_schedule(solver, starts)
```

### 3. Multi-Objective Optimization

Balance competing objectives (throughput vs. fairness vs. deadlines):

```python
class MultiObjectivePrioritizer:
    """Balance multiple goals using Pareto optimization"""

    def __init__(self, objectives: Dict[str, float]):
        """
        objectives: weights for different goals
        {
            "throughput": 0.3,     # Maximize tasks completed
            "value": 0.4,          # Maximize business value
            "fairness": 0.2,       # Minimize max wait time
            "deadline_adherence": 0.1  # Meet deadlines
        }
        """
        self.objectives = objectives

    def prioritize(self, tasks: List[Task]) -> List[Task]:
        """
        Find Pareto-optimal schedule balancing all objectives
        """
        # Score each task on each objective
        scores = {task: self.multi_objective_score(task) for task in tasks}

        # Sort by composite score
        return sorted(tasks, key=lambda t: scores[t], reverse=True)

    def multi_objective_score(self, task: Task) -> float:
        """Weighted combination of objectives"""
        return (
            self.objectives["throughput"] * task.throughput_contribution() +
            self.objectives["value"] * task.business_value() +
            self.objectives["fairness"] * task.fairness_score() +
            self.objectives["deadline_adherence"] * task.deadline_score()
        )
```

### 4. Adaptive Weight Learning

Automatically tune priority weights based on outcomes:

```python
class AdaptivePrioritizer:
    """Learn optimal weights from feedback"""

    def __init__(self):
        self.weights = {"urgency": 0.3, "impact": 0.3, "effort": 0.2, "aging": 0.2}
        self.performance_history = []

    def update_weights(self, completed_tasks: List[Task], metrics: Dict):
        """
        Adjust weights based on outcomes:
        - Did high-priority tasks actually matter?
        - Were deadlines met?
        - Was throughput optimal?
        - Customer satisfaction scores
        """
        # Reinforcement learning approach
        if metrics["sla_compliance"] < 0.95:
            # Missed deadlines → increase urgency/deadline weight
            self.weights["urgency"] += 0.05
            self.weights["impact"] -= 0.05

        if metrics["starvation_count"] > 5:
            # Tasks starving → increase aging weight
            self.weights["aging"] += 0.05
            self.weights["urgency"] -= 0.05

        if metrics["value_per_hour"] < target:
            # Low value delivery → increase impact weight
            self.weights["impact"] += 0.05
            self.weights["effort"] -= 0.05

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        self.performance_history.append({
            "timestamp": datetime.now(),
            "weights": self.weights.copy(),
            "metrics": metrics
        })
```

## Comparison with Related Patterns

| Pattern | Focus | Scheduling | Adaptation | When to Use |
|---------|-------|------------|------------|-------------|
| **Prioritization** | Task ranking | Multi-criteria priority | Dynamic rebalancing | Competing demands, limited resources |
| **Goal Management** | Goal decomposition | Dependency-based | Replanning on failure | Complex multi-step objectives |
| **Agent Communication** | Coordination | Negotiated allocation | Message-driven | Multi-agent systems |
| **Planning** | Strategy creation | Upfront plan | Replanning cycles | Known problem space |
| **Load Balancing** | Resource distribution | Round-robin, least-loaded | Reactive to load | Stateless parallel work |

**Key Differences:**

**Prioritization vs. Goal Management:**
- Prioritization: Flat list of tasks, ranked by priority
- Goal Management: Hierarchical goals with dependencies

**Prioritization vs. Planning:**
- Prioritization: Ongoing, continuous decision-making
- Planning: Upfront strategy, then execute plan

**Prioritization vs. Load Balancing:**
- Prioritization: Tasks have different values/priorities
- Load Balancing: All requests equal, distribute evenly

## Common Pitfalls

### 1. Too Many Priority Levels

**Problem**: 20 priority levels makes decisions complex and meaningless

**Solution**: Use 3-5 levels (Critical, High, Medium, Low, Deferred)

```python
# Bad: Too granular
priority_levels = range(1, 21)  # 20 levels

# Good: Clear categories
class PriorityLevel(Enum):
    CRITICAL = 10  # P0: Production down
    HIGH = 7       # P1: Important deadline
    MEDIUM = 5     # P2: Normal work
    LOW = 2        # P3: Nice to have
    DEFERRED = 0   # P4: Backlog
```

### 2. Ignoring Context and Dependencies

**Problem**: High-priority task blocked by low-priority prerequisite

**Solution**: Use priority inheritance or boost blocking tasks

```python
def adjust_for_dependencies(task: Task, queue: PriorityQueue) -> float:
    """Boost priority of blocking tasks"""
    if task.is_blocking(high_priority_tasks=queue.get_high_priority()):
        # Inherit priority from highest task blocked
        max_blocked_priority = max(t.priority() for t in task.blocks)
        return max(task.priority(), max_blocked_priority)
    return task.priority()
```

### 3. Static Weights Don't Fit All Situations

**Problem**: Same priority weights used for incidents vs. feature development

**Solution**: Use context-specific weight profiles

```python
# Different weights for different contexts
CONTEXTS = {
    "incident_response": {
        "urgency": 0.6,      # Time matters most
        "impact": 0.3,
        "effort": 0.1,
    },
    "feature_development": {
        "impact": 0.4,       # Value matters most
        "strategic_value": 0.3,
        "effort": 0.2,       # Prefer quick wins
        "urgency": 0.1,
    },
    "support_tickets": {
        "urgency": 0.3,
        "customer_tier": 0.3,  # VIP customers matter
        "impact": 0.2,
        "effort": 0.2,
    }
}
```

### 4. No Maximum Wait Time Guarantee

**Problem**: Low-priority tasks wait indefinitely despite aging

**Solution**: Set hard limits on wait time

```python
def enforce_max_wait_time(queue: PriorityQueue, max_wait_minutes: int = 480):
    """Force-promote tasks waiting too long"""
    now = time.time()

    for task in queue.tasks:
        wait_time = (now - task.arrival_time) / 60  # minutes

        if wait_time >= max_wait_minutes:
            # Override priority - force to front of queue
            task.override_priority = 999.0
            logger.warning(f"Task {task.id} forced to front after {wait_time:.0f} min wait")
```

### 5. Thrashing from Excessive Rebalancing

**Problem**: Constantly recalculating priorities and reordering queue

**Solution**: Batch updates, rate-limit rebalancing

```python
class RateLimitedRebalancer:
    """Prevent excessive rebalancing"""

    def __init__(self, min_interval_seconds: int = 60):
        self.last_rebalance = 0
        self.min_interval = min_interval_seconds
        self.pending_changes = []

    def request_rebalance(self, reason: str):
        """Queue rebalance request"""
        self.pending_changes.append((time.time(), reason))

    def maybe_rebalance(self, queue: PriorityQueue):
        """Rebalance if enough time passed and changes pending"""
        now = time.time()

        if now - self.last_rebalance < self.min_interval:
            return  # Too soon

        if not self.pending_changes:
            return  # No changes

        # Perform rebalance
        queue.recalculate_priorities()
        self.last_rebalance = now
        self.pending_changes.clear()
```

### 6. Priority Inflation/Gaming

**Problem**: Everyone marks their tasks as urgent to jump the queue

**Solution**: Validate priorities, require justification, use system-calculated scores

```python
class PriorityValidator:
    """Prevent priority gaming"""

    def validate_priority(self, task: Task, user_claimed: int) -> int:
        """
        Cross-check user-claimed priority with objective criteria
        """
        # Calculate system priority based on objective data
        system_priority = self.calculate_objective_priority(task)

        # Allow user to boost within reason
        max_boost = 2.0

        if user_claimed > system_priority + max_boost:
            logger.warning(f"Priority inflation detected: Task {task.id}, "
                          f"claimed {user_claimed} vs system {system_priority}")

            # Require justification for large deviations
            if not task.priority_justification:
                return system_priority  # Cap at system priority

            # Flag for manager review
            self.flag_for_review(task, user_claimed, system_priority)

        return min(user_claimed, system_priority + max_boost)
```

## Conclusion

The Prioritization pattern is essential for making intelligent decisions about resource allocation and task execution order in resource-constrained environments. By systematically assessing urgency, impact, effort, deadlines, and strategic value, systems can maximize throughput, meet SLAs, deliver business value, and ensure fairness across competing demands.

**Use Prioritization when:**
- Tasks compete for limited resources (time, compute, people)
- Different tasks have different values and urgency levels
- Deadlines and SLAs must be met consistently
- You need to optimize for throughput, value, or fairness
- Starvation prevention is important
- Transparent, explainable prioritization decisions required

**Implementation checklist:**
- ✅ Define clear, measurable priority criteria (urgency, impact, effort, deadlines)
- ✅ Implement multi-criteria scoring with tunable weights
- ✅ Add aging mechanism to prevent starvation
- ✅ Monitor queue health (depth, wait times, deadline risks)
- ✅ Use appropriate scheduling algorithm (EDF for deadlines, WFQ for fairness)
- ✅ Log priority decisions for analysis and tuning
- ✅ Implement resource-aware allocation
- ✅ Set maximum wait time guarantees
- ✅ Validate and guard against priority gaming
- ✅ Measure and optimize based on business metrics

**Key Takeaways:**
- 🎯 Prioritization enables optimal resource allocation under constraints
- 📊 Multi-criteria scoring provides explainable, tunable decisions
- ⏰ Deadline-aware scheduling ensures SLA compliance
- ⚖️ Aging mechanisms prevent starvation of low-priority work
- 🔄 Dynamic rebalancing adapts to changing conditions
- 📈 Continuous monitoring and tuning improves outcomes over time
- 🎲 Guard against gaming and priority inflation
- 🔍 Transparency and logging enable data-driven optimization

---

*Prioritization transforms chaotic task queues into strategic, value-optimized execution—ensuring that resources flow to where they matter most, deadlines are met, and no task is left behind indefinitely.*
