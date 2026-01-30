# Prioritization Pattern - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Prioritization Directory
```bash
cd orchestration/prioritization
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
- **Option 1**: Basic Prioritization (multi-criteria scoring, FIFO with priority)
- **Option 2**: Advanced Prioritization (dynamic rebalancing, deadline-aware, resource management)
- **Option 3**: Run all examples

---

## 📖 Understanding Prioritization in 30 Seconds

**Prioritization** = Intelligently ranking tasks and allocating limited resources to maximize value

The pattern follows this flow:
1. **Assess**: Evaluate each task on urgency, impact, effort, deadlines
2. **Score**: Calculate weighted priority scores
3. **Rank**: Order the queue by priority
4. **Execute**: Work on highest-priority ready tasks first
5. **Rebalance**: Adjust priorities as conditions change (aging, deadlines)

---

## 🎯 Key Concepts

### Priority Scoring
```
Priority Score = (
    Weight_Urgency × Urgency +
    Weight_Impact × Impact +
    Weight_Effort × (1 / Effort) +  # Quick wins boost
    Aging_Bonus  # Prevent starvation
)
```

### Priority Dimensions
- **Urgency**: How time-sensitive? (1-10)
- **Impact**: How many affected? Business value? (1-10)
- **Effort**: Time/resources required (hours)
- **Deadline**: Days/hours until due
- **Aging**: How long has it waited? (increases over time)

### Scheduling Strategies
- **Priority Queue**: Highest priority first
- **Earliest Deadline First (EDF)**: Nearest deadline first
- **Shortest Job First (SJF)**: Quick wins for throughput
- **Weighted Fair Queuing**: Ensure all levels get attention

---

## 💡 Example Scenarios

### Basic Example: Support Tickets

```
Incoming Tickets:

A: Password Reset
   Urgency: 5, Impact: 1, Effort: 5 min → Priority: 3.8

B: Payment Processing Down
   Urgency: 10, Impact: 10, Effort: 30 min → Priority: 9.8

C: Feature Request (waiting 2 days)
   Urgency: 2, Impact: 5, Effort: 4 hours → Priority: 3.5 + 2.4 (aging) = 5.9

D: Enterprise Dashboard Issue
   Urgency: 8, Impact: 8, Effort: 15 min → Priority: 8.9

Execution Order: B (9.8) → D (8.9) → C (5.9) → A (3.8)

Outcome:
✓ Critical revenue issue resolved first
✓ Enterprise SLA maintained
✓ Feature request didn't starve (aging helped)
✓ Simple task handled when capacity available
```

### Advanced Example: Development Tasks

```
Tasks with Deadlines and Resource Constraints:

1. Security Vulnerability Fix
   Urgency: 10, Impact: 10, Effort: 2h, Deadline: TODAY
   → Priority: 10.0 (Critical + Deadline pressure)

2. Payment Gateway Integration
   Urgency: 7, Impact: 8, Effort: 2 days, Deadline: 7 days
   → Priority: 7.8

3. Refactor Legacy Code (waiting 45 days)
   Urgency: 3, Impact: 4, Effort: 1 day, No deadline
   → Priority: 3.5 + 4.5 (aging) = 8.0

4. Minor UI Fix
   Urgency: 2, Impact: 2, Effort: 30 min
   → Priority: 2.5

Schedule (2 developers, 8h/day):
Day 1:
  Dev 1: Task 1 (2h) → Task 3 (6h)  [Parallel execution]
  Dev 2: Task 2 (8h)
Day 2:
  Dev 1: Task 3 (2h) → Task 4 (0.5h)
  Dev 2: Task 2 (8h remaining)

Outcome:
✓ Security patched immediately
✓ Payment gateway on track
✓ Technical debt addressed (aging boost prevented starvation)
✓ Efficient parallel execution
```

---

## 🛠️ What Each Example Shows

### Basic Implementation (`prioritization_basic.py`)
- Multi-criteria priority scoring
- Weighted priority calculation
- Simple FIFO queue with priority override
- Priority visualization
- Aging mechanism basics

**Best for**: Understanding core concepts, simple task queues

### Advanced Implementation (`prioritization_advanced.py`)
- Deadline-aware scheduling (EDF)
- Resource capacity management
- Dynamic priority rebalancing
- Preemption for critical tasks
- Real-time dashboard
- Complex scenario handling

**Best for**: Production systems, incident response, project management

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| Priority Scoring | Weighted multi-criteria | Context-aware dynamic |
| Scheduling | Priority queue | EDF + Resource-aware |
| Aging | Simple linear | Exponential with thresholds |
| Resource Management | None | Capacity tracking |
| Preemption | No | Yes, for critical tasks |
| Deadline Handling | Basic | Sophisticated (SLA tracking) |
| Visualization | Simple ranking | Rich dashboard |
| Complexity | Low | Medium-High |

**Recommendation**: Start with Basic to learn concepts, use Advanced for production needs.

---

## 🔧 Try These Scenarios

### Support/Ticketing Systems
```
"Prioritize 10 customer support tickets with varying urgency and customer tiers"
"Handle critical production incident among routine requests"
"Ensure VIP customer tickets processed within SLA"
```

### Software Development
```
"Balance security fixes, feature deadlines, and technical debt"
"Schedule bug fixes and features with resource constraints"
"Optimize developer allocation across multiple priorities"
```

### Incident Response
```
"Triage multiple active incidents by severity and impact"
"Coordinate response to simultaneous P1 and P2 issues"
"Balance immediate fixes vs. long-term stability work"
```

### Project Management
```
"Prioritize sprint backlog items by value and dependencies"
"Allocate team capacity across competing project demands"
"Handle urgent requests while maintaining strategic work"
```

---

## ⚙️ Customization Tips

### Adjust Priority Weights

In `prioritization_basic.py` or `prioritization_advanced.py`:
```python
# Tune weights based on your goals
WEIGHTS = {
    "urgency": 0.4,    # Time-sensitivity
    "impact": 0.3,     # Business value
    "effort": 0.2,     # Quick wins
    "aging": 0.1,      # Starvation prevention
}

# Different profiles for different contexts
INCIDENT_RESPONSE_WEIGHTS = {
    "urgency": 0.6,    # Time matters most
    "impact": 0.3,
    "effort": 0.1,
}

FEATURE_DEVELOPMENT_WEIGHTS = {
    "impact": 0.4,     # Value matters most
    "strategic": 0.3,
    "effort": 0.2,
    "urgency": 0.1,
}
```

### Modify Aging Strategy

```python
# Linear aging: constant boost per time unit
def linear_aging(wait_minutes: float) -> float:
    return wait_minutes * 0.05

# Exponential aging: accelerating boost
def exponential_aging(wait_minutes: float) -> float:
    if wait_minutes < 60:
        return wait_minutes * 0.01
    else:
        return 0.6 + ((wait_minutes - 60) ** 1.2) * 0.01

# Threshold aging: big boost after waiting X time
def threshold_aging(wait_minutes: float) -> float:
    if wait_minutes < 120:
        return 0.0
    elif wait_minutes < 240:
        return 2.0
    else:
        return 5.0
```

### Configure Resource Limits

```python
# In prioritization_advanced.py
RESOURCE_CONFIG = {
    "total_capacity": 10.0,      # Total available resources
    "max_parallel": 5,            # Max concurrent tasks
    "reserve_capacity": 0.2,      # Keep 20% for emergencies
    "preemption_enabled": True,   # Allow interrupting tasks
    "preemption_threshold": 3.0,  # Priority gap needed
}
```

---

## ⚡ Common Issues & Solutions

### Issue: "Low-priority tasks never execute"
**Solution**: Increase aging factor or set maximum wait time guarantees.
```python
MAX_WAIT_TIME_MINUTES = 480  # Force to front after 8 hours
AGING_FACTOR = 0.15  # More aggressive aging
```

### Issue: "Too many high-priority tasks"
**Solution**: Use multi-level queues or stricter priority validation.
```python
# Separate queues for different priority classes
CRITICAL_QUEUE  # P0: Top 5% of tasks
HIGH_QUEUE      # P1: Next 20%
MEDIUM_QUEUE    # P2: Next 50%
LOW_QUEUE       # P3: Remaining 25%
```

### Issue: "Deadline misses increasing"
**Solution**: Switch to EDF (Earliest Deadline First) scheduling.
```python
SCHEDULING_ALGORITHM = "EDF"  # vs "PRIORITY" or "SJF"
DEADLINE_WEIGHT = 0.5  # Increase deadline importance
```

### Issue: "Resource thrashing from rebalancing"
**Solution**: Rate-limit priority recalculations.
```python
REBALANCE_INTERVAL_SECONDS = 60  # Only rebalance every minute
REBALANCE_TRIGGER_THRESHOLD = 5   # Or when 5+ new tasks arrive
```

### Issue: "Priority gaming by users"
**Solution**: Implement validation and caps on user-set priorities.
```python
def validate_priority(user_claimed: int, system_calculated: int) -> int:
    max_boost = 2.0
    if user_claimed > system_calculated + max_boost:
        # Require justification or manager approval
        return system_calculated + max_boost
    return user_claimed
```

---

## 📈 Monitoring Priority Health

### Key Metrics to Track

```
Queue Health:
- Queue depth (tasks waiting)
- Average wait time
- Max wait time (detect starvation)
- Throughput (tasks/hour)

Priority Metrics:
- Priority distribution (histogram)
- Priority accuracy (high = important?)
- Gaming attempts (suspicious inflation)

Deadline Metrics:
- SLA compliance rate
- Deadline violations
- At-risk tasks (likely to miss)

Resource Metrics:
- Utilization rate
- Idle time
- Context switch overhead
```

### Example Monitoring Output

```
╔══════════════════════════════════════════════╗
║     Priority Queue Health Monitor            ║
╠══════════════════════════════════════════════╣
║ Queue Depth: 25 tasks                        ║
║ Average Wait: 45 minutes                     ║
║ Max Wait: 180 minutes ⚠️ STARVATION RISK     ║
║                                              ║
║ Priority Distribution:                       ║
║   Critical (9-10): ███░ 15%                  ║
║   High (7-8):      ██████░ 30%               ║
║   Medium (5-6):    ████████░ 40%             ║
║   Low (1-4):       ███░ 15%                  ║
║                                              ║
║ SLA Compliance: 94.5% (target: 95%)          ║
║ Resource Utilization: 87%                    ║
║ Throughput: 12 tasks/hour                    ║
║                                              ║
║ ⚠️ Alerts:                                   ║
║ - 3 tasks waiting > 3 hours                  ║
║ - 2 tasks at risk of missing deadline        ║
╚══════════════════════════════════════════════╝
```

---

## 🎓 Learning Path

1. ✅ **Start**: Run basic example, see multi-criteria scoring
2. ✅ **Understand**: Watch how priorities determine execution order
3. ✅ **Observe**: See aging prevent starvation of low-priority tasks
4. ✅ **Explore**: Run advanced example with deadlines and resources
5. ✅ **Experiment**: Adjust weights, compare different strategies
6. ✅ **Monitor**: Track queue health metrics and optimize
7. ✅ **Integrate**: Apply to your own task queuing systems

---

## 🌟 Pro Tips

### 1. Start Simple, Add Complexity Gradually
- Begin with 2-3 priority criteria (urgency, impact)
- Add effort consideration for quick wins
- Introduce aging after observing starvation
- Add deadline handling when SLAs matter
- Layer in resource management as needed

### 2. Tune Weights Based on Outcomes
- Monitor if high-priority tasks actually matter more
- Adjust weights if wrong things get prioritized
- Different weights for different contexts (incidents vs. features)
- Use A/B testing to compare strategies
- Consider ML-based weight learning from historical data

### 3. Prevent Starvation Proactively
- Always implement some form of aging
- Set hard maximum wait time limits (e.g., 8 hours)
- Use weighted fair queuing to guarantee progress
- Monitor oldest task age continuously
- Alert when tasks wait too long

### 4. Balance Optimization vs. Overhead
- Don't rebalance every second (batch updates)
- Cache priority calculations
- Use approximate algorithms for large queues
- Prioritization time should be << task execution time
- Monitor rebalancing overhead

### 5. Make Priorities Explainable
- Log why each task got its priority
- Show criteria breakdown to users
- Provide estimated wait times
- Explain queue position changes
- Enable debugging of priority decisions

### 6. Guard Against Gaming
- Validate user-claimed priorities
- Cross-check with objective criteria
- Require justification for "critical" tasks
- Flag suspicious patterns (everyone claiming urgent)
- Use system-calculated priorities when possible

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Implementation Details**: Check source code comments
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🔍 Understanding the Code

### Basic Implementation Structure

```python
# 1. Define task with priority dimensions
@dataclass
class Task:
    id: str
    urgency: int      # 1-10
    impact: int       # 1-10
    effort: float     # hours
    arrival_time: float

    def priority_score(self) -> float:
        """Calculate weighted priority"""
        return (
            WEIGHTS["urgency"] * self.urgency +
            WEIGHTS["impact"] * self.impact +
            WEIGHTS["effort"] * (10 / self.effort) +
            self.aging_bonus()
        )

# 2. Priority queue implementation
class PriorityQueue:
    def __init__(self):
        self.tasks = []

    def add(self, task: Task):
        heapq.heappush(self.tasks, (-task.priority_score(), task))

    def pop_highest(self) -> Task:
        _, task = heapq.heappop(self.tasks)
        return task

# 3. Execute in priority order
while queue.has_tasks():
    task = queue.pop_highest()
    execute_task(task)
```

### Advanced Implementation Structure

```python
# 1. Deadline-aware task
@dataclass
class DeadlineTask(Task):
    deadline: datetime

    def deadline_pressure(self) -> float:
        """Urgency based on deadline proximity"""
        time_remaining = (self.deadline - datetime.now()).total_seconds()
        if time_remaining <= 0:
            return 100.0  # Overdue!
        else:
            return 10.0 / (time_remaining / 3600)  # Hours remaining

# 2. Resource manager
class ResourceManager:
    def __init__(self, capacity: float):
        self.total_capacity = capacity
        self.available = capacity

    def can_start(self, task: Task) -> bool:
        return self.available >= task.required_capacity

    def allocate(self, task: Task):
        self.available -= task.required_capacity

# 3. Dynamic scheduler
class DynamicScheduler:
    def rebalance(self):
        """Recalculate priorities periodically"""
        for task in self.queue.tasks:
            # Update aging bonus
            # Check deadline pressure
            # Adjust for resource availability
            task.recalculate_priority()

        self.queue.resort()
```

---

## 🎯 When to Use Which Example

### Use Basic When:
- Learning the prioritization pattern
- Simple task queues (support tickets, work items)
- Static or slowly-changing priorities
- No hard deadlines or SLAs
- Single resource pool
- Prototyping quickly

### Use Advanced When:
- Production systems with SLAs
- Complex scheduling with deadlines
- Multiple resource types and constraints
- Need preemption for critical tasks
- Dynamic priority rebalancing important
- Want detailed monitoring and dashboards
- Managing incidents or critical workflows

---

## 💻 Quick Test Commands

```bash
# Test basic implementation
uv run python src/prioritization_basic.py

# Test advanced implementation with deadlines
uv run python src/prioritization_advanced.py

# Run both and compare approaches
bash run.sh
```

---

## 📖 Pattern Relationships

**Prioritization works well with:**
- **Goal Management**: Prioritize which goals to pursue
- **Planning**: Prioritize which plan steps to execute first
- **Multi-Agent**: Prioritize task allocation across agents
- **Resource Management**: Optimize capacity allocation

**Prioritization differs from:**
- **Load Balancing**: All tasks equal, distribute evenly
- **FIFO Queuing**: First-in-first-out, no priority
- **Round Robin**: Fair rotation, ignores urgency

---

**Happy Prioritizing! 🎯**

For questions or issues, refer to the full [README.md](./README.md).
