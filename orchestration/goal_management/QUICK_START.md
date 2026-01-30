# Goal Management Pattern - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Goal Management Directory
```bash
cd orchestration/goal_management
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
- **Option 1**: Basic Goal Management (hierarchical decomposition)
- **Option 2**: Advanced Goal Management (dynamic replanning, parallel execution)
- **Option 3**: Run all examples

---

## 📖 Understanding Goal Management in 30 Seconds

**Goal Management** = Breaking down complex objectives into manageable, trackable subgoals

The pattern follows this flow:
1. **Decompose**: Break main goal into hierarchical subgoals and tasks
2. **Map Dependencies**: Identify what must complete before what
3. **Execute**: Run tasks respecting dependencies, parallel where possible
4. **Monitor**: Track progress, detect blockers
5. **Adapt**: Replan when failures occur or better approaches emerge

---

## 🎯 Key Concepts

### Goal Hierarchy
```
Main Goal
├─ Subgoal 1
│  ├─ Task 1.1
│  └─ Task 1.2
├─ Subgoal 2 (depends on Subgoal 1)
│  ├─ Task 2.1
│  └─ Task 2.2
└─ Subgoal 3 (can run in parallel with Subgoal 2)
```

### Goal Status
- **PENDING**: Not yet started
- **IN_PROGRESS**: Currently executing
- **COMPLETED**: Successfully finished
- **FAILED**: Encountered error
- **BLOCKED**: Waiting for dependencies

### Dependencies
- Some goals must wait for others to complete
- Independent goals can run in parallel
- Dependencies prevent wasted work on prerequisites

---

## 💡 Example Scenarios

### Basic Example: Research Report

```
Goal: "Write a research report on renewable energy"

Decomposed into:
├─ Define scope (topics, timeframe)
├─ Gather data (research papers, statistics)
├─ Analyze trends (identify patterns)
└─ Write report (draft, visualize, refine)

Execution:
1. Define scope [COMPLETED - 5 min]
2. Gather data [IN PROGRESS - 15 min]
3. Analyze trends [BLOCKED - waiting for data]
4. Write report [PENDING]

Progress: 25% complete
```

### Advanced Example: Software Project

```
Goal: "Implement user authentication system"

Decomposed into:
├─ Requirements & Design
├─ Backend Implementation
│  ├─ Database models
│  ├─ JWT tokens
│  └─ OAuth integration
├─ Frontend Implementation (parallel with Backend)
│  ├─ UI components
│  └─ Token handling
├─ Testing (depends on Backend + Frontend)
└─ Deployment (depends on Testing)

Dynamic Features:
- Backend and Frontend run in parallel
- OAuth fails → automatically replans with alternative
- Continuous progress monitoring
- Resource allocation optimization
```

---

## 🛠️ What Each Example Shows

### Basic Implementation (`goal_basic.py`)
- Hierarchical goal decomposition
- Sequential task execution
- Dependency tracking
- Simple progress visualization
- Basic replanning on failure

**Best for**: Understanding core concepts, straightforward projects

### Advanced Implementation (`goal_advanced.py`)
- Goal graph with complex dependencies
- Parallel execution where possible
- Dynamic priority adjustment
- Continuous monitoring and replanning
- Resource allocation
- Detailed progress dashboard

**Best for**: Complex projects, production systems, multi-agent coordination

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| Goal Structure | Tree hierarchy | Graph with dependencies |
| Execution | Sequential | Parallel + Sequential |
| Replanning | On failure only | Continuous optimization |
| Monitoring | Simple status | Real-time dashboard |
| Visualization | Text-based | Rich progress display |
| Complexity | Low | Medium-High |

**Recommendation**: Start with Basic, move to Advanced for complex projects.

---

## 🔧 Try These Queries

### Simple Projects
```
"Plan a weekend trip to the mountains"
"Organize a birthday party for 20 people"
"Learn Python basics in 2 weeks"
```

### Medium Projects
```
"Write a comprehensive blog post on AI safety"
"Build a personal website with portfolio"
"Prepare for a technical interview"
```

### Complex Projects
```
"Launch a small online business"
"Develop a mobile app MVP"
"Conduct a research study on user behavior"
```

---

## ⚙️ Customization Tips

### Adjust Goal Granularity

In `goal_basic.py` or `goal_advanced.py`:
```python
# Control how finely goals are broken down
MAX_GOAL_DEPTH = 3  # Levels of decomposition
MIN_TASK_MINUTES = 10  # Minimum task size
MAX_TASK_MINUTES = 60  # Maximum before decomposing
```

### Change Execution Strategy

```python
# In goal_advanced.py
EXECUTION_MODE = "parallel"  # or "sequential"
MAX_PARALLEL_TASKS = 3
ENABLE_DYNAMIC_REPLANNING = True
```

### Customize Progress Display

```python
def display_progress(goal_tree: dict):
    """Customize how progress is shown"""
    # Add your own formatting, colors, charts, etc.
    pass
```

---

## ⚡ Common Issues & Solutions

### Issue: "Too many subgoals generated"
**Solution**: Adjust `MAX_GOAL_DEPTH` or provide more specific main goal.

### Issue: "Goals executing in wrong order"
**Solution**: Check dependency declarations, ensure prerequisites are correct.

### Issue: "Replanning taking too long"
**Solution**: Reduce replanning frequency or simplify failure handling logic.

### Issue: "Progress not updating"
**Solution**: Ensure status is being updated after each goal completion.

---

## 📈 Progress Tracking

### What Gets Tracked

- **Completion Percentage**: Overall and per-goal progress
- **Time Tracking**: Duration for each goal
- **Dependency Status**: What's blocking what
- **Failure Handling**: What failed and how it was replanned
- **Resource Usage**: API calls, compute time

### Example Progress Output

```
╔════════════════════════════════════════════╗
║     Goal Management Progress               ║
╠════════════════════════════════════════════╣
║ Overall: 60% ███████████░░░░░░░░           ║
║                                             ║
║ ✓ Define Scope         [COMPLETED]         ║
║ ⟳ Gather Data          [IN PROGRESS - 80%] ║
║ ⏸ Analyze Trends       [BLOCKED]           ║
║ ○ Write Report         [PENDING]           ║
║                                             ║
║ Active: 1 | Blocked: 1 | Remaining: 2      ║
║ Est. Time: 15 minutes remaining            ║
╚════════════════════════════════════════════╝
```

---

## 🎓 Learning Path

1. ✅ **Start**: Run basic example, see hierarchical decomposition
2. ✅ **Understand**: Watch how dependencies control execution order
3. ✅ **Explore**: Run advanced example, see parallel execution
4. ✅ **Experiment**: Try different goal types and complexities
5. ✅ **Customize**: Modify decomposition logic, add monitoring features
6. ✅ **Integrate**: Use in your own projects for complex objectives

---

## 🌟 Pro Tips

### 1. Smart Decomposition
- Aim for 3-7 main subgoals per main goal
- Each task should be 10-60 minutes
- Make dependencies explicit, not implicit
- Include verification/testing as goals

### 2. Effective Monitoring
- Check progress regularly (every 5-10 minutes for long tasks)
- Set timeouts to detect stuck goals
- Log all status changes for debugging
- Visualize critical path

### 3. Replanning Strategy
- Don't replan too eagerly (let failures accumulate context)
- Try alternative approaches, not just retry same thing
- Preserve partial progress when replanning
- Learn from failures to improve future plans

### 4. Parallel Execution
- Identify truly independent goals (no hidden dependencies)
- Balance parallelism with resource constraints
- Monitor concurrent execution for conflicts
- Use sequential execution when uncertain

### 5. Goal Quality
- Clear success criteria for each goal
- Testable/verifiable completion
- Avoid vague goals ("improve system" → "reduce load time by 20%")
- Include time estimates for planning

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Implementation Details**: Check source code comments
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🔍 Understanding the Code

### Basic Implementation Structure

```python
# 1. Define goal structure
@dataclass
class Goal:
    id: str
    title: str
    status: GoalStatus
    dependencies: List[str]
    children: List[Goal]

# 2. Decompose main goal
def decompose_goal(main_goal: str) -> Goal:
    # LLM breaks down into hierarchy
    pass

# 3. Execute with dependency respect
def execute_goal_tree(root: Goal):
    while not all_completed(root):
        ready_goals = get_ready_goals(root)
        for goal in ready_goals:
            execute_goal(goal)

# 4. Monitor and replan
def monitor_progress(root: Goal):
    if has_failures(root):
        replan_failed_goals(root)
```

### Advanced Implementation Structure

```python
# 1. Goal graph with complex dependencies
class GoalGraph:
    def __init__(self):
        self.nodes: Dict[str, Goal] = {}
        self.edges: Dict[str, Set[str]] = {}

# 2. Parallel execution coordinator
class ParallelExecutor:
    def execute_parallel(self, goals: List[Goal]):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_goal, g) for g in goals]
            wait(futures)

# 3. Dynamic replanning engine
class ReplanningEngine:
    def analyze_failure(self, goal: Goal) -> str:
        # Determine why goal failed
        pass

    def create_alternative(self, goal: Goal) -> Goal:
        # Generate new approach
        pass
```

---

## 🎯 When to Use Which Example

### Use Basic When:
- Learning the pattern
- Straightforward projects with clear steps
- Limited parallelism opportunities
- Simple dependency chains
- Fast prototyping

### Use Advanced When:
- Production systems
- Complex projects with many interdependencies
- Need parallel execution for efficiency
- Require continuous adaptation
- Multi-agent coordination
- Resource optimization important

---

## 💻 Quick Test Command

```bash
# Test basic implementation
uv run python src/goal_basic.py

# Test advanced implementation
uv run python src/goal_advanced.py

# Run both and compare
bash run.sh
```

---

**Happy Goal Managing! 🎯**

For questions or issues, refer to the full [README.md](./README.md).
