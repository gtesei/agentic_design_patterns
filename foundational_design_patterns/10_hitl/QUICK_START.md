# Human-in-the-Loop (HITL) - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the HITL Directory
```bash
cd foundational_design_patterns/10_hitl
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
- **Option 1**: Basic HITL with Console Approval
- **Option 2**: Advanced HITL with Risk-Based Checkpoints
- **Option 3**: LangGraph HITL Integration
- **Option 4**: Run all examples

---

## 📖 Understanding HITL in 30 Seconds

**HITL** = **Human-in-the-Loop**

The agent workflow includes checkpoints where humans review and approve actions:

1. **Agent Proposes**: Creates a plan or action
2. **Checkpoint**: Pauses and asks for human approval
3. **Human Reviews**: Examines the proposed action
4. **Human Decides**: Approve / Modify / Reject
5. **Agent Continues**: Proceeds based on human input

**Why?** Safety, compliance, trust, and accountability for high-stakes decisions.

---

## 🛠️ Available Approval Mechanisms

### Synchronous Approval (Blocking)
- Agent pauses immediately
- Waits for human input via CLI prompt
- Simple, direct, real-time review
- Best for: Interactive workflows, immediate decisions

### Asynchronous Approval (Queue-Based)
- Agent submits request to queue
- Human reviews queue when available
- Agent polls for approval status
- Best for: Batch processing, non-urgent workflows

### Conditional Approval (Risk-Based)
- Automatic risk scoring of actions
- Low-risk: auto-approve
- High-risk: require human approval
- Best for: Scalable workflows, intelligent routing

### LangGraph Integration
- Checkpoints built into workflow graph
- State persistence across approvals
- Resume from last checkpoint
- Best for: Complex multi-step workflows

---

## 💡 Example Workflows to Try

### Simple Content Approval
```
Task: "Write a customer email offering a 10% discount"

Agent: Drafts email content
CHECKPOINT → Human reviews tone and accuracy
Human: Approves
Agent: Marks as ready to send
```

### Financial Transaction
```
Task: "Process refund of $150 to customer"

Agent: Prepares refund transaction
CHECKPOINT → Human verifies customer identity and amount
Human: Approves
Agent: Executes refund
```

### Multi-Step with Conditional Approval
```
Task: "Respond to 3 customer support tickets"

Ticket 1 (Low Risk: $10 refund):
  Agent: Auto-approves and processes

Ticket 2 (Medium Risk: Policy exception):
  CHECKPOINT → Human reviews
  Human: Approves with modification

Ticket 3 (High Risk: Account closure):
  CHECKPOINT → Human reviews
  Human: Rejects, requests escalation
```

### Data Deletion
```
Task: "Delete user account data for GDPR request"

Agent: Identifies data to delete (irreversible)
CHECKPOINT → Human verifies compliance and scope
Human: Approves
Agent: Executes deletion
CHECKPOINT → Human confirms completion
Human: Verifies audit trail
```

---

## 🎯 Key Concepts

### Checkpoints
Strategic pause points where human input is required:
- **Pre-execution**: Before taking any action
- **Mid-workflow**: Between multi-step operations
- **Pre-commit**: Before finalizing irreversible changes
- **Post-execution**: After completion, before delivery

### Approval States
Human responses to checkpoint reviews:
- ✅ **Approve**: Proceed as planned
- ✏️ **Modify**: Make changes and re-submit
- ❌ **Reject**: Do not proceed
- 💡 **Guide**: Provide feedback without blocking

### Risk Scoring
Automatic assessment of action risk level:
```python
Risk Factors:
  - Financial amount (higher = riskier)
  - Irreversibility (can't undo = riskier)
  - Data sensitivity (PII = riskier)
  - External visibility (public = riskier)
  - Scope of impact (many users = riskier)
```

### Audit Trails
Comprehensive logging for compliance:
- Who requested the action
- What was proposed
- Who reviewed it
- What decision was made
- When it occurred
- Why (approval/rejection reason)

---

## 📊 Comparison of Approaches

| Approach | Latency | Complexity | Best For |
|----------|---------|------------|----------|
| **Synchronous** | Real-time | Low | Interactive workflows |
| **Asynchronous** | Minutes-hours | Medium | Batch processing |
| **Conditional** | Mixed | Medium | High-volume, variable risk |
| **LangGraph** | Real-time | High | Complex multi-step |

### When to Use Each

**Synchronous**:
- User is available now
- Decision needed immediately
- Simple workflows

**Asynchronous**:
- Processing many items
- Reviewers check queue periodically
- Non-urgent tasks

**Conditional**:
- Wide range of risk levels
- Want to minimize human reviews
- Scalability is important

**LangGraph**:
- Complex multi-agent workflows
- Need state persistence
- Multiple checkpoints in sequence

---

## 🔧 Customization Tips

### Adjust Risk Threshold

In `hitl_conditional.py`, modify the risk calculation:

```python
def calculate_risk_score(action: str, context: Dict) -> float:
    """Customize risk factors and weights"""
    risk_score = 0.0

    # Add your custom risk factors
    if context.get("amount", 0) > 1000:
        risk_score += 0.4  # Adjust weight

    if context.get("your_custom_factor"):
        risk_score += 0.3  # Add new factor

    return min(risk_score, 1.0)

# Change approval threshold
RISK_THRESHOLD = 0.5  # Lower = more approvals required
```

### Customize Checkpoint Display

In any implementation, modify the display function:

```python
def display_checkpoint(action: str, context: Dict):
    """Customize how approval requests look"""
    print("\n" + "="*70)
    print("🔍 YOUR CUSTOM HEADER")
    print("="*70)

    # Add custom formatting
    print(f"Action: {action}")

    # Add custom context fields
    if "your_field" in context:
        print(f"Special Info: {context['your_field']}")

    print("="*70)
```

### Add Custom Approval Levels

```python
class ApprovalLevel:
    """Define your organization's approval hierarchy"""
    JUNIOR = "junior_analyst"
    SENIOR = "senior_analyst"
    MANAGER = "manager"
    DIRECTOR = "director"

def route_by_amount(amount: float) -> str:
    """Route based on your business rules"""
    if amount < 100:
        return ApprovalLevel.JUNIOR
    elif amount < 1000:
        return ApprovalLevel.SENIOR
    elif amount < 10000:
        return ApprovalLevel.MANAGER
    else:
        return ApprovalLevel.DIRECTOR
```

### Implement Timeout Handling

```python
def checkpoint_with_timeout(action: str, context: Dict,
                           timeout_seconds: int = 300):
    """Add timeout to any checkpoint"""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Approval timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        return human_checkpoint(action, context)
    except TimeoutError:
        print(f"⏱️ Timeout after {timeout_seconds}s, using default: REJECT")
        return False, "timeout_rejection"
    finally:
        signal.alarm(0)  # Cancel alarm
```

---

## ⚡ Common Issues & Solutions

### Issue: "Approval fatigue - too many checkpoints"
**Solution**: Implement conditional approval with risk-based routing. Auto-approve low-risk actions.

```python
# Before: All actions require approval
for action in actions:
    human_checkpoint(action, context)

# After: Only high-risk actions require approval
for action in actions:
    risk = calculate_risk_score(action, context)
    if risk > 0.5:
        human_checkpoint(action, context)
    else:
        auto_approve(action)
```

### Issue: "Agent hangs waiting for approval"
**Solution**: Implement timeouts with safe defaults.

```python
# Add timeout to prevent infinite waiting
response = wait_for_approval(timeout=300)  # 5 minutes
if response is None:
    # Use safe default (usually reject)
    return False, "timeout"
```

### Issue: "Inconsistent approval decisions"
**Solution**: Provide clear criteria and training for reviewers.

```python
# Add guidance to approval prompt
print("""
Approval Criteria:
  ✅ Approve if: Amount < $500 AND customer in good standing
  ✏️ Modify if: Request needs minor adjustments
  ❌ Reject if: Violates policy OR high risk
""")
```

### Issue: "No audit trail of approvals"
**Solution**: Implement comprehensive logging.

```python
def log_approval(checkpoint_id: str, decision: str, approver: str):
    """Log all approval decisions"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_id,
        "decision": decision,
        "approver": approver
    }
    # Write to database or log file
    audit_log.write(json.dumps(log_entry))
```

### Issue: "Scalability - human bottleneck"
**Solution**: Use batch approvals and progressive automation.

```python
# Batch similar items for efficient review
pending = get_pending_approvals()
grouped = group_by_similarity(pending)

for group in grouped:
    print(f"Review {len(group)} similar items:")
    decision = input("Approve all? (y/n): ")
    if decision == 'y':
        approve_batch(group)
```

### Issue: "Reviewers don't have enough context"
**Solution**: Provide comprehensive, structured context.

```python
def prepare_context(action: str, data: Dict) -> Dict:
    """Enrich context with all relevant information"""
    return {
        "action": action,
        "rationale": explain_why(action, data),
        "impact": calculate_impact(action, data),
        "risk_factors": identify_risks(action, data),
        "similar_past_actions": find_similar(action),
        "alternatives": suggest_alternatives(action, data)
    }
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Implementation Details**: See [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) (if available)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ **Start**: Run the synchronous example to see basic HITL
2. ✅ **Understand**: See how agent pauses for human input
3. ✅ **Explore**: Try the conditional example to see risk-based routing
4. ✅ **Experiment**: Test different risk thresholds and approval criteria
5. ✅ **Advanced**: Explore LangGraph integration for complex workflows
6. ✅ **Customize**: Add your own risk factors and approval logic
7. ✅ **Integrate**: Use HITL in your production applications

---

## 🌟 Pro Tips

### 1. Start Conservative, Automate Gradually
```python
# Phase 1: Approve everything manually
risk_threshold = 0.0  # All actions require approval

# Phase 2: After 100 approvals, auto-approve low risk
risk_threshold = 0.3

# Phase 3: After 1000 approvals, increase threshold
risk_threshold = 0.5
```

### 2. Make Approvals Fast and Easy
- Show context clearly
- Provide default actions
- Enable keyboard shortcuts
- Support batch operations
- Remember past decisions

### 3. Learn from Approval Patterns
```python
# Track what gets approved vs. rejected
approval_history = track_decisions()

# Identify patterns for auto-approval
auto_approve_patterns = find_always_approved(approval_history)

# Create templates for common scenarios
create_templates(auto_approve_patterns)
```

### 4. Implement Role-Based Access
```python
# Route to appropriate approver based on action
if action_type == "financial" and amount > 10000:
    require_approval_from(Role.FINANCE_DIRECTOR)
elif action_type == "customer_communication":
    require_approval_from(Role.SUPPORT_MANAGER)
```

### 5. Use Approval Metrics
```python
metrics = {
    "approval_rate": 0.85,  # 85% approved
    "avg_review_time": 45,   # 45 seconds average
    "timeout_rate": 0.02,    # 2% timeout
    "modification_rate": 0.10 # 10% need changes
}

# Optimize based on metrics
if metrics["approval_rate"] > 0.95:
    # Increase auto-approval threshold
    increase_threshold()
```

### 6. Design for Audit Compliance
```python
# Every approval must be:
# - Traceable (who approved)
# - Time-stamped (when)
# - Justified (why)
# - Immutable (can't change history)

class ComplianceLogger:
    def log_approval(self, checkpoint, approver, decision, reason):
        entry = create_audit_entry(checkpoint, approver, decision, reason)
        write_to_immutable_log(entry)
        notify_compliance_system(entry)
```

### 7. Handle Edge Cases
```python
# What if human is unavailable?
→ Set timeout with safe default (usually reject)

# What if approval is ambiguous?
→ Provide escalation path to supervisor

# What if reviewer makes mistake?
→ Allow corrections within time window, log changes

# What if system crashes during approval?
→ Use checkpointing to resume from last state
```

---

## 🎯 Real-World Use Cases

### E-commerce: Order Cancellations
```
Low risk ($0-50): Auto-approve
Medium risk ($50-500): Manager approval
High risk (>$500): Director approval + verification
```

### Healthcare: Prescription Changes
```
All changes require approval:
- Physician reviews AI recommendation
- Verifies patient history
- Approves or modifies dosage
- System executes approved order
```

### Finance: Transaction Processing
```
Risk-based approval:
- Known payee + small amount: Auto-approve
- New payee OR large amount: Manual review
- International + large: Multi-level approval
```

### Content Publishing: Social Media
```
Every post requires approval:
- Agent drafts content
- Marketing reviews brand alignment
- Legal reviews compliance claims
- Approved posts scheduled
```

### IT Operations: Infrastructure Changes
```
Change type-based approval:
- Config update: Team lead
- Service restart: Senior engineer
- Database migration: DBA + Manager
- Production deployment: Change board
```

---

## 🚦 Decision Framework: Do I Need HITL?

### Ask Yourself:

**1. What happens if the agent makes a mistake?**
- Minimal impact → Maybe not
- Moderate impact → Probably yes
- Severe impact → Definitely yes

**2. Is this action reversible?**
- Easily reversed → Maybe not
- Hard to reverse → Probably yes
- Irreversible → Definitely yes

**3. Are there compliance requirements?**
- No regulations → Maybe not
- Some regulations → Probably yes
- Strict regulations → Definitely yes

**4. How much do stakeholders trust the system?**
- Fully trusted → Maybe not
- Building trust → Probably yes
- No trust yet → Definitely yes

**5. What's the cost of human review?**
- Low cost + high value → Yes
- High cost + low value → Optimize with conditional
- High cost + high stakes → Yes, optimize over time

---

**Happy Building! 🚀**

For questions or issues, refer to the full [README.md](./README.md).

---

*Remember: HITL is about combining the best of both worlds—AI efficiency with human judgment. Start conservative, learn from patterns, and automate progressively as trust and confidence grow.*
