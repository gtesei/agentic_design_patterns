# Agent-to-Agent Communication - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the A2A Communication Directory
```bash
cd orchestration/agent_communication
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic A2A with Direct Messaging
- **Option 2**: Advanced A2A with Pub-Sub and Negotiation
- **Option 3**: Run all examples

---

## 📖 Understanding A2A Communication in 30 Seconds

**A2A Communication** = Agents talking to each other through structured messages

Instead of hardcoding how agents work together, they:
1. **Discover** who can help with a task
2. **Send messages** to request help or share information
3. **Coordinate** dynamically based on capabilities
4. **Track conversations** to maintain context

Like humans collaborating via email, chat, and meetings!

---

## 🛠️ Communication Patterns

### Direct Messaging
```
Agent A ──[request]──> Agent B
Agent A <──[response]─┘
```
Point-to-point communication for specific requests.

### Broadcast
```
Agent A ──[announcement]──┬──> Agent B
                          ├──> Agent C
                          └──> Agent D
```
Share information with multiple agents at once.

### Pub-Sub (Topic-Based)
```
Publisher ──[topic:research]──> Broker ──> Subscriber 1
                                       └──> Subscriber 2
```
Agents subscribe to topics they care about.

### Negotiation
```
Agent A ──[can you help?]──> Agent B
Agent A <──[yes, for $X]─── Agent B
Agent A ──[agreed]────────> Agent B
```
Agents negotiate task assignment and terms.

---

## 💡 Example Scenarios

### Scenario 1: Research Team (Basic)
```
User: "Research quantum computing and write a summary"

1. Coordinator → Researcher: "Research quantum computing"
2. Researcher → Coordinator: "Here are the findings..."
3. Coordinator → Writer: "Write summary from this research"
4. Writer → Coordinator: "Here's the summary..."
5. Coordinator → Reviewer: "Review this summary"
6. Reviewer → Coordinator: "Approved with minor edits"
```

### Scenario 2: Software Dev Team (Advanced)
```
User: "Implement user authentication feature"

1. PM broadcasts: "New feature: user authentication"
2. Developer1: "I can do it in 3 days"
3. Developer2: "I can do it in 2 days"
4. PM awards to Developer2
5. Developer2 → Tester: "Ready for testing"
6. Tester → Developer2: "Found 2 bugs"
7. Developer2 → Reviewer: "Fixed, please review"
8. Reviewer → Developer2: "Approved!"
```

---

## 🎯 Key Concepts

### Message Types

**Request**: Ask for help
```python
{
    "type": "request",
    "action": "research_topic",
    "params": {"topic": "AI safety"}
}
```

**Response**: Reply with results
```python
{
    "type": "response",
    "status": "success",
    "data": {"findings": [...]}
}
```

**Notification**: Broadcast information
```python
{
    "type": "notification",
    "event": "task_completed",
    "agent": "researcher"
}
```

### Agent Registry

Agents register their capabilities:
```python
registry.register("researcher", capabilities=["research", "fact_check"])
registry.register("writer", capabilities=["writing", "editing"])

# Find who can help
capable_agents = registry.discover(capability="research")
```

### Conversation Tracking

Related messages grouped together:
```python
conversation_id = "conv_123"

# All messages share the same conversation_id
send_message(recipient="agent_b",
             conversation_id=conversation_id,
             content="Start research")

# Later, get full conversation history
history = get_conversation(conversation_id)
```

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| Communication | Direct messaging | Pub-sub + negotiation |
| Agent Discovery | Hardcoded | Dynamic registry |
| Coordination | Sequential | Negotiation-based |
| Message Tracking | Simple logs | Full conversation history |
| Visualization | Text output | Rich formatted traces |
| Complexity | Low | Medium |

**Recommendation**: Start with Basic to understand concepts, then explore Advanced for production features.

---

## 🔧 Customization Tips

### Add Your Own Agent

```python
from typing import Dict, Any

class MyCustomAgent:
    def __init__(self, agent_id: str, capabilities: list[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        if message["type"] == "request":
            # Handle request
            return {
                "type": "response",
                "status": "success",
                "data": {"result": "..."}
            }
        return {"type": "response", "status": "error"}

# Register with system
registry.register("my_agent",
                 capabilities=["custom_capability"],
                 agent=MyCustomAgent("my_agent", ["custom_capability"]))
```

### Define Custom Message Types

```python
class CustomMessage:
    def __init__(self, sender, recipient, action, params):
        self.sender = sender
        self.recipient = recipient
        self.type = "custom_request"
        self.action = action
        self.params = params
        self.conversation_id = generate_conversation_id()
```

### Add Negotiation Logic

```python
def negotiate_task(task, available_agents):
    """Simple negotiation protocol"""
    bids = []

    # Request bids from all capable agents
    for agent in available_agents:
        bid = send_message(agent, {
            "type": "bid_request",
            "task": task
        })
        bids.append(bid)

    # Select best bid (lowest cost, fastest time, etc.)
    winner = min(bids, key=lambda b: b["cost"] + b["time"])

    # Award task
    send_message(winner["agent_id"], {
        "type": "task_awarded",
        "task": task
    })

    return winner
```

---

## ⚡ Common Issues & Solutions

### Issue: "Agent not found"
**Solution**: Make sure the agent is registered before sending messages.
```python
# Check if agent exists
if agent_id in registry.agents:
    send_message(agent_id, message)
else:
    alternatives = registry.discover(required_capability)
```

### Issue: Messages arrive out of order
**Solution**: Use sequence numbers in conversation tracking.
```python
message["sequence"] = next_sequence_number()
```

### Issue: Agent doesn't respond (timeout)
**Solution**: Always set timeouts on requests.
```python
response = send_message_with_timeout(
    agent_id,
    message,
    timeout=30  # seconds
)
```

### Issue: Too many messages, system slows down
**Solution**: Use async messaging and batching.
```python
# Batch multiple messages
messages = [msg1, msg2, msg3]
send_batch(recipient, messages)
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Basic Example**: See [src/communication_basic.py](./src/communication_basic.py)
- **Advanced Example**: See [src/communication_advanced.py](./src/communication_advanced.py)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ **Start**: Run the basic example
2. ✅ **Understand**: See how agents send messages and coordinate
3. ✅ **Observe**: Watch the message flow visualization
4. ✅ **Explore**: Run the advanced example with pub-sub
5. ✅ **Experiment**: Modify agent capabilities and see routing change
6. ✅ **Customize**: Add your own agents and message types
7. ✅ **Integrate**: Use A2A communication in your applications

---

## 🌟 Pro Tips

1. **Message Protocol**: Define clear, typed message formats
2. **Discovery**: Use capability-based agent discovery
3. **Timeouts**: Always set timeouts to prevent hanging
4. **Logging**: Log all messages for debugging and analysis
5. **Conversations**: Track related messages with conversation IDs
6. **Error Handling**: Handle agent failures gracefully
7. **Start Simple**: Begin with direct messaging, add complexity later
8. **Visualization**: Use message flow diagrams to understand behavior

---

## 🔍 Quick Reference

### Send a Direct Message
```python
send_message(
    sender="agent_a",
    recipient="agent_b",
    content={"action": "do_something", "params": {...}}
)
```

### Broadcast to Multiple Agents
```python
broadcast_message(
    sender="coordinator",
    content={"announcement": "New task available"}
)
```

### Discover Capable Agents
```python
agents = registry.discover(capability="research")
```

### Track Conversation
```python
conv_id = start_conversation(participants=["agent_a", "agent_b"])
send_message(recipient="agent_b", conversation_id=conv_id, ...)
history = get_conversation(conv_id)
```

### Request with Timeout
```python
response = send_message_with_timeout(
    recipient="agent_x",
    message={...},
    timeout=30
)
```

---

## 📈 Metrics to Monitor

- **Message Latency**: Time from send to receive
- **Response Time**: How long agents take to respond
- **Success Rate**: % of messages delivered successfully
- **Agent Utilization**: How busy each agent is
- **Queue Depth**: How many messages are pending

---

**Happy Agent Communication! 🤖💬**

For questions or issues, refer to the full [README.md](./README.md).
