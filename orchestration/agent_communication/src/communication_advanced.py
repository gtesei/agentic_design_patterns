"""
Agent-to-Agent Communication Pattern: Advanced Implementation

This example demonstrates advanced A2A communication with:
- Pub-sub message bus for broadcasting
- Agent discovery and capability matching
- Negotiation protocol for task assignment
- Asynchronous communication patterns
- Message queue and conversation tracking
- Rich visualization of communication flow

Scenario: Software development team working on a feature
- Product Manager: Defines requirements and assigns tasks
- Developers: Bid on tasks and implement features
- Tester: Tests implementations
- Reviewer: Reviews code quality
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-4")


# --- Enhanced Message Protocol ---


@dataclass
class Message:
    """Enhanced message format with additional metadata"""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    recipient: str = ""  # Can be specific agent or "broadcast"
    message_type: Literal["request", "response", "notification", "bid", "proposal"] = "request"
    content: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    reply_to: Optional[str] = None  # Reference to message being replied to
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher priority messages processed first

    def __str__(self) -> str:
        action = self.content.get("action", self.content.get("event", self.content.get("proposal", "message")))
        return f"[{self.message_type.upper():12}] {self.sender:20} → {self.recipient:20} | {action}"


# --- Pub-Sub Message Broker ---


class PubSubBroker:
    """Topic-based message distribution system"""

    def __init__(self):
        self.topics: Dict[str, List[str]] = defaultdict(list)  # topic -> [subscriber_ids]
        self.message_queues: Dict[str, deque] = defaultdict(deque)  # agent_id -> message_queue
        self.message_history: List[Message] = []

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic"""
        if agent_id not in self.topics[topic]:
            self.topics[topic].append(agent_id)
            print(f"  ✓ Agent '{agent_id}' subscribed to topic '{topic}'")

    def publish(self, topic: str, message: Message):
        """Publish message to all subscribers of a topic"""
        self.message_history.append(message)
        subscribers = self.topics.get(topic, [])

        if subscribers:
            print(f"\n📢 BROADCAST to topic '{topic}': {message.content.get('event', message.content)}")
            for subscriber in subscribers:
                if subscriber != message.sender:  # Don't send to self
                    self.message_queues[subscriber].append(message)
                    print(f"  → Queued for {subscriber}")
        else:
            print(f"\n⚠️  No subscribers for topic '{topic}'")

    def send_direct(self, message: Message):
        """Send direct message to specific agent"""
        self.message_history.append(message)
        self.message_queues[message.recipient].append(message)

    def get_messages(self, agent_id: str) -> List[Message]:
        """Get all pending messages for an agent"""
        messages = list(self.message_queues[agent_id])
        self.message_queues[agent_id].clear()
        return sorted(messages, key=lambda m: (-m.priority, m.timestamp))


# --- Enhanced Agent Registry ---


class AgentRegistry:
    """Registry with capability matching and load tracking"""

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}

    def register(
        self, agent_id: str, capabilities: List[str], metadata: Dict[str, Any] = None, max_load: int = 5
    ):
        """Register an agent with capabilities and load limits"""
        self.agents[agent_id] = {
            "capabilities": capabilities,
            "metadata": metadata or {},
            "status": "available",
            "current_load": 0,
            "max_load": max_load,
            "registered_at": datetime.now(),
        }
        print(f"✓ Registered '{agent_id}' | Capabilities: {capabilities} | Max load: {max_load}")

    def discover(self, capability: str, available_only: bool = True) -> List[str]:
        """Find agents with a capability, optionally filtering by availability"""
        candidates = []
        for agent_id, info in self.agents.items():
            if capability in info["capabilities"]:
                if not available_only or (info["status"] == "available" and info["current_load"] < info["max_load"]):
                    candidates.append(agent_id)
        return candidates

    def update_load(self, agent_id: str, delta: int):
        """Update agent's current load"""
        if agent_id in self.agents:
            self.agents[agent_id]["current_load"] += delta
            if self.agents[agent_id]["current_load"] >= self.agents[agent_id]["max_load"]:
                self.agents[agent_id]["status"] = "busy"
            else:
                self.agents[agent_id]["status"] = "available"

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get agent information"""
        return self.agents.get(agent_id, {})


# --- Base Agent with Negotiation Support ---


class AdvancedAgent:
    """Base agent class with pub-sub and negotiation support"""

    def __init__(self, agent_id: str, capabilities: List[str], system_prompt: str, broker: PubSubBroker):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.system_prompt = system_prompt
        self.broker = broker
        self.active_tasks: List[str] = []
        self.workload = 0

    def subscribe_to_topics(self, topics: List[str]):
        """Subscribe to relevant topics"""
        for topic in topics:
            self.broker.subscribe(self.agent_id, topic)

    def broadcast(self, topic: str, message_type: str, content: Dict[str, Any], conversation_id: str):
        """Broadcast message to a topic"""
        message = Message(
            sender=self.agent_id,
            recipient="broadcast",
            message_type=message_type,
            content={**content, "topic": topic},
            conversation_id=conversation_id,
        )
        self.broker.publish(topic, message)

    def send_direct(self, recipient: str, message_type: str, content: Dict[str, Any], conversation_id: str, reply_to: str = None):
        """Send direct message to another agent"""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id,
            reply_to=reply_to,
        )
        self.broker.send_direct(message)
        print(f"\n✉️  {message}")

    def process_messages(self) -> List[Message]:
        """Process all pending messages"""
        messages = self.broker.get_messages(self.agent_id)
        responses = []

        for message in messages:
            print(f"\n📬 {self.agent_id} received: {message}")
            response = self.handle_message(message)
            if response:
                responses.append(response)

        return responses

    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message - to be overridden by subclasses"""
        raise NotImplementedError

    def can_accept_task(self, task_complexity: int) -> bool:
        """Check if agent can accept a new task"""
        return self.workload + task_complexity <= 10


# --- Specialized Agents ---


class ProductManagerAgent(AdvancedAgent):
    """Product Manager agent that defines requirements and assigns tasks"""

    def __init__(self, broker: PubSubBroker):
        super().__init__(
            agent_id="product_manager",
            capabilities=["planning", "requirements", "task_assignment"],
            system_prompt="""You are a Product Manager responsible for defining clear requirements
            and coordinating development tasks. Break down features into clear, actionable tasks.""",
            broker=broker,
        )
        self.pending_bids: Dict[str, List[Dict]] = {}  # task_id -> [bids]

    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle task completion notifications and bids"""
        if message.message_type == "bid":
            task_id = message.content.get("task_id")
            if task_id not in self.pending_bids:
                self.pending_bids[task_id] = []

            self.pending_bids[task_id].append(
                {
                    "agent_id": message.sender,
                    "estimated_time": message.content.get("estimated_time", 5),
                    "confidence": message.content.get("confidence", 0.5),
                    "cost": message.content.get("cost", 10),
                }
            )
            print(f"  💰 Bid received from {message.sender}: {message.content}")

        elif message.message_type == "notification":
            event = message.content.get("event")
            if event == "task_completed":
                print(f"  ✅ Task completed by {message.sender}")

        return None

    def create_and_assign_task(self, task: Dict[str, Any], conversation_id: str, registry: AgentRegistry):
        """Create a task and run negotiation to assign it"""
        task_id = str(uuid4())
        task["task_id"] = task_id

        print(f"\n{'='*100}")
        print(f"📋 NEW TASK: {task['title']}")
        print(f"{'='*100}")
        print(f"Description: {task['description']}")
        print(f"Required capability: {task['required_capability']}")
        print(f"Task ID: {task_id}")

        # Step 1: Discover capable agents
        print(f"\n🔍 Discovering agents with capability '{task['required_capability']}'...")
        capable_agents = registry.discover(task["required_capability"])
        print(f"Found {len(capable_agents)} capable agents: {capable_agents}")

        if not capable_agents:
            print("❌ No capable agents available!")
            return None

        # Step 2: Broadcast call for proposals
        print(f"\n📢 Broadcasting Call for Proposals...")
        self.broadcast(
            topic="task_assignment",
            message_type="proposal",
            content={"event": "call_for_proposals", "task": task},
            conversation_id=conversation_id,
        )

        # Step 3: Collect bids (simulated - in real system would wait for async responses)
        print(f"\n⏳ Collecting bids...")
        self.pending_bids[task_id] = []

        # Trigger agents to send bids
        return task_id

    def select_winner(self, task_id: str, conversation_id: str) -> Optional[str]:
        """Select winning bid and award task"""
        bids = self.pending_bids.get(task_id, [])

        if not bids:
            print("❌ No bids received!")
            return None

        print(f"\n📊 Evaluating {len(bids)} bids...")
        for bid in bids:
            score = self._score_bid(bid)
            bid["score"] = score
            print(
                f"  {bid['agent_id']:20} | Time: {bid['estimated_time']:2}d | "
                f"Confidence: {bid['confidence']:.2f} | Cost: ${bid['cost']:3} | Score: {score:.2f}"
            )

        # Select best bid
        winner = max(bids, key=lambda b: b["score"])
        print(f"\n🏆 Winner: {winner['agent_id']} (Score: {winner['score']:.2f})")

        # Award task
        self.send_direct(
            recipient=winner["agent_id"],
            message_type="request",
            content={"action": "task_awarded", "task_id": task_id},
            conversation_id=conversation_id,
        )

        return winner["agent_id"]

    def _score_bid(self, bid: Dict) -> float:
        """Score a bid based on time, confidence, and cost"""
        # Lower time is better, higher confidence is better, lower cost is better
        time_score = 10 / max(bid["estimated_time"], 1)  # Inverse of time
        confidence_score = bid["confidence"] * 10
        cost_score = 100 / max(bid["cost"], 1)  # Inverse of cost

        # Weighted combination
        return (time_score * 0.3) + (confidence_score * 0.5) + (cost_score * 0.2)


class DeveloperAgent(AdvancedAgent):
    """Developer agent that bids on and implements tasks"""

    def __init__(self, agent_id: str, broker: PubSubBroker, skill_level: float = 0.8):
        super().__init__(
            agent_id=agent_id,
            capabilities=["development", "coding", "implementation"],
            system_prompt=f"""You are a software developer with skill level {skill_level}.
            You implement features based on requirements and write clean, well-tested code.""",
            broker=broker,
        )
        self.skill_level = skill_level

    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle task proposals and assignments"""
        if message.message_type == "proposal" and message.content.get("event") == "call_for_proposals":
            # Evaluate if we want to bid
            task = message.content.get("task", {})
            return self._submit_bid(task, message.conversation_id)

        elif message.message_type == "request" and message.content.get("action") == "task_awarded":
            task_id = message.content.get("task_id")
            return self._implement_task(task_id, message.conversation_id)

        return None

    def _submit_bid(self, task: Dict, conversation_id: str) -> Optional[Message]:
        """Submit a bid for a task"""
        if not self.can_accept_task(task.get("complexity", 5)):
            print(f"  ⚠️  {self.agent_id} is too busy to bid")
            return None

        # Calculate bid parameters
        estimated_time = max(1, int(5 / self.skill_level))
        confidence = self.skill_level * (1 - self.workload / 10)
        cost = int(100 * self.skill_level * estimated_time)

        print(f"\n💼 {self.agent_id} preparing bid...")

        self.send_direct(
            recipient="product_manager",
            message_type="bid",
            content={
                "task_id": task["task_id"],
                "estimated_time": estimated_time,
                "confidence": confidence,
                "cost": cost,
            },
            conversation_id=conversation_id,
        )

        return None

    def _implement_task(self, task_id: str, conversation_id: str) -> Optional[Message]:
        """Implement the awarded task"""
        print(f"\n🔨 {self.agent_id} implementing task {task_id}...")

        # Use LLM to generate implementation
        response = llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=f"Implement a user authentication feature. Provide a brief technical description of the implementation (2-3 sentences)."
                ),
            ]
        )

        implementation = response.content
        print(f"  ✓ Implementation complete: {implementation[:100]}...")

        self.active_tasks.append(task_id)
        self.workload += 5

        # Notify completion and request testing
        self.broadcast(
            topic="development",
            message_type="notification",
            content={"event": "implementation_complete", "task_id": task_id, "implementation": implementation},
            conversation_id=conversation_id,
        )

        return None


class TesterAgent(AdvancedAgent):
    """Tester agent that tests implementations"""

    def __init__(self, broker: PubSubBroker):
        super().__init__(
            agent_id="tester",
            capabilities=["testing", "qa", "validation"],
            system_prompt="""You are a QA tester responsible for testing implementations
            and identifying bugs or issues.""",
            broker=broker,
        )

    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle implementation completion notifications"""
        if message.message_type == "notification" and message.content.get("event") == "implementation_complete":
            task_id = message.content.get("task_id")
            implementation = message.content.get("implementation", "")
            return self._test_implementation(task_id, implementation, message.sender, message.conversation_id)

        return None

    def _test_implementation(self, task_id: str, implementation: str, developer: str, conversation_id: str):
        """Test an implementation"""
        print(f"\n🧪 {self.agent_id} testing implementation from {developer}...")

        # Use LLM to generate test results
        response = llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=f"Test this implementation:\n\n{implementation}\n\nProvide test results: pass/fail and any issues found (2-3 sentences)."
                ),
            ]
        )

        test_results = response.content
        passed = "pass" in test_results.lower() and "fail" not in test_results.lower()

        print(f"  {'✅' if passed else '❌'} Tests {'passed' if passed else 'failed'}")

        # Send results to developer
        self.send_direct(
            recipient=developer,
            message_type="response",
            content={"action": "test_results", "task_id": task_id, "passed": passed, "results": test_results},
            conversation_id=conversation_id,
        )

        # If passed, notify for review
        if passed:
            self.broadcast(
                topic="development",
                message_type="notification",
                content={"event": "ready_for_review", "task_id": task_id, "developer": developer},
                conversation_id=conversation_id,
            )


class ReviewerAgent(AdvancedAgent):
    """Reviewer agent that performs code reviews"""

    def __init__(self, broker: PubSubBroker):
        super().__init__(
            agent_id="reviewer",
            capabilities=["review", "code_review", "quality_assurance"],
            system_prompt="""You are a code reviewer responsible for ensuring code quality,
            best practices, and maintainability.""",
            broker=broker,
        )

    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle review requests"""
        if message.message_type == "notification" and message.content.get("event") == "ready_for_review":
            task_id = message.content.get("task_id")
            developer = message.content.get("developer")
            return self._review_code(task_id, developer, message.conversation_id)

        return None

    def _review_code(self, task_id: str, developer: str, conversation_id: str):
        """Perform code review"""
        print(f"\n👀 {self.agent_id} reviewing code from {developer}...")

        # Use LLM to generate review
        response = llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=f"Review a user authentication implementation. Provide feedback on code quality, security, and best practices (2-3 sentences)."
                ),
            ]
        )

        review_feedback = response.content
        approved = "approve" in review_feedback.lower() or "good" in review_feedback.lower()

        print(f"  {'✅' if approved else '⚠️ '} Review {'approved' if approved else 'needs changes'}")

        # Send feedback
        self.send_direct(
            recipient=developer,
            message_type="response",
            content={
                "action": "review_feedback",
                "task_id": task_id,
                "approved": approved,
                "feedback": review_feedback,
            },
            conversation_id=conversation_id,
        )

        if approved:
            # Broadcast completion
            self.broadcast(
                topic="task_assignment",
                message_type="notification",
                content={"event": "task_completed", "task_id": task_id},
                conversation_id=conversation_id,
            )


# --- Advanced Coordinator ---


class AdvancedA2ACoordinator:
    """Advanced coordinator with pub-sub and negotiation"""

    def __init__(self):
        self.broker = PubSubBroker()
        self.registry = AgentRegistry()
        self.conversation_id = str(uuid4())
        self.agents: Dict[str, AdvancedAgent] = {}

        self._initialize_system()

    def _initialize_system(self):
        """Initialize all agents and subscriptions"""
        print(f"\n{'='*100}")
        print("INITIALIZING AGENT-TO-AGENT COMMUNICATION SYSTEM")
        print(f"{'='*100}\n")

        # Create agents
        pm = ProductManagerAgent(self.broker)
        dev1 = DeveloperAgent("developer_1", self.broker, skill_level=0.9)
        dev2 = DeveloperAgent("developer_2", self.broker, skill_level=0.75)
        tester = TesterAgent(self.broker)
        reviewer = ReviewerAgent(self.broker)

        self.agents = {
            "product_manager": pm,
            "developer_1": dev1,
            "developer_2": dev2,
            "tester": tester,
            "reviewer": reviewer,
        }

        # Register agents
        print("\n📝 REGISTERING AGENTS")
        print("─" * 100)
        self.registry.register("product_manager", pm.capabilities, max_load=10)
        self.registry.register("developer_1", dev1.capabilities, max_load=3)
        self.registry.register("developer_2", dev2.capabilities, max_load=3)
        self.registry.register("tester", tester.capabilities, max_load=5)
        self.registry.register("reviewer", reviewer.capabilities, max_load=5)

        # Subscribe to topics
        print("\n📡 SETTING UP PUB-SUB SUBSCRIPTIONS")
        print("─" * 100)
        pm.subscribe_to_topics(["task_assignment"])
        dev1.subscribe_to_topics(["task_assignment", "development"])
        dev2.subscribe_to_topics(["task_assignment", "development"])
        tester.subscribe_to_topics(["development"])
        reviewer.subscribe_to_topics(["development"])

    def run_workflow(self, feature_description: str):
        """Run the complete software development workflow"""
        print(f"\n\n{'='*100}")
        print("STARTING SOFTWARE DEVELOPMENT WORKFLOW")
        print(f"{'='*100}")
        print(f"Feature: {feature_description}")
        print(f"Conversation ID: {self.conversation_id}")
        print(f"{'='*100}\n")

        pm = self.agents["product_manager"]

        # Phase 1: Task creation and assignment
        task = {
            "title": feature_description,
            "description": f"Implement {feature_description} with proper authentication, security, and testing",
            "required_capability": "development",
            "complexity": 5,
        }

        task_id = pm.create_and_assign_task(task, self.conversation_id, self.registry)

        # Phase 2: Process messages (developers submit bids)
        print(f"\n{'─'*100}")
        print("PHASE 1: BID COLLECTION")
        print(f"{'─'*100}")

        for agent_id in ["developer_1", "developer_2"]:
            self.agents[agent_id].process_messages()

        # Phase 3: Award task
        print(f"\n{'─'*100}")
        print("PHASE 2: TASK AWARD")
        print(f"{'─'*100}")

        winner = pm.select_winner(task_id, self.conversation_id)

        # Phase 4: Implementation
        print(f"\n{'─'*100}")
        print("PHASE 3: IMPLEMENTATION")
        print(f"{'─'*100}")

        self.agents[winner].process_messages()

        # Phase 5: Testing
        print(f"\n{'─'*100}")
        print("PHASE 4: TESTING")
        print(f"{'─'*100}")

        self.agents["tester"].process_messages()

        # Phase 6: Review
        print(f"\n{'─'*100}")
        print("PHASE 5: CODE REVIEW")
        print(f"{'─'*100}")

        self.agents["reviewer"].process_messages()

        # Phase 7: Final notifications
        self.agents["product_manager"].process_messages()

        # Display summary
        self._display_summary()

    def _display_summary(self):
        """Display workflow summary"""
        print(f"\n\n{'='*100}")
        print("WORKFLOW SUMMARY")
        print(f"{'='*100}\n")

        print(f"Total messages exchanged: {len(self.broker.message_history)}")
        print(f"\nMessage flow visualization:")
        print("─" * 100)

        for i, msg in enumerate(self.broker.message_history, 1):
            sender = msg.sender[:18]
            recipient = msg.recipient[:18] if msg.recipient != "broadcast" else "ALL SUBSCRIBERS"
            msg_type = msg.message_type[:10]
            print(f"{i:3}. {sender:20} → {recipient:20} [{msg_type:12}]")

        print("\n" + "─" * 100)
        print("AGENT STATUS")
        print("─" * 100)

        for agent_id, agent in self.agents.items():
            info = self.registry.get_agent_info(agent_id)
            status = info.get("status", "unknown")
            load = info.get("current_load", 0)
            max_load = info.get("max_load", 0)
            print(f"{agent_id:20} | Status: {status:10} | Load: {load}/{max_load}")


# --- Main Execution ---


def main():
    """Run the advanced A2A communication example"""
    print(
        """
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║        Agent-to-Agent Communication - Advanced Implementation                  ║
    ║                                                                                ║
    ║  Demonstrates pub-sub messaging, negotiation, and dynamic coordination         ║
    ║  Scenario: Software development team with task bidding                        ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Create coordinator
    coordinator = AdvancedA2ACoordinator()

    # Run workflow
    coordinator.run_workflow("User Authentication Feature")

    print(
        """

    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                          Example Complete!                                     ║
    ║                                                                                ║
    ║  The Advanced A2A Communication system demonstrated:                          ║
    ║  • Pub-sub message broker with topic-based routing                           ║
    ║  • Dynamic agent discovery and capability matching                           ║
    ║  • Negotiation protocol with competitive bidding                             ║
    ║  • Asynchronous message queues and processing                                ║
    ║  • Multi-phase workflow coordination                                         ║
    ║  • Agent load tracking and availability management                           ║
    ║  • Rich communication flow visualization                                     ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    )


if __name__ == "__main__":
    main()
