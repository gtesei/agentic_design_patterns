"""
Agent-to-Agent Communication Pattern: Basic Implementation

This example demonstrates basic A2A communication with:
- Direct message passing between agents
- Simple message protocol (request, response, notification)
- Agent registry and capability discovery
- Sequential and parallel workflows
- Message logging and visualization

Scenario: Research team collaborating on a report about AI safety
- Researcher: Gathers information from sources
- Writer: Drafts content based on research
- Reviewer: Reviews and provides feedback
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")


# --- Message Protocol ---


@dataclass
class Message:
    """Standard message format for agent communication"""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: Literal["request", "response", "notification"] = "request"
    content: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"[{self.message_type.upper()}] {self.sender} → {self.recipient}: {self.content.get('action', self.content.get('status', 'notification'))}"


# --- Agent Registry ---


class AgentRegistry:
    """Registry for agent discovery and capability matching"""

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.message_log: List[Message] = []

    def register(self, agent_id: str, capabilities: List[str], metadata: Dict[str, Any] = None):
        """Register an agent with its capabilities"""
        self.agents[agent_id] = {
            "capabilities": capabilities,
            "metadata": metadata or {},
            "status": "available",
            "registered_at": datetime.now(),
        }
        print(f"✓ Registered agent '{agent_id}' with capabilities: {capabilities}")

    def discover(self, capability: str) -> List[str]:
        """Find agents with a specific capability"""
        return [
            agent_id
            for agent_id, info in self.agents.items()
            if capability in info["capabilities"] and info["status"] == "available"
        ]

    def log_message(self, message: Message):
        """Log a message for tracking"""
        self.message_log.append(message)

    def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """Get all messages in a conversation"""
        return [msg for msg in self.message_log if msg.conversation_id == conversation_id]


# --- Agent Base Class ---


class Agent:
    """Base class for agents in the A2A system"""

    def __init__(self, agent_id: str, capabilities: List[str], system_prompt: str):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.system_prompt = system_prompt
        self.inbox: List[Message] = []

    def receive_message(self, message: Message):
        """Receive and store a message"""
        self.inbox.append(message)

    def process_message(self, message: Message) -> Message:
        """Process a message and generate response - to be overridden"""
        raise NotImplementedError

    def send_message(
        self,
        recipient: str,
        message_type: Literal["request", "response", "notification"],
        content: Dict[str, Any],
        conversation_id: str,
    ) -> Message:
        """Create and send a message"""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id,
        )
        return message


# --- Specialized Agents ---


class ResearchAgent(Agent):
    """Agent specialized in research and information gathering"""

    def __init__(self):
        super().__init__(
            agent_id="researcher",
            capabilities=["research", "fact_checking", "information_gathering"],
            system_prompt="""You are a research agent specialized in gathering and synthesizing information.
            Your role is to conduct thorough research on topics and provide comprehensive, accurate findings.
            Focus on key facts, statistics, and important concepts.""",
        )

    def process_message(self, message: Message) -> Message:
        """Process research requests"""
        if message.message_type == "request" and message.content.get("action") == "research":
            topic = message.content.get("topic", "")

            # Use LLM to generate research findings
            response = llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=f"Research the topic: '{topic}'. Provide key findings, facts, and important concepts in 3-4 paragraphs."
                    ),
                ]
            )

            return self.send_message(
                recipient=message.sender,
                message_type="response",
                content={"status": "success", "findings": response.content, "topic": topic},
                conversation_id=message.conversation_id,
            )

        return self.send_message(
            recipient=message.sender,
            message_type="response",
            content={"status": "error", "message": "Unknown action"},
            conversation_id=message.conversation_id,
        )


class WriterAgent(Agent):
    """Agent specialized in writing and content creation"""

    def __init__(self):
        super().__init__(
            agent_id="writer",
            capabilities=["writing", "content_creation", "summarization"],
            system_prompt="""You are a writing agent specialized in creating clear, engaging content.
            Your role is to transform research findings into well-written reports and summaries.
            Write in a professional yet accessible style.""",
        )

    def process_message(self, message: Message) -> Message:
        """Process writing requests"""
        if message.message_type == "request" and message.content.get("action") == "write":
            research_findings = message.content.get("research_findings", "")
            topic = message.content.get("topic", "")

            # Use LLM to generate written content
            response = llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=f"Write a comprehensive report on '{topic}' based on these research findings:\n\n{research_findings}\n\nCreate a well-structured report with an introduction, main points, and conclusion."
                    ),
                ]
            )

            return self.send_message(
                recipient=message.sender,
                message_type="response",
                content={"status": "success", "report": response.content, "topic": topic},
                conversation_id=message.conversation_id,
            )

        return self.send_message(
            recipient=message.sender,
            message_type="response",
            content={"status": "error", "message": "Unknown action"},
            conversation_id=message.conversation_id,
        )


class ReviewerAgent(Agent):
    """Agent specialized in reviewing and quality checking"""

    def __init__(self):
        super().__init__(
            agent_id="reviewer",
            capabilities=["review", "quality_check", "feedback"],
            system_prompt="""You are a reviewer agent specialized in quality checking and providing constructive feedback.
            Your role is to review content for accuracy, clarity, completeness, and quality.
            Provide specific, actionable feedback.""",
        )

    def process_message(self, message: Message) -> Message:
        """Process review requests"""
        if message.message_type == "request" and message.content.get("action") == "review":
            report = message.content.get("report", "")
            topic = message.content.get("topic", "")

            # Use LLM to review content
            response = llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=f"Review this report on '{topic}':\n\n{report}\n\nProvide feedback on:\n1. Accuracy and completeness\n2. Clarity and structure\n3. Overall quality\n4. Suggestions for improvement\n\nIf the report is good, approve it. Otherwise, provide specific revisions needed."
                    ),
                ]
            )

            # Determine if approved
            is_approved = "approve" in response.content.lower() and "revision" not in response.content.lower()

            return self.send_message(
                recipient=message.sender,
                message_type="response",
                content={"status": "approved" if is_approved else "needs_revision", "feedback": response.content},
                conversation_id=message.conversation_id,
            )

        return self.send_message(
            recipient=message.sender,
            message_type="response",
            content={"status": "error", "message": "Unknown action"},
            conversation_id=message.conversation_id,
        )


# --- Coordinator (Orchestrates Agent Communication) ---


class CommunicationState(TypedDict):
    """State for the communication workflow"""

    topic: str
    conversation_id: str
    research_findings: str
    report: str
    review_feedback: str
    status: str
    messages: List[Message]


class A2ACoordinator:
    """Coordinates agent-to-agent communication"""

    def __init__(self):
        self.registry = AgentRegistry()
        self.agents: Dict[str, Agent] = {}
        self.conversation_id = str(uuid4())

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Create and register all agents"""
        # Create agents
        researcher = ResearchAgent()
        writer = WriterAgent()
        reviewer = ReviewerAgent()

        # Register agents
        self.agents = {"researcher": researcher, "writer": writer, "reviewer": reviewer}

        for agent_id, agent in self.agents.items():
            self.registry.register(agent_id, agent.capabilities)

    def send_message(self, message: Message) -> Message:
        """Send a message to an agent and get response"""
        self.registry.log_message(message)
        print(f"\n📨 {message}")

        # Deliver to recipient
        if message.recipient in self.agents:
            agent = self.agents[message.recipient]
            agent.receive_message(message)
            response = agent.process_message(message)
            self.registry.log_message(response)
            print(f"📬 {response}")
            return response

        raise ValueError(f"Agent '{message.recipient}' not found")

    def run_workflow(self, topic: str):
        """Run the complete research → write → review workflow"""
        print(f"\n{'='*100}")
        print(f"Starting A2A Communication Workflow")
        print(f"Topic: {topic}")
        print(f"Conversation ID: {self.conversation_id}")
        print(f"{'='*100}\n")

        # Step 1: Research phase
        print("\n" + "─" * 100)
        print("PHASE 1: RESEARCH")
        print("─" * 100)

        research_msg = Message(
            sender="coordinator",
            recipient="researcher",
            message_type="request",
            content={"action": "research", "topic": topic},
            conversation_id=self.conversation_id,
        )

        research_response = self.send_message(research_msg)
        research_findings = research_response.content.get("findings", "")

        # Step 2: Writing phase
        print("\n" + "─" * 100)
        print("PHASE 2: WRITING")
        print("─" * 100)

        write_msg = Message(
            sender="coordinator",
            recipient="writer",
            message_type="request",
            content={"action": "write", "topic": topic, "research_findings": research_findings},
            conversation_id=self.conversation_id,
        )

        write_response = self.send_message(write_msg)
        report = write_response.content.get("report", "")

        # Step 3: Review phase
        print("\n" + "─" * 100)
        print("PHASE 3: REVIEW")
        print("─" * 100)

        review_msg = Message(
            sender="coordinator",
            recipient="reviewer",
            message_type="request",
            content={"action": "review", "topic": topic, "report": report},
            conversation_id=self.conversation_id,
        )

        review_response = self.send_message(review_msg)
        review_status = review_response.content.get("status", "")
        review_feedback = review_response.content.get("feedback", "")

        # Display final results
        self._display_results(topic, research_findings, report, review_status, review_feedback)

        return {
            "topic": topic,
            "research_findings": research_findings,
            "report": report,
            "review_status": review_status,
            "review_feedback": review_feedback,
        }

    def _display_results(self, topic, research_findings, report, review_status, review_feedback):
        """Display the final results"""
        print("\n" + "=" * 100)
        print("FINAL RESULTS")
        print("=" * 100)

        print(f"\n📊 TOPIC: {topic}\n")

        print("─" * 100)
        print("1. RESEARCH FINDINGS")
        print("─" * 100)
        print(research_findings)

        print("\n" + "─" * 100)
        print("2. WRITTEN REPORT")
        print("─" * 100)
        print(report)

        print("\n" + "─" * 100)
        print("3. REVIEW FEEDBACK")
        print("─" * 100)
        print(f"Status: {review_status.upper()}")
        print(f"\n{review_feedback}")

        print("\n" + "─" * 100)
        print("COMMUNICATION SUMMARY")
        print("─" * 100)
        history = self.registry.get_conversation_history(self.conversation_id)
        print(f"Total messages exchanged: {len(history)}")
        print(f"Participants: {set(msg.sender for msg in history) | set(msg.recipient for msg in history)}")

        print("\nMessage flow:")
        for i, msg in enumerate(history, 1):
            print(f"  {i}. {msg.sender:15} → {msg.recipient:15} [{msg.message_type:12}]")


# --- Main Execution ---


def main():
    """Run the basic A2A communication example"""
    print(
        """
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║          Agent-to-Agent Communication - Basic Implementation                   ║
    ║                                                                                ║
    ║  Demonstrates direct message passing between specialized agents                ║
    ║  Scenario: Research team collaborating on a report                            ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Create coordinator
    coordinator = A2ACoordinator()

    # Run workflow
    result = coordinator.run_workflow(topic="AI Safety and Alignment")

    print(
        """
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                          Example Complete!                                     ║
    ║                                                                                ║
    ║  The A2A Communication system demonstrated:                                   ║
    ║  • Agent registration and capability discovery                                ║
    ║  • Direct message passing between agents                                      ║
    ║  • Sequential workflow coordination                                           ║
    ║  • Message logging and conversation tracking                                  ║
    ║  • Specialized agents working together on a complex task                      ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    )


if __name__ == "__main__":
    main()
