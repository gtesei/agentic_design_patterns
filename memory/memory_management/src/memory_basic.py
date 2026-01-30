"""
Basic Memory Management Demo

Demonstrates:
- Conversation buffer (last N messages)
- Automatic summarization when buffer is full
- Memory persistence (save/load)
- Simple retrieval by recency
"""

import os
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables from project root
load_dotenv("../../.env")


class ConversationMemory:
    """Basic memory system with buffer and summarization"""

    def __init__(self, buffer_size: int = 10, llm: Optional[ChatOpenAI] = None):
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)
        self.summaries: List[str] = []
        self.full_history: List[Dict[str, str]] = []
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to memory"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        # Add to buffer
        self.buffer.append(message)

        # Add to full history
        self.full_history.append(message)

        print(f"  [Memory] Added {role} message (buffer: {len(self.buffer)}/{self.buffer_size})")

        # Check if buffer is full and should be summarized
        if len(self.buffer) == self.buffer_size and len(self.full_history) > self.buffer_size:
            self._summarize_buffer()

    def _summarize_buffer(self) -> None:
        """Summarize the buffer to make room for new messages"""
        print("\n  [Memory] Buffer full! Creating summary of conversation...")

        # Get messages to summarize (exclude the most recent few)
        messages_to_summarize = list(self.buffer)[: self.buffer_size // 2]

        # Create summary
        summary_prompt = self._create_summary_prompt(messages_to_summarize)
        response = self.llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content

        # Store summary
        self.summaries.append(
            {
                "summary": summary,
                "message_count": len(messages_to_summarize),
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(f"  [Memory] Summary created: '{summary[:80]}...'")

    def _create_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Create prompt for summarizing conversation"""
        conversation_text = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
        )

        return f"""Summarize the following conversation in 2-3 sentences, capturing the key points,
topics discussed, and any important facts or preferences mentioned:

{conversation_text}

Summary:"""

    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get conversation context (summaries + recent buffer)"""
        context = []

        # Add summaries as system messages
        for summary_obj in self.summaries:
            context.append(
                {
                    "role": "system",
                    "content": f"[Previous conversation summary]: {summary_obj['summary']}",
                }
            )

        # Add recent messages from buffer
        recent_messages = list(self.buffer)[-max_messages:]
        context.extend(recent_messages)

        return context

    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """Get the N most recent messages"""
        return list(self.buffer)[-count:]

    def save(self, filepath: str) -> None:
        """Save memory to disk"""
        data = {
            "buffer": list(self.buffer),
            "summaries": self.summaries,
            "full_history": self.full_history,
            "metadata": {
                "buffer_size": self.buffer_size,
                "saved_at": datetime.now().isoformat(),
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n  [Memory] Saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load memory from disk"""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.buffer = deque(data["buffer"], maxlen=self.buffer_size)
        self.summaries = data["summaries"]
        self.full_history = data["full_history"]

        print(f"\n  [Memory] Loaded from {filepath}")

    def display_stats(self) -> None:
        """Display memory statistics"""
        print("\n" + "=" * 60)
        print("MEMORY STATISTICS")
        print("=" * 60)
        print(f"Buffer size: {len(self.buffer)}/{self.buffer_size}")
        print(f"Total messages: {len(self.full_history)}")
        print(f"Summaries: {len(self.summaries)}")
        print(f"Oldest message: {self.full_history[0]['timestamp'] if self.full_history else 'N/A'}")
        print(
            f"Newest message: {self.full_history[-1]['timestamp'] if self.full_history else 'N/A'}"
        )
        print("=" * 60)

    def display_buffer(self) -> None:
        """Display current buffer contents"""
        print("\n" + "-" * 60)
        print("CURRENT BUFFER")
        print("-" * 60)
        for i, msg in enumerate(self.buffer, 1):
            role_color = "\033[94m" if msg["role"] == "user" else "\033[92m"
            reset = "\033[0m"
            print(f"{i}. {role_color}{msg['role'].upper()}{reset}: {msg['content'][:80]}...")
        print("-" * 60)

    def display_summaries(self) -> None:
        """Display all summaries"""
        if not self.summaries:
            print("\nNo summaries yet.")
            return

        print("\n" + "-" * 60)
        print("CONVERSATION SUMMARIES")
        print("-" * 60)
        for i, summary_obj in enumerate(self.summaries, 1):
            print(f"\nSummary {i} ({summary_obj['message_count']} messages):")
            print(f"  {summary_obj['summary']}")
        print("-" * 60)


class BasicMemoryAgent:
    """Simple conversational agent with memory"""

    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def chat(self, user_message: str) -> str:
        """Process user message and generate response"""
        # Add user message to memory
        self.memory.add_message("user", user_message)

        # Get conversation context
        context = self.memory.get_context()

        # Convert to LangChain messages
        messages = []
        for msg in context:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Generate response
        response = self.llm.invoke(messages)
        assistant_message = response.content

        # Add assistant response to memory
        self.memory.add_message("assistant", assistant_message)

        return assistant_message


def demonstrate_basic_memory():
    """Demonstrate basic memory management"""
    print("\n" + "=" * 60)
    print("BASIC MEMORY MANAGEMENT DEMO")
    print("=" * 60)
    print("\nProblem: Maintain conversation history across multiple turns")
    print("Solution: Buffer memory + automatic summarization\n")

    # Initialize memory with small buffer to trigger summarization
    memory = ConversationMemory(buffer_size=6)
    agent = BasicMemoryAgent(memory)

    # Scenario: Personal assistant conversation
    print("\n--- Conversation Scenario: Personal Assistant ---\n")

    conversations = [
        ("My name is Alice and I'm a software engineer at TechCorp", "Nice to meet you!"),
        (
            "I'm working on a Python project about machine learning",
            "That sounds interesting!",
        ),
        ("I love hiking on weekends, especially in the mountains", "Great hobby!"),
        (
            "My favorite food is Italian cuisine, particularly pasta",
            "Delicious choice!",
        ),
        (
            "I have a meeting tomorrow at 2pm with the data science team",
            "I'll remember that.",
        ),
        ("I'm also learning Spanish in my free time", "Buena suerte!"),
        # These next messages will trigger summarization
        (
            "What do you remember about me?",
            "Let me recall...",
        ),
        ("Tell me about my work", "Let me check..."),
    ]

    for i, (user_msg, _) in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}")
        print(f"{'='*60}")
        print(f"\n👤 USER: {user_msg}")

        response = agent.chat(user_msg)

        print(f"\n🤖 ASSISTANT: {response}")

        # Show buffer after key turns
        if i in [3, 6, 8]:
            memory.display_buffer()

        # Show summaries after they're created
        if i == 6:
            memory.display_summaries()

    # Display final memory stats
    memory.display_stats()

    # Demonstrate memory persistence
    print("\n\n--- Demonstrating Memory Persistence ---\n")
    save_path = "./memory_data/conversation_basic.json"
    memory.save(save_path)

    # Load and verify
    print("\nLoading memory from disk...")
    new_memory = ConversationMemory(buffer_size=6)
    new_memory.load(save_path)
    new_memory.display_stats()

    # Continue conversation with loaded memory
    print("\n\n--- Continuing Conversation with Loaded Memory ---\n")
    new_agent = BasicMemoryAgent(new_memory)

    print(f"\n{'='*60}")
    print("Turn 9 (After Reload)")
    print(f"{'='*60}")
    user_msg = "What was my favorite food again?"
    print(f"\n👤 USER: {user_msg}")

    response = new_agent.chat(user_msg)
    print(f"\n🤖 ASSISTANT: {response}")

    # Show that agent remembers (from summary or buffer)
    print("\n✅ Agent remembered information from earlier in the conversation!")

    # Display retrieval visualization
    print("\n\n--- Memory Retrieval Visualization ---\n")
    context = new_memory.get_context()

    print("Context provided to LLM:")
    print("-" * 60)
    for i, msg in enumerate(context, 1):
        role_icon = "📋" if msg["role"] == "system" else ("👤" if msg["role"] == "user" else "🤖")
        role_str = msg["role"].upper()
        content_preview = msg["content"][:100]
        if len(msg["content"]) > 100:
            content_preview += "..."
        print(f"{i}. {role_icon} {role_str}: {content_preview}")
    print("-" * 60)

    print("\n\n--- Key Features Demonstrated ---\n")
    print("✅ Buffer Memory: Recent messages kept in sliding window")
    print("✅ Automatic Summarization: Old messages condensed when buffer full")
    print("✅ Memory Persistence: Save/load conversation state")
    print("✅ Context Assembly: Summaries + buffer provided to LLM")
    print("✅ Efficient Retrieval: Fast access to relevant context")


if __name__ == "__main__":
    demonstrate_basic_memory()
