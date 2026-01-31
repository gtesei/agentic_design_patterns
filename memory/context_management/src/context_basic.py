"""
Basic Context Management Demo

Demonstrates:
- Sliding window for conversation history
- Simple compression via summarization
- Token counting with tiktoken
- Priority-based inclusion
- Visual context packing diagram

Problem: How do we fit a long conversation into a limited context window?
Solution: Use sliding window + summarization to keep recent messages and compress old ones.
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import os
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables from project root
load_dotenv("../../.env")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens accurately using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def print_box(content: str, title: str = ""):
    """Print content in a box"""
    lines = content.split('\n')
    max_width = max(len(line) for line in lines) if lines else 0
    max_width = max(max_width, len(title))
    width = min(max_width + 4, 80)

    print("\n┌" + "─" * (width - 2) + "┐")
    if title:
        print(f"│ {title:<{width - 4}} │")
        print("├" + "─" * (width - 2) + "┤")

    for line in lines:
        print(f"│ {line:<{width - 4}} │")

    print("└" + "─" * (width - 2) + "┘")


class Message:
    """Represents a single conversation message"""

    def __init__(
        self,
        role: Literal["user", "assistant"],
        content: str,
        timestamp: Optional[datetime] = None,
        priority: Literal["critical", "high", "medium", "low"] = "medium"
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.priority = priority
        self.token_count = count_tokens(content)

    def __repr__(self) -> str:
        return f"Message(role={self.role}, tokens={self.token_count}, priority={self.priority})"


class BasicContextManager:
    """
    Basic context management with sliding window and compression.

    Strategy:
    1. Keep most recent N messages in full (sliding window)
    2. Summarize older messages beyond the window
    3. Always include critical priority messages
    4. Respect token budget
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        window_size: int = 5,
        response_reserve: int = 2000,
        model: str = "gpt-4o-mini"
    ):
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.response_reserve = response_reserve
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)

        # Storage
        self.messages: List[Message] = []
        self.summary: Optional[str] = None
        self.summary_tokens: int = 0

        print(f"\n[Context Manager Initialized]")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Window size: {window_size} messages")
        print(f"  Response reserve: {response_reserve} tokens")
        print(f"  Available for context: {max_tokens - response_reserve} tokens")

    def add_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        priority: Literal["critical", "high", "medium", "low"] = "medium"
    ) -> None:
        """Add a message to the conversation history"""
        message = Message(role=role, content=content, priority=priority)
        self.messages.append(message)

        print(f"\n[Message Added]")
        print(f"  Role: {role}")
        print(f"  Tokens: {message.token_count}")
        print(f"  Priority: {priority}")
        print(f"  Total messages: {len(self.messages)}")

    def _get_recent_messages(self) -> List[Message]:
        """Get messages within sliding window"""
        return self.messages[-self.window_size:]

    def _get_old_messages(self) -> List[Message]:
        """Get messages outside sliding window"""
        if len(self.messages) <= self.window_size:
            return []
        return self.messages[:-self.window_size]

    def _summarize_old_messages(self, messages: List[Message]) -> str:
        """Summarize old messages to save tokens"""
        if not messages:
            return ""

        # Construct text to summarize
        conversation_text = ""
        for msg in messages:
            conversation_text += f"{msg.role.upper()}: {msg.content}\n\n"

        # Create summary prompt
        prompt = f"""Summarize the following conversation history concisely, preserving key facts and context:

{conversation_text}

Provide a brief summary (2-3 sentences) of the main points."""

        print(f"\n[Summarizing {len(messages)} old messages...]")

        response = self.llm.invoke([HumanMessage(content=prompt)])
        summary = response.content

        summary_tokens = count_tokens(summary)
        original_tokens = sum(msg.token_count for msg in messages)

        print(f"  Original tokens: {original_tokens}")
        print(f"  Summary tokens: {summary_tokens}")
        print(f"  Compression ratio: {summary_tokens / original_tokens:.1%}")

        return summary

    def _get_critical_messages(self) -> List[Message]:
        """Get all critical priority messages"""
        return [msg for msg in self.messages if msg.priority == "critical"]

    def build_context(self, current_query: str) -> Dict:
        """
        Build optimized context for the current query.

        Process:
        1. Count available tokens
        2. Include critical messages (always)
        3. Include recent messages (sliding window)
        4. Summarize old messages if needed
        5. Assemble final context
        """
        print_section("Building Optimized Context")

        # Calculate available tokens
        query_tokens = count_tokens(current_query)
        available = self.max_tokens - self.response_reserve - query_tokens

        print(f"\n[Token Budget]")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Response reserve: {self.response_reserve}")
        print(f"  Query tokens: {query_tokens}")
        print(f"  Available for context: {available}")

        # Get message groups
        critical_msgs = self._get_critical_messages()
        recent_msgs = self._get_recent_messages()
        old_msgs = self._get_old_messages()

        # Calculate token usage
        critical_tokens = sum(msg.token_count for msg in critical_msgs if msg not in recent_msgs)
        recent_tokens = sum(msg.token_count for msg in recent_msgs)

        print(f"\n[Message Groups]")
        print(f"  Critical: {len(critical_msgs)} messages, {critical_tokens} tokens")
        print(f"  Recent (window): {len(recent_msgs)} messages, {recent_tokens} tokens")
        print(f"  Old (to summarize): {len(old_msgs)} messages")

        # Build context parts
        context_parts = []
        tokens_used = 0

        # 1. Add summary of old messages (if exists)
        if old_msgs:
            if not self.summary or len(old_msgs) > 0:
                self.summary = self._summarize_old_messages(old_msgs)
                self.summary_tokens = count_tokens(self.summary)

            if self.summary:
                context_parts.append(f"[Previous conversation summary]\n{self.summary}")
                tokens_used += self.summary_tokens

        # 2. Add critical messages not in recent window
        for msg in critical_msgs:
            if msg not in recent_msgs:
                context_parts.append(f"{msg.role.upper()}: {msg.content}")
                tokens_used += msg.token_count

        # 3. Add recent messages (sliding window)
        for msg in recent_msgs:
            context_parts.append(f"{msg.role.upper()}: {msg.content}")

        tokens_used += recent_tokens

        # Assemble final context
        final_context = "\n\n".join(context_parts)
        final_tokens = count_tokens(final_context)

        print(f"\n[Context Assembly]")
        print(f"  Summary tokens: {self.summary_tokens if self.summary else 0}")
        print(f"  Critical tokens: {critical_tokens}")
        print(f"  Recent tokens: {recent_tokens}")
        print(f"  Total context tokens: {final_tokens}")
        print(f"  Budget utilization: {final_tokens / available:.1%}")

        # Visualize context packing
        self._visualize_context_packing(
            available=available,
            summary=self.summary_tokens if self.summary else 0,
            critical=critical_tokens,
            recent=recent_tokens,
            query=query_tokens,
            response=self.response_reserve
        )

        return {
            "context": final_context,
            "tokens_used": final_tokens,
            "tokens_available": available,
            "utilization": final_tokens / available,
            "messages_included": len(recent_msgs) + len([m for m in critical_msgs if m not in recent_msgs]),
            "messages_summarized": len(old_msgs)
        }

    def _visualize_context_packing(
        self,
        available: int,
        summary: int,
        critical: int,
        recent: int,
        query: int,
        response: int
    ):
        """Visualize how tokens are packed into the context window"""
        print_section("Context Window Visualization")

        total = summary + critical + recent + query + response
        bar_width = 50

        # Calculate bar segments
        summary_bar = int((summary / self.max_tokens) * bar_width) if summary > 0 else 0
        critical_bar = int((critical / self.max_tokens) * bar_width)
        recent_bar = int((recent / self.max_tokens) * bar_width)
        query_bar = int((query / self.max_tokens) * bar_width)
        response_bar = int((response / self.max_tokens) * bar_width)
        unused = bar_width - (summary_bar + critical_bar + recent_bar + query_bar + response_bar)

        # Create visualization
        bar = ""
        if summary_bar > 0:
            bar += "S" * summary_bar
        if critical_bar > 0:
            bar += "C" * critical_bar
        bar += "R" * recent_bar
        bar += "Q" * query_bar
        bar += "r" * response_bar
        bar += "·" * unused

        print(f"\n[{bar}]")
        print(f"\nLegend:")
        print(f"  S = Summary ({summary} tokens)")
        print(f"  C = Critical messages ({critical} tokens)")
        print(f"  R = Recent messages ({recent} tokens)")
        print(f"  Q = Query ({query} tokens)")
        print(f"  r = Response reserve ({response} tokens)")
        print(f"  · = Unused ({self.max_tokens - total} tokens)")
        print(f"\nTotal: {total} / {self.max_tokens} tokens ({total / self.max_tokens:.1%})")


def demonstrate_basic_context_management():
    """Demonstrate basic context management with a growing conversation"""

    print_section("Basic Context Management Demo")
    print("\nProblem: How to handle a conversation that exceeds token limits?")
    print("Solution: Sliding window + compression")

    # Initialize context manager
    context_mgr = BasicContextManager(
        max_tokens=8000,
        window_size=5,
        response_reserve=2000
    )

    # Simulate a long conversation
    conversation = [
        ("user", "Hello! I'm planning a trip to Japan.", "medium"),
        ("assistant", "That's exciting! Japan is a wonderful destination. What cities are you planning to visit?", "medium"),
        ("user", "I'm thinking Tokyo, Kyoto, and maybe Osaka. How long should I stay?", "medium"),
        ("assistant", "I'd recommend at least 10-14 days to enjoy those cities. Tokyo needs 3-4 days, Kyoto 3-4 days, and Osaka 2-3 days. That leaves time for day trips too.", "medium"),
        ("user", "What's the best time to visit?", "medium"),
        ("assistant", "Spring (March-May) for cherry blossoms or Fall (September-November) for autumn colors are ideal. Summer is hot and humid, winter is cold but less crowded.", "medium"),
        ("user", "I'm interested in visiting during cherry blossom season. When exactly should I go?", "critical"),
        ("assistant", "Peak cherry blossom season in Tokyo and Kyoto is typically late March to early April, usually around March 25 - April 10. Book accommodations early as this is the busiest season!", "critical"),
        ("user", "What about food? I'm vegetarian.", "medium"),
        ("assistant", "Japan is challenging for vegetarians, but not impossible! Look for shojin ryori (Buddhist vegetarian cuisine). Many restaurants now have vegetarian options, especially in big cities. Learn to say 'watashi wa bejitarian desu' (I am vegetarian).", "medium"),
        ("user", "How much should I budget per day?", "high"),
        ("assistant", "Budget roughly ¥15,000-20,000 per day ($100-135 USD) for mid-range travel. This covers accommodation, food, local transport, and activities. Add extra for shopping and JR Pass if traveling between cities.", "high"),
        ("user", "What's the best way to get around Tokyo?", "medium"),
        ("assistant", "Tokyo's train and subway system is incredibly efficient. Get a Suica or Pasmo card for easy payment. Google Maps works great for navigation. Avoid taxis (expensive) and rush hour (7-9am, 5-7pm).", "medium"),
        ("user", "Any must-see spots in Kyoto?", "medium"),
        ("assistant", "Must-sees in Kyoto: Fushimi Inari Shrine (1000 torii gates), Kinkaku-ji (Golden Pavilion), Arashiyama Bamboo Grove, and Gion district for geishas. Consider a day trip to Nara to see the deer!", "medium"),
        ("user", "Thanks! One more question: Do I need travel insurance?", "medium"),
        ("assistant", "Yes, strongly recommended! Japan has excellent healthcare but it's expensive for foreigners. Get comprehensive travel insurance covering medical, cancellation, and lost luggage. Many credit cards offer travel insurance too.", "medium"),
    ]

    # Add messages to simulate conversation
    print_section("Simulating Long Conversation")
    print(f"Adding {len(conversation)} messages to the conversation...")

    for role, content, priority in conversation:
        context_mgr.add_message(role, content, priority)

    # Build context for a new query
    current_query = "Can you summarize the key points about my Japan trip planning, especially the cherry blossom timing?"

    print_section("Current Query")
    print_box(current_query, "User Query")

    # Build optimized context
    result = context_mgr.build_context(current_query)

    # Show the built context
    print_section("Optimized Context Built")
    print_box(result["context"], "Final Context")

    # Show metrics
    print_section("Context Management Metrics")
    print(f"\n  Messages in conversation: {len(context_mgr.messages)}")
    print(f"  Messages included (full): {result['messages_included']}")
    print(f"  Messages summarized: {result['messages_summarized']}")
    print(f"  Context tokens used: {result['tokens_used']}")
    print(f"  Context tokens available: {result['tokens_available']}")
    print(f"  Utilization: {result['utilization']:.1%}")

    # Calculate savings
    full_tokens = sum(msg.token_count for msg in context_mgr.messages)
    savings = 1 - (result['tokens_used'] / full_tokens)

    print(f"\n[Cost Savings]")
    print(f"  Full conversation: {full_tokens} tokens")
    print(f"  Optimized context: {result['tokens_used']} tokens")
    print(f"  Reduction: {savings:.1%}")
    print(f"  Cost savings: {savings:.1%} (if charged per token)")

    # Show comparison visualization
    print_section("Before vs After Comparison")

    print("\n❌ Without Context Management:")
    print(f"   - All {len(conversation)} messages: {full_tokens} tokens")
    print(f"   - Exceeds budget: ❌ (would fail or truncate badly)")
    print(f"   - Cost per query: ${full_tokens * 0.00001:.4f}")

    print("\n✅ With Context Management:")
    print(f"   - {result['messages_included']} recent + {result['messages_summarized']} summarized: {result['tokens_used']} tokens")
    print(f"   - Within budget: ✅ ({result['utilization']:.1%} utilization)")
    print(f"   - Cost per query: ${result['tokens_used'] * 0.00001:.4f}")
    print(f"   - Savings: {savings:.1%}")

    print_section("Key Takeaways")
    print("""
1. Sliding Window: Keep recent messages for context continuity
2. Compression: Summarize old messages to save tokens
3. Prioritization: Critical messages always included
4. Token Budget: Respect limits while maximizing information
5. Cost Savings: Reduce tokens by 50-90% while maintaining quality

This basic approach works well for:
- Chat applications with long conversations
- Sequential interactions where recency matters
- Simple token management without complex relevance scoring
    """)


if __name__ == "__main__":
    demonstrate_basic_context_management()
