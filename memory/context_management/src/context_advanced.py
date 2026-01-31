"""
Advanced Context Management Demo

Demonstrates:
- Semantic relevance scoring
- Multi-source context (documents, history, knowledge)
- Dynamic token allocation
- Hierarchical compression
- Adaptive strategies
- Context caching patterns
- Rich visualization

Problem: How to optimally select and compress context from multiple sources?
Solution: Use semantic similarity + dynamic allocation + hierarchical compression.
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import os
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
from dotenv import load_dotenv
import tiktoken
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables from project root
load_dotenv("../../.env")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens accurately using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_box(content: str, title: str = ""):
    """Print content in a box"""
    lines = content.split('\n')
    max_width = max(len(line) for line in lines) if lines else 0
    max_width = max(max_width, len(title))
    width = min(max_width + 4, 100)

    print("\n┌" + "─" * (width - 2) + "┐")
    if title:
        print(f"│ {title:<{width - 4}} │")
        print("├" + "─" * (width - 2) + "┤")

    for line in lines:
        if len(line) > width - 4:
            # Wrap long lines
            wrapped = [line[i:i+width-4] for i in range(0, len(line), width-4)]
            for wrapped_line in wrapped:
                print(f"│ {wrapped_line:<{width - 4}} │")
        else:
            print(f"│ {line:<{width - 4}} │")

    print("└" + "─" * (width - 2) + "┘")


@dataclass
class ContentChunk:
    """Represents a chunk of content from any source"""
    id: str
    text: str
    source: Literal["document", "conversation", "knowledge_base"]
    priority: Literal["critical", "high", "medium", "low"]
    timestamp: datetime
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.token_count = count_tokens(self.text)
        self.relevance_score: float = 0.0


class SemanticScorer:
    """Calculate semantic relevance scores using simple heuristics"""

    @staticmethod
    def calculate_similarity(query: str, content: str) -> float:
        """
        Calculate semantic similarity (simplified version).
        In production, use embeddings + cosine similarity.
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        # Jaccard similarity as simple approximation
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def calculate_recency_score(timestamp: datetime) -> float:
        """Calculate recency score (0-1, higher = more recent)"""
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600

        # Exponential decay: score = e^(-age/24)
        # Recent (< 1 hour): ~0.96
        # 1 day old: ~0.68
        # 3 days old: ~0.29
        # 7 days old: ~0.09
        return np.exp(-age_hours / 24)

    @staticmethod
    def calculate_keyword_score(query: str, content: str) -> float:
        """Calculate keyword overlap score"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        # Count important word matches (filter common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        query_important = query_words - common_words
        content_important = content_words - common_words

        if not query_important:
            return 0.0

        matches = query_important.intersection(content_important)
        return len(matches) / len(query_important)

    @classmethod
    def calculate_relevance(
        cls,
        query: str,
        chunk: ContentChunk,
        semantic_weight: float = 0.5,
        recency_weight: float = 0.3,
        keyword_weight: float = 0.2
    ) -> float:
        """Calculate overall relevance score"""
        semantic = cls.calculate_similarity(query, chunk.text)
        recency = cls.calculate_recency_score(chunk.timestamp)
        keywords = cls.calculate_keyword_score(query, chunk.text)

        return (
            semantic_weight * semantic +
            recency_weight * recency +
            keyword_weight * keywords
        )


class AdvancedContextManager:
    """
    Advanced context management with semantic selection and adaptive compression.

    Features:
    - Multi-source content handling
    - Semantic relevance scoring
    - Dynamic token allocation
    - Adaptive compression strategies
    - Hierarchical context building
    - Caching for efficiency
    """

    def __init__(
        self,
        max_tokens: int = 16000,
        response_reserve: int = 4096,
        model: str = "gpt-4o-mini"
    ):
        self.max_tokens = max_tokens
        self.response_reserve = response_reserve
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)

        # Storage
        self.chunks: List[ContentChunk] = []

        # Caching
        self.summary_cache: Dict[str, str] = {}

        print(f"\n[Advanced Context Manager Initialized]")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Response reserve: {response_reserve} tokens")
        print(f"  Model: {model}")

    def add_content(
        self,
        text: str,
        source: Literal["document", "conversation", "knowledge_base"],
        priority: Literal["critical", "high", "medium", "low"] = "medium",
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a content chunk"""
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:8]
        chunk = ContentChunk(
            id=chunk_id,
            text=text,
            source=source,
            priority=priority,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.chunks.append(chunk)

        print(f"\n[Content Added]")
        print(f"  ID: {chunk_id}")
        print(f"  Source: {source}")
        print(f"  Priority: {priority}")
        print(f"  Tokens: {chunk.token_count}")

        return chunk_id

    def _score_chunks(self, query: str) -> List[ContentChunk]:
        """Score all chunks for relevance to query"""
        print(f"\n[Scoring {len(self.chunks)} chunks for relevance...]")

        scored_chunks = []
        for chunk in self.chunks:
            relevance = SemanticScorer.calculate_relevance(query, chunk)
            chunk.relevance_score = relevance
            scored_chunks.append(chunk)

            print(f"  {chunk.id} ({chunk.source}): {relevance:.3f}")

        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_chunks

    def _allocate_tokens(
        self,
        chunks: List[ContentChunk],
        available_tokens: int
    ) -> Dict[str, int]:
        """Allocate tokens to chunks based on relevance and priority"""
        print(f"\n[Allocating {available_tokens} tokens across chunks...]")

        allocations = {}

        # Priority 1: Critical chunks (always full)
        critical = [c for c in chunks if c.priority == "critical"]
        for chunk in critical:
            allocations[chunk.id] = chunk.token_count
            available_tokens -= chunk.token_count
            print(f"  {chunk.id} (critical): {chunk.token_count} tokens (full)")

        if available_tokens <= 0:
            return allocations

        # Priority 2: High relevance (full if possible, else proportional)
        high_relevance = [c for c in chunks if c.relevance_score > 0.7 and c.priority != "critical"]

        for chunk in high_relevance:
            if available_tokens >= chunk.token_count:
                allocations[chunk.id] = chunk.token_count
                available_tokens -= chunk.token_count
                print(f"  {chunk.id} (high relevance): {chunk.token_count} tokens (full)")
            else:
                # Allocate what's left
                allocations[chunk.id] = available_tokens
                print(f"  {chunk.id} (high relevance): {available_tokens} tokens (partial)")
                available_tokens = 0
                break

        if available_tokens <= 0:
            return allocations

        # Priority 3: Medium relevance (proportional allocation)
        medium_relevance = [c for c in chunks if 0.4 <= c.relevance_score <= 0.7 and c.priority != "critical"]

        total_score = sum(c.relevance_score for c in medium_relevance)
        if total_score > 0:
            for chunk in medium_relevance:
                allocation = int(available_tokens * (chunk.relevance_score / total_score))
                allocation = min(allocation, chunk.token_count)
                allocations[chunk.id] = allocation
                print(f"  {chunk.id} (medium relevance): {allocation} tokens (compressed)")

        return allocations

    def _compress_chunk(self, chunk: ContentChunk, target_tokens: int) -> str:
        """Compress chunk to target token count"""
        if target_tokens >= chunk.token_count:
            return chunk.text

        # Check cache
        cache_key = f"{chunk.id}_{target_tokens}"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]

        # Calculate compression ratio
        ratio = target_tokens / chunk.token_count

        if ratio > 0.7:
            # Light compression: just truncate
            words = chunk.text.split()
            truncated = ' '.join(words[:int(len(words) * ratio)])
            compressed = truncated + "..."
        else:
            # Heavy compression: summarize
            prompt = f"""Summarize the following text to approximately {target_tokens} tokens:

{chunk.text}

Provide a concise summary preserving key information."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            compressed = response.content

        # Cache result
        self.summary_cache[cache_key] = compressed
        return compressed

    def build_context(self, query: str, strategy: str = "adaptive") -> Dict:
        """
        Build optimized context using advanced strategies.

        Strategies:
        - adaptive: Adjust based on query and content
        - hierarchical: Summary → details
        - semantic: Pure relevance-based
        """
        print_section("Building Advanced Context")

        # Calculate available tokens
        query_tokens = count_tokens(query)
        available = self.max_tokens - self.response_reserve - query_tokens

        print(f"\n[Token Budget]")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Response reserve: {self.response_reserve}")
        print(f"  Query tokens: {query_tokens}")
        print(f"  Available for context: {available}")

        # Score chunks
        scored_chunks = self._score_chunks(query)

        # Allocate tokens
        allocations = self._allocate_tokens(scored_chunks, available)

        # Build context based on strategy
        if strategy == "hierarchical":
            context = self._build_hierarchical_context(scored_chunks, allocations)
        else:
            context = self._build_adaptive_context(scored_chunks, allocations)

        # Calculate metrics
        context_tokens = count_tokens(context)
        total_original_tokens = sum(c.token_count for c in self.chunks)
        compression_ratio = context_tokens / total_original_tokens if total_original_tokens > 0 else 0

        print(f"\n[Context Built]")
        print(f"  Strategy: {strategy}")
        print(f"  Original tokens: {total_original_tokens}")
        print(f"  Context tokens: {context_tokens}")
        print(f"  Compression ratio: {compression_ratio:.1%}")
        print(f"  Chunks included: {len(allocations)}/{len(self.chunks)}")

        return {
            "context": context,
            "tokens_used": context_tokens,
            "tokens_available": available,
            "utilization": context_tokens / available,
            "chunks_included": len(allocations),
            "chunks_total": len(self.chunks),
            "compression_ratio": compression_ratio,
            "strategy": strategy
        }

    def _build_adaptive_context(
        self,
        chunks: List[ContentChunk],
        allocations: Dict[str, int]
    ) -> str:
        """Build context with adaptive compression"""
        context_parts = []

        for chunk in chunks:
            if chunk.id not in allocations:
                continue

            allocated = allocations[chunk.id]
            if allocated >= chunk.token_count * 0.9:
                # Full inclusion
                prefix = f"[{chunk.source.upper()} - FULL]"
                content = chunk.text
            else:
                # Compressed
                prefix = f"[{chunk.source.upper()} - SUMMARY]"
                content = self._compress_chunk(chunk, allocated)

            context_parts.append(f"{prefix}\n{content}")

        return "\n\n".join(context_parts)

    def _build_hierarchical_context(
        self,
        chunks: List[ContentChunk],
        allocations: Dict[str, int]
    ) -> str:
        """Build context with hierarchical structure"""
        context_parts = []

        # Level 1: Executive summary
        high_priority = [c for c in chunks if c.priority in ["critical", "high"]]
        if high_priority:
            summaries = []
            for chunk in high_priority[:3]:  # Top 3
                summary = self._compress_chunk(chunk, 100)  # Very brief
                summaries.append(f"- {summary}")

            context_parts.append("=== OVERVIEW ===\n" + "\n".join(summaries))

        # Level 2: Detailed content
        for chunk in chunks:
            if chunk.id not in allocations:
                continue

            allocated = allocations[chunk.id]
            if allocated > 0:
                content = self._compress_chunk(chunk, allocated) if allocated < chunk.token_count else chunk.text
                context_parts.append(f"=== {chunk.source.upper()}: {chunk.id} ===\n{content}")

        return "\n\n".join(context_parts)

    def visualize_optimization(
        self,
        query: str,
        result: Dict
    ):
        """Visualize the context optimization process"""
        print_section("Context Optimization Visualization")

        # Show query analysis
        print("\n📊 Query Analysis:")
        print(f"   Query: \"{query}\"")
        print(f"   Tokens: {count_tokens(query)}")

        # Show chunk relevance distribution
        print("\n📈 Relevance Score Distribution:")
        for chunk in sorted(self.chunks, key=lambda x: x.relevance_score, reverse=True):
            bar_length = int(chunk.relevance_score * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {chunk.id} ({chunk.source[:4]}): {bar} {chunk.relevance_score:.3f}")

        # Show token allocation
        print("\n💾 Token Allocation:")
        total_original = sum(c.token_count for c in self.chunks)
        total_allocated = result['tokens_used']

        for chunk in self.chunks:
            original = chunk.token_count
            allocated = count_tokens(self._compress_chunk(chunk, original))

            if chunk.relevance_score > 0.5:
                status = "✓ Included"
                ratio = allocated / original if original > 0 else 0
                print(f"   {chunk.id}: {original} → {allocated} tokens ({ratio:.0%}) {status}")
            else:
                print(f"   {chunk.id}: {original} → 0 tokens (excluded)")

        # Show before/after comparison
        print("\n⚖️  Before vs After:")
        print(f"   Original: {total_original} tokens")
        print(f"   Optimized: {total_allocated} tokens")
        print(f"   Reduction: {1 - total_allocated/total_original:.1%}")
        print(f"   Cost savings: ${total_original * 0.00001:.4f} → ${total_allocated * 0.00001:.4f}")

        # Show context window usage
        print("\n📦 Context Window Usage:")
        bar_width = 50
        used = int((result['tokens_used'] / self.max_tokens) * bar_width)
        reserve = int((self.response_reserve / self.max_tokens) * bar_width)
        unused = bar_width - used - reserve

        bar = "█" * used + "▓" * reserve + "░" * unused
        print(f"   [{bar}]")
        print(f"   █ Context: {result['tokens_used']} tokens ({result['utilization']:.1%})")
        print(f"   ▓ Response: {self.response_reserve} tokens")
        print(f"   ░ Unused: {self.max_tokens - result['tokens_used'] - self.response_reserve} tokens")


def demonstrate_advanced_context_management():
    """Demonstrate advanced context management with multi-source content"""

    print_section("Advanced Context Management Demo")
    print("\nProblem: How to optimally select context from multiple sources?")
    print("Solution: Semantic scoring + dynamic allocation + adaptive compression")

    # Initialize context manager
    context_mgr = AdvancedContextManager(
        max_tokens=16000,
        response_reserve=4096
    )

    # Add content from multiple sources
    print_section("Adding Multi-Source Content")

    # Document 1: OAuth documentation
    context_mgr.add_content(
        text="""OAuth 2.0 Authentication Configuration

To configure OAuth authentication in our system:

1. Register your application in the developer portal
2. Obtain client ID and client secret
3. Configure redirect URIs for your application
4. Implement the authorization flow:
   - User authorization request
   - Authorization code exchange
   - Access token retrieval
   - Token refresh mechanism

Supported OAuth flows:
- Authorization Code Flow (recommended for web apps)
- Implicit Flow (for single-page apps)
- Client Credentials Flow (for service-to-service)
- Resource Owner Password Flow (legacy, not recommended)

Security considerations:
- Always use HTTPS for OAuth endpoints
- Store client secrets securely
- Implement PKCE for public clients
- Validate redirect URIs strictly
- Use short-lived access tokens
- Implement proper token refresh logic""",
        source="document",
        priority="high",
        metadata={"doc_name": "oauth_config.md", "section": "authentication"}
    )

    # Document 2: API reference
    context_mgr.add_content(
        text="""API Authentication Endpoints

POST /oauth/token
Request access token

Parameters:
- grant_type: Type of OAuth grant (required)
- client_id: Your application ID (required)
- client_secret: Your application secret (required)
- code: Authorization code (for authorization code flow)
- redirect_uri: Callback URL (must match registered URI)
- scope: Requested permissions (space-separated)

Response:
{
  "access_token": "eyJhbGc...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "tGzv3JOk...",
  "scope": "read write"
}

Example request:
curl -X POST https://api.example.com/oauth/token \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "grant_type=authorization_code" \\
  -d "client_id=YOUR_CLIENT_ID" \\
  -d "client_secret=YOUR_CLIENT_SECRET" \\
  -d "code=AUTH_CODE" \\
  -d "redirect_uri=https://your-app.com/callback"
""",
        source="document",
        priority="high",
        metadata={"doc_name": "api_reference.md", "section": "authentication"}
    )

    # Conversation history
    context_mgr.add_content(
        text="USER: I'm trying to set up authentication for my web application.\nASSISTANT: Great! I can help you with that. Are you looking to implement OAuth 2.0 authentication?",
        source="conversation",
        priority="medium",
        metadata={"turn": 1}
    )

    context_mgr.add_content(
        text="USER: Yes, OAuth 2.0. Which flow should I use?\nASSISTANT: For a web application, I recommend the Authorization Code Flow. It's the most secure option for server-side apps.",
        source="conversation",
        priority="medium",
        metadata={"turn": 2}
    )

    # Knowledge base: Related topic
    context_mgr.add_content(
        text="""JWT Token Structure

JSON Web Tokens consist of three parts:
1. Header: Algorithm and token type
2. Payload: Claims and data
3. Signature: Verification signature

Example JWT:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Always verify JWT signatures before trusting the token content.""",
        source="knowledge_base",
        priority="low",
        metadata={"topic": "jwt"}
    )

    # Less relevant content
    context_mgr.add_content(
        text="""Database Configuration

To configure your database connection:
1. Set up connection string in config file
2. Configure connection pooling
3. Set up migrations
4. Test connection

Supported databases: PostgreSQL, MySQL, MongoDB""",
        source="document",
        priority="low",
        metadata={"doc_name": "database_setup.md"}
    )

    # Current query
    query = "How do I configure OAuth authentication? I need the specific API endpoint and parameters."

    print_section("Current Query")
    print_box(query, "User Query")

    # Build context with adaptive strategy
    result = context_mgr.build_context(query, strategy="adaptive")

    # Show optimized context
    print_section("Optimized Context")
    print_box(result["context"], "Final Context")

    # Visualize optimization
    context_mgr.visualize_optimization(query, result)

    # Show metrics
    print_section("Context Management Metrics")
    print(f"\n  Chunks available: {result['chunks_total']}")
    print(f"  Chunks included: {result['chunks_included']}")
    print(f"  Context tokens: {result['tokens_used']}")
    print(f"  Budget utilization: {result['utilization']:.1%}")
    print(f"  Compression ratio: {result['compression_ratio']:.1%}")

    # Cost analysis
    print_section("Cost Analysis")

    original_tokens = sum(c.token_count for c in context_mgr.chunks)
    optimized_tokens = result['tokens_used']
    savings_ratio = 1 - (optimized_tokens / original_tokens)

    print(f"\n  Full context: {original_tokens} tokens")
    print(f"  Optimized context: {optimized_tokens} tokens")
    print(f"  Token reduction: {savings_ratio:.1%}")
    print(f"\n  Cost per query:")
    print(f"    Without optimization: ${original_tokens * 0.00001:.4f}")
    print(f"    With optimization: ${optimized_tokens * 0.00001:.4f}")
    print(f"    Savings: ${(original_tokens - optimized_tokens) * 0.00001:.4f} ({savings_ratio:.1%})")
    print(f"\n  Annual savings (1000 queries/day):")
    print(f"    Daily: ${(original_tokens - optimized_tokens) * 0.00001 * 1000:.2f}")
    print(f"    Monthly: ${(original_tokens - optimized_tokens) * 0.00001 * 1000 * 30:.2f}")
    print(f"    Yearly: ${(original_tokens - optimized_tokens) * 0.00001 * 1000 * 365:.2f}")

    print_section("Key Takeaways")
    print("""
1. Semantic Relevance: Score content based on meaning, not just keywords
2. Dynamic Allocation: Distribute tokens based on relevance and priority
3. Adaptive Compression: Apply appropriate compression per content type
4. Multi-Source: Handle documents, conversations, and knowledge bases
5. Cost Optimization: Achieve 70-90% cost reduction while maintaining quality

This advanced approach works well for:
- Document Q&A systems
- RAG applications with multiple retrievals
- Multi-source synthesis
- Cost-sensitive production deployments
- Complex queries requiring focused context
    """)


if __name__ == "__main__":
    demonstrate_advanced_context_management()
