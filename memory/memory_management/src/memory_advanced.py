"""
Advanced Memory Management Demo

Demonstrates:
- Semantic memory using vector embeddings
- Entity memory (track people, places, preferences)
- Memory importance scoring
- Intelligent retrieval (relevance + recency)
- Memory consolidation (merge related memories)
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv

import numpy as np
import networkx as nx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from project root
load_dotenv("../../.env")


class Memory:
    """Represents a single memory with metadata"""

    def __init__(
        self,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.importance = importance
        self.timestamp = datetime.now()
        self.access_count = 0
        self.last_accessed = None
        self.metadata = metadata or {}

    def access(self) -> None:
        """Record memory access"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def calculate_recency_score(self) -> float:
        """Calculate recency score (decay over time)"""
        days_old = (datetime.now() - self.timestamp).days
        return 1.0 / (1.0 + days_old / 7.0)  # Decay over weeks

    def calculate_importance_score(self) -> float:
        """Calculate dynamic importance based on multiple factors"""
        score = self.importance

        # Boost by access frequency
        access_score = min(self.access_count / 5.0, 0.3)
        score += access_score

        # Boost by memory type
        type_boost = {"preference": 0.2, "fact": 0.15, "conversation": 0.0}
        score += type_boost.get(self.memory_type, 0.0)

        return min(score, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create Memory from dictionary"""
        memory = cls(
            content=data["content"],
            memory_type=data["memory_type"],
            importance=data["importance"],
            metadata=data["metadata"],
        )
        memory.id = data["id"]
        memory.timestamp = datetime.fromisoformat(data["timestamp"])
        memory.access_count = data["access_count"]
        if data["last_accessed"]:
            memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return memory


class SemanticMemory:
    """Advanced memory system with semantic search and entity tracking"""

    def __init__(self, collection_name: str = "memories"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Initialize Qdrant in-memory
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # In-memory stores
        self.memories: Dict[str, Memory] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.memory_graph = nx.Graph()  # Associative memory network

    def add_memory(
        self,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Add a memory with semantic embedding"""
        # Create memory object
        memory = Memory(content, memory_type, importance, metadata)

        # Store in memory dict
        self.memories[memory.id] = memory

        # Generate embedding
        embedding = self.embeddings.embed_query(content)

        # Store in vector database
        point = PointStruct(
            id=memory.id,
            vector=embedding,
            payload={
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "timestamp": memory.timestamp.isoformat(),
                "metadata": metadata or {},
            },
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

        # Extract and store entities
        self._extract_entities(memory)

        # Create associations with related memories
        self._create_associations(memory, embedding)

        print(
            f"  [Memory] Stored {memory_type} memory (importance: {importance:.2f}): '{content[:60]}...'"
        )

        return memory

    def _extract_entities(self, memory: Memory) -> None:
        """Extract entities from memory content"""
        # Use LLM to extract entities
        prompt = f"""Extract key entities from this text. Identify:
- People (names)
- Places (locations)
- Preferences (likes/dislikes)
- Facts (concrete information)

Text: {memory.content}

Return ONLY a JSON object with entities, like:
{{"people": ["Alice"], "places": ["New York"], "preferences": ["loves pizza"], "facts": ["works at TechCorp"]}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            entities = json.loads(content)

            # Store entities
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity_type not in self.entities:
                        self.entities[entity_type] = {}

                    if entity not in self.entities[entity_type]:
                        self.entities[entity_type][entity] = {
                            "mentions": [],
                            "first_seen": memory.timestamp,
                            "importance": 0.5,
                        }

                    self.entities[entity_type][entity]["mentions"].append(
                        {"memory_id": memory.id, "timestamp": memory.timestamp}
                    )

                    print(f"    [Entity] Extracted {entity_type}: {entity}")
        except Exception as e:
            print(f"    [Entity] Extraction failed: {e}")

    def _create_associations(self, memory: Memory, embedding: List[float]) -> None:
        """Create associations with similar memories"""
        # Find similar memories
        similar = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=3,
            score_threshold=0.7,
        )

        # Add edges in memory graph
        self.memory_graph.add_node(memory.id, memory=memory)

        for result in similar:
            if result.id != memory.id:
                self.memory_graph.add_edge(memory.id, result.id, similarity=result.score)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.5,
        importance_weight: float = 0.2,
    ) -> List[Tuple[Memory, float]]:
        """Retrieve memories using hybrid scoring"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search vector store
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=top_k * 2
        )

        # Score and rank results
        scored_memories = []
        for result in results:
            memory = self.memories.get(result.id)
            if not memory:
                continue

            # Mark as accessed
            memory.access()

            # Calculate hybrid score
            relevance_score = result.score
            recency_score = memory.calculate_recency_score()
            importance_score = memory.calculate_importance_score()

            hybrid_score = (
                relevance_weight * relevance_score
                + recency_weight * recency_score
                + importance_weight * importance_score
            )

            scored_memories.append((memory, hybrid_score))

        # Sort by hybrid score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return scored_memories[:top_k]

    def get_entity(self, entity_type: str, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific entity"""
        return self.entities.get(entity_type, {}).get(entity_name)

    def get_all_entities(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked entities"""
        return self.entities

    def consolidate_memories(self, similarity_threshold: float = 0.85) -> int:
        """Consolidate similar memories to reduce redundancy"""
        print("\n  [Consolidation] Analyzing memories for consolidation...")

        # Find clusters of similar memories
        clusters = []
        processed = set()

        for memory_id in self.memories:
            if memory_id in processed:
                continue

            # Get embedding
            memory = self.memories[memory_id]
            embedding = self.embeddings.embed_query(memory.content)

            # Find similar memories
            similar = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=10,
                score_threshold=similarity_threshold,
            )

            if len(similar) >= 2:  # At least 2 similar memories
                cluster = [result.id for result in similar]
                clusters.append(cluster)
                processed.update(cluster)

        # Consolidate each cluster
        consolidated_count = 0
        for cluster in clusters:
            if len(cluster) >= 2:
                cluster_memories = [self.memories[mid] for mid in cluster]
                self._consolidate_cluster(cluster_memories)
                consolidated_count += 1

        print(f"  [Consolidation] Consolidated {consolidated_count} memory clusters")
        return consolidated_count

    def _consolidate_cluster(self, memories: List[Memory]) -> None:
        """Consolidate a cluster of similar memories"""
        # Combine contents
        combined_content = "\n".join([m.content for m in memories])

        # Create summary
        prompt = f"""These related memories can be consolidated. Create a single, comprehensive summary:

{combined_content}

Consolidated memory (2-3 sentences):"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        consolidated_content = response.content.strip()

        # Create consolidated memory with high importance
        max_importance = max(m.importance for m in memories)
        self.add_memory(
            content=consolidated_content,
            memory_type="consolidated",
            importance=min(max_importance + 0.1, 1.0),
            metadata={"source_count": len(memories), "source_ids": [m.id for m in memories]},
        )

        # Remove original memories
        for memory in memories:
            if memory.id in self.memories:
                del self.memories[memory.id]

    def display_stats(self) -> None:
        """Display memory statistics"""
        print("\n" + "=" * 60)
        print("SEMANTIC MEMORY STATISTICS")
        print("=" * 60)
        print(f"Total memories: {len(self.memories)}")
        print(f"Memory types: {dict(self._count_by_type())}")
        print(f"Entities tracked: {sum(len(v) for v in self.entities.values())}")
        for entity_type, entity_dict in self.entities.items():
            print(f"  - {entity_type}: {len(entity_dict)}")
        print(f"Memory associations: {self.memory_graph.number_of_edges()}")
        print(
            f"Average importance: {np.mean([m.calculate_importance_score() for m in self.memories.values()]):.2f}"
        )
        print("=" * 60)

    def _count_by_type(self) -> Dict[str, int]:
        """Count memories by type"""
        counts = defaultdict(int)
        for memory in self.memories.values():
            counts[memory.memory_type] += 1
        return counts

    def display_memory_network(self) -> None:
        """Display memory association network"""
        print("\n" + "-" * 60)
        print("MEMORY ASSOCIATION NETWORK")
        print("-" * 60)

        if self.memory_graph.number_of_edges() == 0:
            print("No associations yet.")
            return

        # Get most connected memories
        degree_centrality = nx.degree_centrality(self.memory_graph)
        top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\nMost connected memories:")
        for memory_id, centrality in top_connected:
            memory = self.memories.get(memory_id)
            if memory:
                connections = list(self.memory_graph.neighbors(memory_id))
                print(f"  • {memory.content[:60]}... ({len(connections)} connections)")

        print("-" * 60)

    def display_entities(self) -> None:
        """Display tracked entities"""
        print("\n" + "-" * 60)
        print("ENTITY MEMORY")
        print("-" * 60)

        if not self.entities:
            print("No entities tracked yet.")
            return

        for entity_type, entity_dict in self.entities.items():
            print(f"\n{entity_type.upper()}:")
            for entity_name, entity_data in entity_dict.items():
                mention_count = len(entity_data["mentions"])
                print(f"  • {entity_name} ({mention_count} mentions)")

        print("-" * 60)

    def save(self, filepath: str) -> None:
        """Save memory to disk"""
        data = {
            "memories": {mid: mem.to_dict() for mid, mem in self.memories.items()},
            "entities": self.entities,
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_memories": len(self.memories),
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n  [Memory] Saved to {filepath}")


class AdvancedMemoryAgent:
    """Conversational agent with advanced semantic memory"""

    def __init__(self, memory: SemanticMemory):
        self.memory = memory
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def chat(self, user_message: str, show_retrieval: bool = False) -> str:
        """Process user message with memory retrieval"""
        # Store user message
        self.memory.add_memory(
            content=f"User: {user_message}",
            memory_type="conversation",
            importance=0.6,
            metadata={"role": "user"},
        )

        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(user_message, top_k=5)

        if show_retrieval and relevant_memories:
            print("\n  [Retrieval] Retrieved memories:")
            for i, (memory, score) in enumerate(relevant_memories, 1):
                print(
                    f"    {i}. (score: {score:.3f}) {memory.content[:80]}..."
                )

        # Build context
        memory_context = []
        for memory, score in relevant_memories:
            memory_context.append(f"- {memory.content}")

        context_str = "\n".join(memory_context) if memory_context else "No previous context."

        # Create prompt with memory context
        system_prompt = f"""You are a helpful personal assistant with memory. Use the following
context from past conversations to provide personalized responses:

MEMORY CONTEXT:
{context_str}

Be natural and conversational. Reference past information when relevant."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

        # Generate response
        response = self.llm.invoke(messages)
        assistant_message = response.content

        # Store assistant response
        self.memory.add_memory(
            content=f"Assistant: {assistant_message}",
            memory_type="conversation",
            importance=0.5,
            metadata={"role": "assistant"},
        )

        return assistant_message


def demonstrate_advanced_memory():
    """Demonstrate advanced memory management"""
    print("\n" + "=" * 60)
    print("ADVANCED MEMORY MANAGEMENT DEMO")
    print("=" * 60)
    print("\nProblem: Personal assistant with long-term semantic memory")
    print(
        "Solution: Vector embeddings + entity tracking + importance scoring\n"
    )

    # Initialize semantic memory
    memory = SemanticMemory()
    agent = AdvancedMemoryAgent(memory)

    # Scenario: Personal assistant over multiple days
    print("\n--- Scenario: Personal Assistant (Day 1) ---\n")

    day1_conversations = [
        ("My name is Bob and I work as a data scientist at DataCorp", True),
        ("I'm working on a customer churn prediction project using Python and scikit-learn", True),
        ("My favorite programming language is Python, and I love working with pandas", True),
        ("I enjoy rock climbing on weekends and I'm planning a trip to Yosemite", True),
    ]

    for i, (user_msg, show_retrieval) in enumerate(day1_conversations, 1):
        print(f"\n{'='*60}")
        print(f"Day 1 - Turn {i}")
        print(f"{'='*60}")
        print(f"\n👤 USER: {user_msg}")

        response = agent.chat(user_msg, show_retrieval=show_retrieval)
        print(f"\n🤖 ASSISTANT: {response}")

    # Display entities learned
    memory.display_entities()

    # Simulate day 2
    print("\n\n--- Scenario: Same User, Day 2 (New Session) ---\n")

    day2_conversations = [
        ("Hi! What do you remember about my work?", True),
        ("I'm thinking about learning a new programming language. Any suggestions based on what I do?", True),
        ("What did I say about my hobbies?", True),
    ]

    for i, (user_msg, show_retrieval) in enumerate(day2_conversations, 1):
        print(f"\n{'='*60}")
        print(f"Day 2 - Turn {i}")
        print(f"{'='*60}")
        print(f"\n👤 USER: {user_msg}")

        response = agent.chat(user_msg, show_retrieval=show_retrieval)
        print(f"\n🤖 ASSISTANT: {response}")

    # Display memory statistics
    memory.display_stats()

    # Display memory network
    memory.display_memory_network()

    # Demonstrate memory consolidation
    print("\n\n--- Memory Consolidation ---\n")
    consolidated = memory.consolidate_memories(similarity_threshold=0.75)

    if consolidated > 0:
        memory.display_stats()

    # Demonstrate importance-based retrieval
    print("\n\n--- Importance-Based Memory ---\n")

    # Add some high-importance memories
    memory.add_memory(
        content="CRITICAL: Bob mentioned he's allergic to peanuts",
        memory_type="fact",
        importance=1.0,
        metadata={"category": "health"},
    )

    memory.add_memory(
        content="Bob prefers morning meetings before 10am",
        memory_type="preference",
        importance=0.8,
        metadata={"category": "scheduling"},
    )

    # Query that should retrieve high-importance memory
    print(f"\n{'='*60}")
    print("Testing High-Importance Retrieval")
    print(f"{'='*60}")
    print("\n👤 USER: Can I bring peanut butter cookies to share?")

    response = agent.chat("Can I bring peanut butter cookies to share?", show_retrieval=True)
    print(f"\n🤖 ASSISTANT: {response}")

    # Save memory
    print("\n\n--- Memory Persistence ---\n")
    save_path = "./memory_data/conversation_advanced.json"
    memory.save(save_path)

    # Final statistics
    memory.display_stats()

    print("\n\n--- Key Features Demonstrated ---\n")
    print("✅ Semantic Memory: Vector embeddings for similarity search")
    print("✅ Entity Tracking: Extracted people, places, preferences, facts")
    print("✅ Importance Scoring: High-priority memories prioritized")
    print("✅ Hybrid Retrieval: Relevance + recency + importance")
    print("✅ Memory Consolidation: Similar memories merged")
    print("✅ Associative Network: Related memories connected")
    print("✅ Long-term Learning: Context retained across days")


if __name__ == "__main__":
    demonstrate_advanced_memory()
