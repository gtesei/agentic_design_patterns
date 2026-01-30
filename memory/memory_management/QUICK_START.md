# Memory Management - Quick Start

Get started with Memory Management in 5 minutes.

## Prerequisites

- Python 3.11+
- OpenAI API key

## Installation

```bash
# Navigate to the memory management directory
cd memory/memory_management

# Install dependencies
pip install -r requirements.txt
# or using uv (faster)
uv pip install -r requirements.txt
```

## Setup Environment

Create a `.env` file in the project root with your OpenAI API key:

```bash
# From the agentic_design_patterns root directory
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

## Run Examples

### Basic Memory Example

Demonstrates conversation buffer and automatic summarization:

```bash
./run.sh basic
```

This will show:
- Conversation buffer (last N messages)
- Automatic summarization when buffer is full
- Memory persistence (save/load)
- Simple retrieval by recency

### Advanced Memory Example

Demonstrates semantic memory with vector storage:

```bash
./run.sh advanced
```

This will show:
- Semantic memory using vector embeddings
- Entity memory (tracking people, places, preferences)
- Memory importance scoring
- Intelligent retrieval (relevance + recency)
- Memory consolidation

### Run Both

```bash
./run.sh
```

## What You'll Learn

### Basic Memory (`memory_basic.py`)

**Problem**: Maintain conversation history across multiple turns

**Solution**:
- Keep recent messages in a buffer
- Summarize old conversations
- Persist memory to disk
- Simple recency-based retrieval

**Key Concepts**:
- Buffer memory (sliding window)
- Memory summarization
- Save/load functionality

### Advanced Memory (`memory_advanced.py`)

**Problem**: Personal assistant with long-term semantic memory

**Solution**:
- Vector embeddings for semantic search
- Entity tracking (people, preferences, facts)
- Importance scoring
- Hybrid retrieval (relevance + recency)
- Memory consolidation

**Key Concepts**:
- Vector memory stores
- Semantic similarity search
- Entity extraction
- Memory networks

## Code Overview

### Basic Memory Structure

```python
class ConversationMemory:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)
        self.summaries = []

    def add_message(self, role: str, content: str):
        """Add message to buffer"""
        self.buffer.append({"role": role, "content": content})

    def get_context(self) -> str:
        """Get recent messages + summaries"""
        context = list(self.buffer)
        if self.summaries:
            context = self.summaries + context
        return context
```

### Advanced Memory Structure

```python
class SemanticMemory:
    def __init__(self):
        self.vector_store = QdrantClient(":memory:")
        self.entities = {}  # Entity tracking
        self.memories = []  # All memories

    def add_memory(self, text: str, metadata: dict):
        """Store with embedding"""
        embedding = self.embed(text)
        self.vector_store.upsert(...)

    def retrieve(self, query: str, top_k: int = 5):
        """Semantic similarity search"""
        query_embedding = self.embed(query)
        results = self.vector_store.search(...)
        return results
```

## Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Experiment** with different memory types and retrieval strategies
3. **Modify examples** to fit your use case
4. **Integrate** memory into your own agents

## Common Issues

### Import Errors

If you see import errors, make sure you've installed all dependencies:

```bash
pip install langchain langchain-openai langgraph python-dotenv qdrant-client
```

### API Key Issues

Ensure your `.env` file is in the project root (two levels up):

```bash
# Should be at: agentic_design_patterns/.env
cat ../../.env
```

### Memory Persistence

Memory files are saved to `./memory_data/`. Make sure the directory has write permissions.

## Resources

- [Full Documentation](README.md)
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## Example Output

```
=== Memory Management Demo ===

Conversation Turn 1:
User: My name is Alice and I love Python programming
Assistant: Nice to meet you, Alice! Python is a great language...

[Memory Stored: entity=Alice, preference=Python, sentiment=positive]

Conversation Turn 5:
User: What do you remember about me?
Assistant: You're Alice, and you love Python programming...

[Retrieved: 3 relevant memories from semantic store]

Memory Stats:
- Buffer size: 10 messages
- Total memories: 23
- Entities tracked: 5
- Average retrieval time: 45ms
```

Ready to build agents that remember? Let's go!
