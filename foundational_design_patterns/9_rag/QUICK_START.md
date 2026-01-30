# RAG Pattern - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the RAG Directory
```bash
cd foundational_design_patterns/9_rag
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
- **Option 1**: Basic RAG (simple in-memory implementation)
- **Option 2**: Semantic Search RAG (with Qdrant)
- **Option 3**: Advanced RAG (with re-ranking and citations)
- **Option 4**: Run all examples

---

## 📖 Understanding RAG in 30 Seconds

**RAG** = **R**etrieval-**A**ugmented **G**eneration

Instead of relying only on what the LLM learned during training, RAG:
1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the prompt with retrieved context
3. **Generates** an answer grounded in those documents

```
User Question → Search Documents → Add Context → LLM Response + Citations
```

**Key Benefit**: Answers are accurate, up-to-date, and cite sources!

---

## 🛠️ Available Components

### Vector Databases
- **Qdrant** (in-memory) - Simple, fast, no setup required
- **Qdrant** (persistent) - Save your vector database to disk
- **ChromaDB** - Alternative lightweight option

### Embedding Models
- `all-MiniLM-L6-v2` - Fast, 384 dimensions (default)
- `all-mpnet-base-v2` - Better quality, 768 dimensions
- `text-embedding-3-small` - OpenAI embeddings

### Retrieval Strategies
- **Semantic Search** - Find similar meaning
- **Keyword Search** - Exact word matching
- **Hybrid Search** - Combine both approaches
- **MMR (Maximum Marginal Relevance)** - Diverse results

---

## 💡 Example Queries to Try

### Simple Fact Lookup
```
"What is the company's remote work policy?"
```

### Multi-Document Synthesis
```
"Compare the features of Product A and Product B"
```

### Time-Sensitive Information
```
"What were the Q3 2024 revenue figures?"
```

### Domain-Specific Questions
```
"How do I configure SSL certificates for the API server?"
```

### Complex Research
```
"What are the main arguments for and against renewable energy adoption?"
```

---

## 🎯 Key Concepts

### Embeddings
Convert text to numbers (vectors) that represent meaning:
```
"cat" → [0.23, -0.15, 0.67, ...]
"dog" → [0.21, -0.18, 0.64, ...]  ← Similar to "cat"
"car" → [0.89, 0.34, -0.12, ...]  ← Different from "cat"
```

Similar meaning = similar vectors = found by search!

### Vector Search
Find documents with similar meaning to your query:
1. Convert query to embedding
2. Compare with all document embeddings
3. Return most similar documents (by cosine similarity)

### Chunking
Split large documents into smaller pieces:
- **Why**: Better retrieval precision
- **Size**: 200-1000 tokens per chunk
- **Overlap**: 10-20% to preserve context
- **Strategy**: Split on paragraphs, sections, or sentences

### Augmentation
Add retrieved documents to your LLM prompt:
```
Context: [Retrieved documents here...]

Question: What is photosynthesis?

Answer based on the context above.
```

---

## 📊 Comparison of Approaches

| Approach | Complexity | Accuracy | Speed | Best For |
|----------|-----------|----------|-------|----------|
| **Basic RAG** | Low | Good | Fast | Getting started |
| **Semantic Search** | Medium | Better | Fast | Production use |
| **Hybrid Search** | Medium | Best | Medium | High precision needs |
| **With Re-ranking** | High | Excellent | Slower | When accuracy is critical |

**Recommendation**: Start with Basic, move to Semantic Search for production.

---

## 🔧 Customization Tips

### 1. Add Your Own Documents

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Your documents
documents = [
    "Document 1 content here...",
    "Document 2 content here...",
    "Document 3 content here..."
]

# Add to Qdrant
points = [
    PointStruct(
        id=idx,
        vector=embedding_model.encode(doc).tolist(),
        payload={"text": doc, "source": f"doc_{idx}"}
    )
    for idx, doc in enumerate(documents)
]

qdrant.upsert(collection_name="my_knowledge", points=points)
```

### 2. Load Documents from Files

```python
from pathlib import Path

# Load all text files from a directory
docs_dir = Path("knowledge_base")
documents = []

for file_path in docs_dir.glob("*.txt"):
    with open(file_path, 'r') as f:
        content = f.read()
        documents.append({
            "text": content,
            "source": file_path.name
        })
```

### 3. Adjust Retrieval Parameters

```python
# Retrieve more documents
results = qdrant.search(
    collection_name="knowledge_base",
    query_vector=query_embedding,
    limit=10  # Get top 10 instead of top 3
)

# Filter by score threshold
relevant_results = [
    r for r in results
    if r.score > 0.7  # Only highly relevant docs
]
```

### 4. Customize Chunk Size

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Smaller chunks = more precise retrieval
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Adjust this
    chunk_overlap=30,    # And this
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_text(document)
```

### 5. Choose Different Embedding Model

```python
from sentence_transformers import SentenceTransformer

# Faster, smaller
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim

# Better quality
model = SentenceTransformer('all-mpnet-base-v2')  # 768 dim

# Domain-specific
model = SentenceTransformer('allenai-specter')   # Scientific papers
```

---

## ⚡ Common Issues & Solutions

### Issue: "No relevant documents found"
**Cause**: Query doesn't match document content semantically

**Solutions**:
- Rephrase your query
- Lower the similarity threshold
- Check if documents are actually in the database
- Try hybrid search (semantic + keyword)

```python
# Lower threshold
results = [r for r in results if r.score > 0.5]  # Instead of 0.7
```

### Issue: "Answers don't match my documents"
**Cause**: Retrieved documents not relevant or wrong chunks

**Solutions**:
- Increase number of retrieved documents (k=5 or k=10)
- Adjust chunk size (try 500-700 tokens)
- Add more overlap between chunks (50-100 tokens)
- Verify embeddings were computed correctly

### Issue: "Slow retrieval times"
**Cause**: Large vector database or inefficient search

**Solutions**:
- Use Qdrant's HNSW indexing (already default)
- Reduce number of documents retrieved (k=3 instead of k=10)
- Use quantization for embeddings
- Consider persistent Qdrant with proper indexing

### Issue: "Out of memory errors"
**Cause**: Too many embeddings in memory

**Solutions**:
- Use persistent Qdrant (not `:memory:`)
- Batch document processing
- Use smaller embedding model (384 dim instead of 768)

### Issue: "LLM ignores retrieved context"
**Cause**: Poor prompt engineering

**Solutions**:
- Make prompt explicit: "Answer ONLY based on context"
- Format context clearly with source labels
- Use a better LLM (GPT-4 instead of GPT-3.5)
- Add few-shot examples of good answers

```python
prompt = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
```

### Issue: "Hallucinations in answers"
**Cause**: LLM making up information not in context

**Solutions**:
- Use explicit grounding prompt (see above)
- Lower LLM temperature (0.0 for factual answers)
- Implement confidence scoring
- Filter low-relevance retrievals

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Implementation Details**: See [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🎓 Learning Path

1. ✅ **Understand the concept**: Read "Understanding RAG in 30 Seconds" above
2. ✅ **Run basic example**: See RAG in action with simple demo
3. ✅ **Examine the code**: Look at how retrieval and generation work
4. ✅ **Try your own questions**: Experiment with different queries
5. ✅ **Add your documents**: Replace demo docs with your own
6. ✅ **Tune parameters**: Adjust chunk size, k, similarity threshold
7. ✅ **Monitor quality**: Track retrieval and answer accuracy
8. ✅ **Deploy to production**: Use persistent Qdrant and proper error handling

---

## 🌟 Pro Tips

### 1. Chunk Size Sweet Spot
- **Too small** (50-100 tokens): Loses context, many irrelevant chunks
- **Too large** (2000+ tokens): Less precise, wastes tokens
- **Just right** (300-700 tokens): Balances context and precision

### 2. Always Add Metadata
```python
payload = {
    "text": chunk_content,
    "source": "document.pdf",
    "page": 5,
    "section": "Introduction",
    "date": "2024-01-30"
}
```
Why? Enables filtering, citation, and versioning!

### 3. Implement Hybrid Search
Combine semantic (meaning) + keyword (exact match):
```python
# Search for semantically similar docs
semantic_results = vector_search(query)

# Filter by keyword
hybrid_results = [
    r for r in semantic_results
    if keyword in r.payload["text"].lower()
]
```

### 4. Cache Common Queries
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def rag_query_cached(question: str):
    return rag_query(question)
```

### 5. Monitor Retrieval Quality
```python
def log_retrieval_metrics(results):
    print(f"Retrieved {len(results)} documents")
    print(f"Avg similarity: {sum(r.score for r in results) / len(results):.2f}")
    print(f"Top score: {results[0].score:.2f}")

    # Alert if quality is poor
    if results[0].score < 0.6:
        logger.warning("Low relevance retrieval!")
```

### 6. Handle Edge Cases
```python
def safe_rag(question: str):
    results = retriever.search(question)

    # No results
    if not results:
        return "I don't have any documents about that topic."

    # Low relevance
    if results[0].score < 0.5:
        return "I found some documents but they may not be relevant."

    # Normal processing
    return generate_answer(question, results)
```

### 7. Use Overlapping Chunks
```python
# Without overlap: may split important info
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0  # ❌ Bad
)

# With overlap: preserves context across boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50  # ✅ Good (10% overlap)
)
```

### 8. Structure Your Prompts
```python
# ❌ Weak prompt
prompt = f"Context: {context}\n\nQuestion: {question}"

# ✅ Strong prompt
prompt = f"""You are a helpful assistant. Use the context below to answer the question.

IMPORTANT:
- Base your answer ONLY on the provided context
- If the answer is not in the context, say so
- Cite the source of your information

Context:
{context}

Question: {question}

Answer:"""
```

### 9. Test with Edge Cases
Always test your RAG system with:
- Questions with no relevant docs
- Ambiguous questions
- Questions requiring multiple docs
- Questions with contradictory info in docs
- Questions outside your knowledge base

### 10. Version Your Embeddings
```python
# Tag embeddings with model version
payload = {
    "text": chunk,
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_version": "2024-01-30"
}

# When changing models, re-index everything
```

---

## 🔬 Quick Experiments to Try

### Experiment 1: Chunk Size Impact
```python
# Try different sizes
for chunk_size in [200, 500, 1000]:
    chunks = split_documents(docs, chunk_size)
    index_documents(chunks)
    answer = rag_query("Your question")
    print(f"Chunk size {chunk_size}: {answer}")
```

### Experiment 2: Number of Retrieved Docs
```python
# Try different k values
for k in [1, 3, 5, 10]:
    results = search(query, limit=k)
    answer = generate(question, results)
    print(f"k={k}: {answer}")
```

### Experiment 3: Embedding Models
```python
# Compare models
models = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'paraphrase-MiniLM-L6-v2'
]

for model_name in models:
    model = SentenceTransformer(model_name)
    # Index, search, evaluate
    print(f"{model_name}: accuracy={score}")
```

### Experiment 4: Prompt Variations
```python
prompts = [
    "Answer: {question}\n\nContext: {context}",
    "Context: {context}\n\nQuestion: {question}",
    "Using the context, answer: {question}\n\n{context}"
]

for prompt_template in prompts:
    answer = generate_with_prompt(prompt_template)
    print(f"Prompt variation: {answer}")
```

---

## 🎯 Production Checklist

Before deploying RAG to production:

- [ ] **Testing**: Evaluate on diverse questions and edge cases
- [ ] **Monitoring**: Log retrieval scores and response quality
- [ ] **Error handling**: Graceful failures for no results/low relevance
- [ ] **Caching**: Cache common queries for speed
- [ ] **Persistent storage**: Use disk-based Qdrant, not in-memory
- [ ] **Backup**: Regular backups of vector database
- [ ] **Versioning**: Track embedding model and document versions
- [ ] **Rate limiting**: Prevent abuse of retrieval system
- [ ] **Feedback loops**: Collect thumbs up/down on answers
- [ ] **Metrics dashboard**: Track latency, accuracy, usage
- [ ] **Documentation**: Clear instructions for adding/updating documents
- [ ] **Access control**: Secure sensitive documents appropriately

---

**Happy RAGging! 🔍📚**

For detailed information and advanced techniques, see the full [README.md](./README.md).
