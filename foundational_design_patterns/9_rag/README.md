# RAG (Retrieval-Augmented Generation)

## Overview

**Retrieval-Augmented Generation (RAG)** is a pattern that enhances LLM responses by retrieving relevant information from external knowledge sources before generating answers. Instead of relying solely on the model's parametric knowledge (learned during training), RAG dynamically fetches contextually relevant documents or data chunks to ground responses in factual, up-to-date information.

RAG combines the power of information retrieval systems with the generation capabilities of LLMs, creating a hybrid approach where the model answers questions based on retrieved context rather than pure memorization. This makes it particularly valuable for domain-specific applications, frequently updated information, and reducing hallucinations.

## Why Use This Pattern?

Traditional LLM approaches have significant limitations:

- **Knowledge cutoff**: LLMs only know information from their training data, which becomes outdated
- **Hallucinations**: Models may generate plausible-sounding but incorrect information
- **Domain limitations**: General models lack deep expertise in specialized domains
- **No source attribution**: Pure generation cannot cite sources or provide evidence
- **Static knowledge**: Cannot access real-time or organization-specific information
- **Memory constraints**: Cannot process entire document collections in a single prompt

RAG solves these by:
- **Dynamic knowledge access**: Retrieves current, relevant information on-demand
- **Grounded responses**: Answers are based on actual retrieved documents, reducing hallucinations
- **Source attribution**: Can cite specific documents or passages used in generation
- **Domain specialization**: Works with custom knowledge bases (docs, wikis, databases)
- **Scalability**: Handles massive document collections through efficient retrieval
- **Updatable knowledge**: Add new documents without retraining the model
- **Transparency**: Shows what information was used to generate each response

### Example: Question Answering with RAG

```
Without RAG (Pure LLM):
User: "What is our company's vacation policy for employees hired after 2024?"
LLM: "Typically companies offer 10-15 days of vacation annually..."
→ Generic answer, possibly incorrect, no company-specific details

With RAG (Retrieval + Generation):
User: "What is our company's vacation policy for employees hired after 2024?"

[Retrieval Step]
→ Searches company policy documents
→ Finds: "HR_Policy_2024.pdf", "Employee_Handbook_v3.pdf"
→ Extracts relevant sections about vacation policy

[Augmentation Step]
→ Provides retrieved context to LLM:
  "Context: From Employee Handbook: 'Employees hired after January 1, 2024
   receive 15 days of vacation in year one, increasing to 20 days after 3 years.
   Vacation accrues monthly...'"

[Generation Step]
LLM: "According to the Employee Handbook (v3), employees hired after 2024 receive
     15 days of vacation in their first year, which increases to 20 days after
     3 years of service. Vacation time accrues monthly.

     Source: Employee_Handbook_v3.pdf, Section 4.2"
```

## How It Works

RAG operates in three key stages:

### 1. Retrieve
Query the knowledge base to find relevant documents or passages:
- Convert user query to embedding vector
- Search vector database for semantically similar content
- Return top-k most relevant documents

### 2. Augment
Combine retrieved context with the user's original query:
- Format retrieved documents into prompt context
- Add source metadata (document names, page numbers)
- Structure information for optimal LLM comprehension

### 3. Generate
LLM produces answer grounded in retrieved context:
- Uses retrieved information as factual basis
- Synthesizes multiple sources if needed
- Can cite specific sources in response

```
┌─────────────────────────────────────────────────────┐
│                   User Query                         │
│          "What is quantum entanglement?"            │
└───────────────────┬─────────────────────────────────┘
                    ↓
            ┌───────────────┐
            │   Embedding   │ Convert query to vector
            │     Model     │ [0.23, -0.45, 0.12, ...]
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ Vector Search │ Find similar documents
            │  (Qdrant DB)  │ Similarity: 0.89, 0.85, 0.82
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   Retrieved   │ Top 3 documents:
            │   Documents   │ - physics_textbook.pdf
            └───────┬───────┘   - quantum_mechanics.md
                    ↓           - research_paper_2024.pdf
            ┌───────────────┐
            │  Augmentation │ Combine query + context
            │   (Prompt)    │ "Context: [docs...] Question: ..."
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   LLM Model   │ Generate grounded answer
            │  (GPT-4, etc) │ with source citations
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ Final Answer  │ "Quantum entanglement is a phenomenon
            │  + Sources    │  where particles become correlated..."
            └───────────────┘  Sources: physics_textbook.pdf, p.142
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Enterprise knowledge bases**: Internal documentation, wikis, policies, procedures
- **Customer support**: FAQs, troubleshooting guides, product documentation
- **Research assistance**: Academic papers, technical reports, scientific literature
- **Legal/Compliance**: Contracts, regulations, case law, policy documents
- **Medical/Healthcare**: Clinical guidelines, research papers, patient records
- **News and current events**: Recent articles, real-time information
- **Code documentation**: API docs, code repositories, technical specifications
- **E-commerce**: Product catalogs, reviews, specifications
- **Education**: Textbooks, course materials, learning resources

### ❌ When NOT to Use

- **Simple factual queries**: Questions answerable by LLM's parametric knowledge
- **Creative generation**: Original content creation (stories, marketing copy)
- **Mathematical reasoning**: Pure computation without need for reference material
- **Small context**: When all information fits in a single prompt
- **Real-time data streams**: When data changes too rapidly for indexing
- **Highly structured queries**: SQL/database queries are more appropriate
- **Privacy-sensitive data**: When retrieval systems cannot guarantee data isolation

## Rule of Thumb

**Use RAG when:**
1. Information is **too large** to fit in a single prompt (>100K tokens)
2. Knowledge **changes frequently** and needs to stay current
3. You need **source attribution** and citations
4. Working with **domain-specific** or proprietary knowledge
5. Want to **reduce hallucinations** with grounded responses
6. Need to **update knowledge** without retraining models

**Don't use RAG when:**
1. Information is small enough for prompt engineering
2. Task requires **creative generation** without factual grounding
3. Query doesn't benefit from external knowledge
4. Retrieval latency is unacceptable for your use case
5. Information is highly structured (use databases instead)

## Core Components

### 1. Document Store (Knowledge Base)

The source of truth containing documents to retrieve from:
- Text files (PDF, TXT, Markdown, HTML)
- Structured data (JSON, CSV, databases)
- Code repositories
- Web pages and APIs

**Processing steps:**
- **Chunking**: Split documents into manageable pieces (200-1000 tokens)
- **Metadata**: Attach source, date, author, category information
- **Cleaning**: Remove irrelevant content, format consistently

### 2. Embedding Model

Converts text into dense vector representations:
- **Purpose**: Enables semantic similarity search
- **Examples**: OpenAI text-embedding-3, Sentence-Transformers, Cohere
- **Dimensions**: Typically 384-3072 dimensions
- **Key property**: Similar content has similar embeddings

```python
# Example embedding
query = "What is photosynthesis?"
embedding = embedding_model.encode(query)
# Result: [0.23, -0.45, 0.12, ..., 0.67]  (768 dimensions)
```

### 3. Vector Store (Vector Database)

Stores embeddings and enables fast similarity search:
- **Qdrant**: Open-source, feature-rich, Python-native
- **Pinecone**: Managed cloud service
- **Weaviate**: GraphQL-based vector search
- **Chroma**: Simple, embedded database
- **FAISS**: Facebook's similarity search library

**Key operations:**
- **Upsert**: Add/update document embeddings
- **Search**: Find top-k similar vectors
- **Filter**: Apply metadata filters
- **Hybrid search**: Combine vector + keyword search

### 4. Retriever

Orchestrates the search process:
- Converts queries to embeddings
- Executes vector similarity search
- Re-ranks results (optional)
- Returns relevant documents with scores

### 5. Generator (LLM)

Produces final answer using retrieved context:
- Receives augmented prompt (query + context)
- Generates response grounded in provided documents
- Can cite sources and quote passages
- Handles multi-document synthesis

## Implementation Approaches

### Approach 1: Basic RAG with Qdrant (In-Memory)

Simple implementation for quick prototyping:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize components
qdrant = QdrantClient(":memory:")  # In-memory for testing
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = OpenAI()

# Create collection
qdrant.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Add documents
documents = [
    "Paris is the capital of France.",
    "The Eiffel Tower is 330 meters tall.",
    "Python is a programming language."
]

points = [
    PointStruct(
        id=idx,
        vector=embedding_model.encode(doc).tolist(),
        payload={"text": doc}
    )
    for idx, doc in enumerate(documents)
]

qdrant.upsert(collection_name="knowledge_base", points=points)

# RAG Query
def rag_query(question: str) -> str:
    # Retrieve
    query_vector = embedding_model.encode(question).tolist()
    results = qdrant.search(
        collection_name="knowledge_base",
        query_vector=query_vector,
        limit=3
    )

    # Augment
    context = "\n".join([hit.payload["text"] for hit in results])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Generate
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Use it
answer = rag_query("What is the height of the Eiffel Tower?")
print(answer)
```

### Approach 2: Semantic Search with LangChain

Production-ready with LangChain integration:

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load and chunk documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Qdrant.from_documents(
    chunks,
    embeddings,
    location=":memory:",
    collection_name="kb"
)

# Create RAG chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
answer = qa_chain.invoke({"query": "What is...?"})
print(answer["result"])
```

### Approach 3: Advanced RAG with Re-ranking

Multi-stage retrieval for better precision:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Add re-ranker
compressor = CohereRerank(model="rerank-english-v2.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use in chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

### Approach 4: Hybrid Search (Vector + Keyword)

Combine semantic and lexical search:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Hybrid search function
def hybrid_search(query: str, keywords: list = None):
    query_vector = embedding_model.encode(query).tolist()

    # Optional keyword filter
    query_filter = None
    if keywords:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="keywords",
                    match=MatchValue(value=kw)
                ) for kw in keywords
            ]
        )

    results = qdrant.search(
        collection_name="knowledge_base",
        query_vector=query_vector,
        query_filter=query_filter,
        limit=5
    )

    return results
```

## Key Benefits

### 📚 Dynamic Knowledge Access
- **Always current**: Update knowledge base without retraining models
- **Scalable**: Handle millions of documents efficiently
- **Cost-effective**: No expensive fine-tuning required

### 🎯 Improved Accuracy
- **Grounded responses**: Answers based on actual documents, not hallucinations
- **Domain expertise**: Specialize in your specific domain instantly
- **Factual correctness**: Retrieval ensures information is accurate

### 🔍 Transparency and Trust
- **Source attribution**: Cite specific documents and passages
- **Auditability**: Trace answers back to source material
- **Verifiability**: Users can check original sources

### 💡 Flexibility
- **Multiple domains**: One system, many knowledge bases
- **Custom data**: Use proprietary, internal documents
- **Multi-modal**: Support text, code, structured data

### ⚡ Efficiency
- **Token optimization**: Only send relevant context to LLM
- **Reduced costs**: Smaller prompts = lower API costs
- **Faster responses**: Retrieval faster than processing entire corpus

## Trade-offs

### ⚠️ Retrieval Quality Dependency

**Issue**: Poor retrieval = poor answers, regardless of LLM quality

**Impact**: If relevant docs aren't retrieved, LLM can't answer correctly

**Mitigation**:
- Use high-quality embedding models
- Implement hybrid search (vector + keyword)
- Add re-ranking stage
- Tune chunk size and overlap
- Monitor retrieval metrics (precision@k, recall@k)

### 🕐 Added Latency

**Issue**: Retrieval adds 50-500ms before LLM generation

**Impact**: Slower than pure LLM responses

**Mitigation**:
- Use fast vector databases (Qdrant, Pinecone)
- Implement caching for common queries
- Pre-compute embeddings
- Optimize retrieval parameters (fewer docs)
- Use async retrieval

### 📊 Chunking Challenges

**Issue**: Document splitting can break context or split key information

**Impact**: Retrieved chunks may lack necessary context

**Mitigation**:
- Use recursive chunking with overlap
- Implement semantic chunking (split on topics)
- Keep metadata (document title, section)
- Experiment with chunk sizes (200-1000 tokens)
- Use parent-child document relationships

### 💾 Storage Requirements

**Issue**: Vector embeddings require significant storage (4KB-12KB per document)

**Impact**: Large document sets need substantial disk/memory

**Mitigation**:
- Use quantization for embeddings
- Implement tiered storage
- Regular cleanup of outdated documents
- Use cloud-managed vector databases

### 🔄 Knowledge Base Maintenance

**Issue**: Keeping documents up-to-date requires ongoing effort

**Impact**: Outdated information leads to incorrect answers

**Mitigation**:
- Implement automated ingestion pipelines
- Version documents with timestamps
- Regular audits and updates
- Document lifecycle management

## Best Practices

### 1. Document Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Good chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # Optimal: 200-1000 tokens
    chunk_overlap=50,         # 10% overlap preserves context
    separators=["\n\n", "\n", ". ", " ", ""],  # Respect structure
    length_function=len
)

# Add metadata to chunks
for chunk in chunks:
    chunk.metadata["source"] = "document.pdf"
    chunk.metadata["page"] = page_number
    chunk.metadata["section"] = section_title
```

### 2. Embedding Model Selection

```python
# For English documents
from sentence_transformers import SentenceTransformer

# Fast and lightweight (384 dim)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Better quality (768 dim)
model = SentenceTransformer('all-mpnet-base-v2')

# Domain-specific
model = SentenceTransformer('allenai-specter')  # Scientific papers

# Multilingual
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### 3. Retrieval Configuration

```python
# Configure retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",         # Maximum Marginal Relevance
    search_kwargs={
        "k": 5,                # Top-5 most relevant
        "fetch_k": 20,         # Fetch 20, then MMR to 5
        "lambda_mult": 0.7     # Diversity vs relevance
    }
)
```

### 4. Prompt Engineering for RAG

```python
rag_prompt = """You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I cannot answer this based on the provided information."

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the context above
- Cite specific sources in your answer
- If information is missing, acknowledge it
- Be concise but complete

Answer:"""
```

### 5. Source Attribution

```python
# Track and cite sources
def generate_with_sources(question: str):
    results = retriever.get_relevant_documents(question)

    context = "\n\n".join([
        f"[Source {i+1}]: {doc.page_content}\n(From: {doc.metadata['source']})"
        for i, doc in enumerate(results)
    ])

    prompt = rag_prompt.format(context=context, question=question)
    answer = llm.invoke(prompt)

    sources = [
        f"{doc.metadata['source']}, Page {doc.metadata.get('page', 'N/A')}"
        for doc in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": calculate_confidence(results)
    }
```

### 6. Error Handling

```python
def robust_rag_query(question: str):
    try:
        # Retrieve
        results = retriever.get_relevant_documents(question)

        # Check if results are relevant
        if not results or results[0].metadata.get("score", 0) < 0.5:
            return "I couldn't find relevant information to answer this question."

        # Generate
        answer = generate_with_sources(question)
        return answer

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return "I encountered an error processing your question. Please try again."
```

## Performance Metrics

Track these metrics to optimize RAG systems:

### Retrieval Metrics
- **Precision@K**: % of retrieved docs that are relevant
- **Recall@K**: % of relevant docs that were retrieved
- **Mean Reciprocal Rank (MRR)**: Position of first relevant document
- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality)
- **Retrieval latency**: Time to fetch documents (target: <200ms)

### Generation Metrics
- **Answer accuracy**: Human evaluation or automated (LLM-as-judge)
- **Faithfulness**: Answer consistency with retrieved context
- **Citation quality**: Correct source attribution
- **Hallucination rate**: Answers not supported by context
- **Response completeness**: Addresses all parts of question

### System Metrics
- **End-to-end latency**: Total time (retrieval + generation)
- **Token usage**: Context + prompt + response tokens
- **Cache hit rate**: % of queries served from cache
- **User satisfaction**: Thumbs up/down feedback

### Example Monitoring

```python
import time

def monitored_rag_query(question: str):
    start_time = time.time()

    # Retrieval
    retrieval_start = time.time()
    results = retriever.get_relevant_documents(question)
    retrieval_time = time.time() - retrieval_start

    # Generation
    generation_start = time.time()
    answer = llm.invoke(augmented_prompt)
    generation_time = time.time() - generation_start

    total_time = time.time() - start_time

    metrics = {
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "num_docs_retrieved": len(results),
        "avg_relevance_score": sum(r.metadata["score"] for r in results) / len(results)
    }

    log_metrics(metrics)
    return answer
```

## Example Scenarios

### Scenario 1: Technical Documentation Q&A

```
Document Collection:
- API_Reference.md
- Getting_Started.pdf
- Advanced_Topics.md
- FAQ.txt

User: "How do I authenticate API requests?"

[Retrieval]
→ Finds: API_Reference.md (score: 0.92)
   "Authentication: Include Bearer token in Authorization header.
    Example: Authorization: Bearer YOUR_TOKEN_HERE"

[Generation]
Answer: "To authenticate API requests, include a Bearer token in the
        Authorization header of your HTTP request. Format it as:

        Authorization: Bearer YOUR_TOKEN_HERE

        Source: API_Reference.md, Authentication section"
```

### Scenario 2: Company Policy Assistant

```
Document Collection:
- Employee_Handbook_2024.pdf
- Remote_Work_Policy.pdf
- PTO_Guidelines.pdf
- Benefits_Overview.pdf

User: "Can I work remotely from another country?"

[Retrieval]
→ Finds: Remote_Work_Policy.pdf (score: 0.88)
   "International remote work requires approval from HR and legal.
    Submit request 60 days in advance. Tax implications apply."

[Generation]
Answer: "Yes, but it requires advance approval. According to the Remote Work
        Policy, international remote work requires approval from both HR and
        Legal departments. You must submit your request at least 60 days in
        advance, and be aware that tax implications may apply.

        Source: Remote_Work_Policy.pdf, Section 3.2"
```

### Scenario 3: Research Paper Assistant

```
Document Collection:
- 500 scientific papers on climate change
- Research database with abstracts and citations

User: "What are the latest findings on ocean acidification rates?"

[Retrieval]
→ Finds 5 papers:
   1. "Ocean pH decreased by 0.1 units since pre-industrial era" (2023)
   2. "Acidification accelerating in polar regions" (2024)
   3. "Impact on coral reef ecosystems" (2023)

[Generation]
Answer: "Recent research shows ocean acidification is accelerating. Key findings:

        1. Ocean pH has decreased by approximately 0.1 units since the
           pre-industrial era, representing a 30% increase in acidity.

        2. Polar regions are experiencing particularly rapid acidification rates,
           up to 2-3x faster than global average.

        3. Coral reef ecosystems show measurable stress responses to current
           acidification levels.

        Sources:
        - Smith et al. (2023) Nature Climate, "Global Ocean pH Trends"
        - Johnson et al. (2024) Science, "Polar Ocean Acidification"
        - Lee et al. (2023) Marine Biology, "Coral Response to pH Changes""
```

### Scenario 4: Code Documentation Assistant

```
Document Collection:
- Python codebase with docstrings
- README files
- Architecture documentation

User: "How does the caching mechanism work in the data loader?"

[Retrieval]
→ Finds: data_loader.py
   class DataLoader:
       def __init__(self, cache_ttl=3600):
           """Initialize with LRU cache. TTL in seconds."""
           self.cache = LRUCache(maxsize=1000, ttl=cache_ttl)

[Generation]
Answer: "The data loader uses an LRU (Least Recently Used) cache with
        configurable time-to-live (TTL). By default, cache entries expire
        after 3600 seconds (1 hour). The cache has a maximum size of 1000
        entries, after which the least recently used items are evicted.

        You can customize the TTL when initializing:
        loader = DataLoader(cache_ttl=7200)  # 2 hours

        Source: data_loader.py, DataLoader class"
```

## Advanced Patterns

### 1. Multi-Query RAG

Generate multiple query variations for better retrieval:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Generate multiple query perspectives
base_retriever = vectorstore.as_retriever()
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# Single query → multiple searches → deduplicated results
results = multi_query_retriever.get_relevant_documents(
    "What causes climate change?"
)
# Internally generates:
# - "What are the causes of climate change?"
# - "Why is the Earth's climate changing?"
# - "What factors contribute to global warming?"
```

### 2. Hierarchical RAG

Retrieve at multiple levels (document → section → paragraph):

```python
# Parent-child retrieval
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Create parent chunks
parent_chunks = parent_splitter.split_documents(documents)

# Create child chunks with parent references
for parent in parent_chunks:
    child_chunks = child_splitter.split_text(parent.page_content)
    for child in child_chunks:
        child.metadata["parent_id"] = parent.metadata["id"]
        child.metadata["parent_content"] = parent.page_content

# Retrieve children, return parents for context
```

### 3. Agentic RAG

Combine RAG with ReAct-style reasoning:

```python
# Agent decides when to retrieve and what to search for
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information."""
    results = retriever.get_relevant_documents(query)
    return format_results(results)

agent = create_react_agent(llm, tools=[search_knowledge_base])

# Agent can make multiple retrievals with refined queries
result = agent.invoke({
    "messages": [("user", "Compare the features of Product A and Product B")]
})

# Agent might:
# 1. Search for "Product A features"
# 2. Search for "Product B features"
# 3. Synthesize comparison
```

### 4. Hypothetical Document Embeddings (HyDE)

Generate hypothetical answer, then search for similar documents:

```python
def hyde_rag(question: str):
    # Step 1: Generate hypothetical answer
    hyde_prompt = f"Write a detailed answer to: {question}"
    hypothetical_answer = llm.invoke(hyde_prompt)

    # Step 2: Use hypothetical answer to search
    results = retriever.get_relevant_documents(hypothetical_answer)

    # Step 3: Generate real answer from retrieved docs
    context = format_docs(results)
    final_answer = llm.invoke(f"Context: {context}\n\nQuestion: {question}")

    return final_answer
```

### 5. Self-RAG (Self-Reflective RAG)

Agent critiques and improves its own retrieval and generation:

```python
def self_rag(question: str):
    # Initial retrieval and generation
    docs = retriever.get_relevant_documents(question)
    answer = generate_answer(question, docs)

    # Self-reflection
    reflection_prompt = f"""
    Question: {question}
    Answer: {answer}
    Retrieved documents: {format_docs(docs)}

    Evaluate:
    1. Are the retrieved documents relevant?
    2. Does the answer address the question?
    3. Is additional information needed?
    """

    reflection = llm.invoke(reflection_prompt)

    # If needed, retrieve more or refine
    if "additional information needed" in reflection.lower():
        refined_query = extract_refined_query(reflection)
        additional_docs = retriever.get_relevant_documents(refined_query)
        answer = generate_answer(question, docs + additional_docs)

    return answer
```

## Comparison with Related Patterns

| Pattern | Knowledge Source | Latency | Use Case |
|---------|-----------------|---------|----------|
| **RAG** | External retrieval | Medium (200-1000ms) | Large knowledge bases |
| **Fine-tuning** | Model parameters | Low (50-200ms) | Fixed domain, high volume |
| **Prompt stuffing** | Entire context in prompt | Medium (varies) | Small datasets (<100K tokens) |
| **Tool use** | External APIs/tools | High (500-2000ms) | Real-time data, actions |
| **Knowledge graphs** | Graph database | Medium | Relationship-heavy queries |
| **Semantic cache** | Cached responses | Very low (10-50ms) | Repeated queries |

### RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge updates** | Easy (add documents) | Hard (retrain model) |
| **Setup time** | Minutes | Hours to days |
| **Cost** | Low (storage + API) | High (training compute) |
| **Transparency** | High (see sources) | Low (black box) |
| **Latency** | Higher | Lower |
| **Best for** | Changing knowledge | Fixed domain expertise |

## Common Pitfalls

### 1. Poor Chunking Strategy

**Problem**: Chunks too large (lose precision) or too small (lose context)

**Solution**:
- Experiment with chunk sizes: 200-1000 tokens
- Use overlap: 10-20% of chunk size
- Respect document structure (paragraphs, sections)
- Include metadata for context

### 2. Inadequate Metadata

**Problem**: Cannot filter or trace back to sources

**Solution**:
- Add source file, page number, section
- Include timestamps for versioning
- Tag documents with categories, topics
- Store document hierarchy

### 3. Embedding Model Mismatch

**Problem**: Different models for indexing vs. querying

**Solution**:
- Use the same embedding model consistently
- Version your embeddings when changing models
- Re-index if you change embedding models

### 4. No Retrieval Quality Monitoring

**Problem**: Don't know if retrieval is working well

**Solution**:
- Log relevance scores for retrieved documents
- Monitor precision@k and recall@k
- Implement human feedback loops
- Create evaluation datasets

### 5. Ignoring Failed Retrievals

**Problem**: LLM hallucinates when no relevant docs found

**Solution**:
```python
def safe_rag(question: str):
    results = retriever.get_relevant_documents(question)

    if not results or max(r.metadata["score"] for r in results) < 0.5:
        return "I don't have enough information to answer this question."

    return generate_answer(question, results)
```

### 6. Context Overflow

**Problem**: Too many retrieved documents exceed LLM context window

**Solution**:
- Limit retrieved documents (k=3-5)
- Use map-reduce for many documents
- Implement summarization pipeline
- Use models with larger context windows

### 7. Stale or Duplicate Content

**Problem**: Outdated information or redundant documents

**Solution**:
- Implement document versioning
- Regular cleanup and deduplication
- Use timestamps for freshness filtering
- Automated content refresh pipelines

## Conclusion

Retrieval-Augmented Generation represents a fundamental shift in how we build LLM applications, moving from static knowledge encoded in model weights to dynamic, updateable knowledge bases. By combining the power of semantic search with generative AI, RAG enables accurate, attributable, and adaptable AI systems.

**Use RAG when:**
- Working with large, evolving knowledge bases
- Need source attribution and transparency
- Domain-specific or proprietary information
- Want to reduce hallucinations
- Require up-to-date information
- Cannot fit all context in prompts

**Implementation checklist:**
- ✅ Choose appropriate vector database (Qdrant recommended)
- ✅ Select embedding model matching your domain
- ✅ Implement effective document chunking strategy
- ✅ Add comprehensive metadata to chunks
- ✅ Configure retrieval parameters (k, similarity threshold)
- ✅ Design prompts that emphasize grounding in context
- ✅ Implement source attribution in responses
- ✅ Monitor retrieval and generation metrics
- ✅ Set up feedback loops for continuous improvement
- ✅ Handle edge cases (no results, low confidence)

**Key Takeaways:**
- 🔍 RAG = Retrieve → Augment → Generate
- 📚 Grounds responses in factual documents
- 🎯 Reduces hallucinations significantly
- 📝 Enables source attribution and transparency
- 🔄 Updateable without retraining models
- ⚡ More cost-effective than fine-tuning for knowledge updates
- 🎨 Highly customizable for specific domains
- 📊 Monitor retrieval quality as much as generation quality

**Common gotchas:**
- Chunk size matters: too big or too small both hurt performance
- Embedding model must stay consistent
- Always monitor retrieval relevance scores
- Include metadata for filtering and attribution
- Handle cases when no relevant documents are found
- Consider latency impacts of retrieval step

---

*RAG transforms LLMs from knowledge repositories into knowledge navigators—dynamically accessing the right information at the right time to provide accurate, grounded, and trustworthy responses.*
