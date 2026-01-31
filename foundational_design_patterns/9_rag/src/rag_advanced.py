"""
RAG Pattern: Advanced Implementation with Re-ranking
This example demonstrates advanced RAG techniques including:
- MMR (Maximal Marginal Relevance) for diverse retrieval
- Re-ranking of retrieved documents
- LangChain QA chain for sophisticated generation
- Detailed relevance scoring and analysis
"""

import os
import sys
from typing import List, Tuple, Optional

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))


class AdvancedRAG:
    """Advanced RAG implementation with MMR and re-ranking capabilities."""

    def __init__(self, collection_name: str = "advanced_ai_docs"):
        """Initialize the advanced RAG system.

        Args:
            collection_name: Name of the Qdrant collection
        """
        print("\n🔧 Initializing Advanced RAG System...")

        # Initialize OpenAI embeddings (better quality than sentence-transformers)
        print("   Loading OpenAI embeddings...")
        self.embeddings = OpenAIEmbeddings()

        # Initialize Qdrant in-memory
        print("   Setting up Qdrant vector database (in-memory)")
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = collection_name

        # Initialize LLM for generation
        print("   Connecting to OpenAI for generation")
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")

        # Store documents for later retrieval
        self.documents: List[Document] = []
        self.doc_embeddings: Optional[np.ndarray] = None

        print("✅ Advanced RAG system initialized!\n")

    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """Add documents to the vector database.

        Args:
            documents: List of text documents to add
            metadata: Optional metadata for each document
        """
        print(f"\n📝 Adding {len(documents)} documents to the knowledge base...")

        # Create Document objects
        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]

        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadata)
        ]

        # Generate embeddings for all documents
        print("   Generating embeddings...")
        doc_texts = [doc.page_content for doc in self.documents]
        embeddings_list = self.embeddings.embed_documents(doc_texts)
        self.doc_embeddings = np.array(embeddings_list)

        # Get embedding dimension
        embedding_dim = len(embeddings_list[0])

        # Create Qdrant collection
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

        # Create points for Qdrant
        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={"text": doc.page_content, **doc.metadata}
            )
            for idx, (doc, embedding) in enumerate(zip(self.documents, embeddings_list))
        ]

        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"✅ Successfully indexed {len(documents)} documents\n")

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            List of (Document, score) tuples
        """
        print(f"🔍 RETRIEVAL PHASE (Initial)")
        print(f"   Query: '{query}'")
        print(f"   Retrieving top {k} documents with similarity search...")

        if not self.documents:
            raise ValueError("No documents loaded. Call add_documents() first.")

        # Embed the query
        query_embedding = self.embeddings.embed_query(query)

        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k
        ).points

        # Convert to Document objects with scores
        results = []
        for result in search_results:
            if result.payload is None:
                continue
            # Reconstruct document from payload
            doc = Document(
                page_content=result.payload["text"],
                metadata={k: v for k, v in result.payload.items() if k != "text"}
            )
            results.append((doc, result.score))

        print(f"\n   📊 Initial Retrieval Results:")
        for idx, (doc, score) in enumerate(results, 1):
            print(f"   {idx}. Score: {score:.4f} | {doc.page_content[:80]}...")

        return results

    def mmr_rerank(
        self,
        query: str,
        initial_docs: List[Tuple[Document, float]],
        lambda_mult: float = 0.5,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Re-rank documents using Maximal Marginal Relevance.

        MMR balances relevance to the query with diversity among results,
        reducing redundancy in the retrieved documents.

        Args:
            query: The search query
            initial_docs: Initial retrieved documents with scores
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
            k: Number of documents to return after re-ranking

        Returns:
            Re-ranked list of (Document, score) tuples
        """
        print(f"\n🎯 RE-RANKING PHASE (MMR)")
        print(f"   Lambda: {lambda_mult} (1.0=relevance, 0.0=diversity)")
        print(f"   Selecting top {k} diverse documents...")

        if not initial_docs:
            return []

        # Extract documents and their embeddings
        docs = [doc for doc, _ in initial_docs]

        # Get query embedding
        query_embedding = np.array(self.embeddings.embed_query(query))

        # Get document embeddings
        doc_embeddings = np.array([
            self.embeddings.embed_query(doc.page_content) for doc in docs
        ])

        # Compute similarity to query
        query_similarities = self._cosine_similarity(query_embedding, doc_embeddings)

        # MMR algorithm
        selected_indices: list[int] = []
        remaining_indices = list(range(len(docs)))

        for _ in range(min(k, len(docs))):
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance score
                relevance = query_similarities[idx]

                # Diversity score (max similarity to already selected docs)
                if selected_indices:
                    selected_embeddings = doc_embeddings[selected_indices]
                    diversity = -np.max(
                        self._cosine_similarity(doc_embeddings[idx], selected_embeddings)
                    )
                else:
                    diversity = 0

                # MMR score combines relevance and diversity
                mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Build re-ranked results
        reranked_docs = [
            (docs[idx], query_similarities[idx]) for idx in selected_indices
        ]

        print(f"\n   📊 After MMR Re-ranking:")
        for idx, (doc, score) in enumerate(reranked_docs, 1):
            print(f"   {idx}. Score: {score:.4f} | {doc.page_content[:80]}...")

        return reranked_docs

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vectors.

        Args:
            vec1: First vector or matrix
            vec2: Second vector or matrix

        Returns:
            Cosine similarity score(s)
        """
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

        # Compute dot product
        if vec1_norm.ndim == 1:
            return np.dot(vec2_norm, vec1_norm)
        else:
            return np.dot(vec2_norm, vec1_norm.T)

    def create_qa_chain(self):
        """Create a QA chain with custom prompt template using LCEL.

        Returns:
            Configured LCEL chain and retrieval function
        """
        # Custom prompt template
        template = """You are an AI assistant specialized in explaining AI and machine learning concepts.
Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Provide a clear, concise answer and explain technical terms when necessary.

Context:
{context}

Question: {question}

Detailed Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        if not self.documents:
            raise ValueError("No documents loaded. Call add_documents() first.")

        # Custom retrieval function with MMR
        def retrieve_mmr(query: str) -> List[Document]:
            """Retrieve documents using MMR."""
            # Get initial results
            initial_results = self.retrieve_with_scores(query, k=5)
            # Re-rank with MMR
            reranked = self.mmr_rerank(query, initial_results, lambda_mult=0.5, k=3)
            return [doc for doc, _ in reranked]

        # Format documents helper
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create LCEL chain
        qa_chain = (
            {
                "context": lambda q: format_docs(retrieve_mmr(q)),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Return chain and retrieval function
        return qa_chain, retrieve_mmr

    def query_with_comparison(self, query: str, k: int = 5, mmr_k: int = 3):
        """Execute RAG query showing before/after re-ranking comparison.

        Args:
            query: The user's question
            k: Number of documents to retrieve initially
            mmr_k: Number of documents after MMR re-ranking
        """
        print(f"\n{'='*80}")
        print(f"🎯 ADVANCED RAG QUERY: {query}")
        print(f"{'='*80}\n")

        # Step 1: Initial retrieval
        initial_docs = self.retrieve_with_scores(query, k=k)

        # Step 2: Re-rank with MMR
        reranked_docs = self.mmr_rerank(query, initial_docs, lambda_mult=0.5, k=mmr_k)

        # Step 3: Generate answer using QA chain
        print(f"\n💡 GENERATION PHASE")
        print(f"   Generating answer with LangChain LCEL chain...")

        qa_chain, retrieve_func = self.create_qa_chain()

        # Get source documents
        source_docs = retrieve_func(query)

        # Generate answer
        answer = qa_chain.invoke(query)

        print(f"\n📋 FINAL ANSWER:")
        print(f"   {answer}")

        print(f"\n📚 SOURCE DOCUMENTS USED:")
        for idx, doc in enumerate(source_docs, 1):
            print(f"   {idx}. {doc.page_content[:100]}...")

        print(f"\n{'='*80}\n")

        return {"result": answer, "source_documents": source_docs}

    def query_simple(self, query: str) -> str:
        """Simple query using the QA chain with MMR retrieval.

        Args:
            query: The user's question

        Returns:
            Generated answer
        """
        print(f"\n{'='*80}")
        print(f"🎯 QUERY: {query}")
        print(f"{'='*80}\n")

        qa_chain, _ = self.create_qa_chain()
        answer = qa_chain.invoke(query)

        print(f"📋 ANSWER: {answer}\n")
        print(f"{'='*80}\n")

        return answer


def main():
    """Main function demonstrating advanced RAG usage."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           RAG Pattern - Advanced Implementation               ║
    ║                                                               ║
    ║  Demonstrating: MMR Re-ranking + LangChain QA Chain           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Initialize advanced RAG system
    rag = AdvancedRAG()

    # Enhanced documents about AI/ML topics with more detail
    documents = [
        "Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions. The three main types are supervised learning, unsupervised learning, and reinforcement learning.",

        "Deep Learning is a subset of machine learning based on artificial neural networks with representation learning. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, and natural language processing. These models can automatically learn hierarchical representations of data.",

        "Neural Networks are computing systems inspired by biological neural networks in animal brains. An artificial neural network consists of interconnected nodes (artificial neurons) organized in layers: input layer, hidden layers, and output layer. Each connection has a weight that adjusts during training through backpropagation algorithm.",

        "Natural Language Processing (NLP) is a branch of AI concerned with giving computers the ability to understand text and spoken words in the same way humans can. NLP combines computational linguistics with statistical, machine learning, and deep learning models. Applications include machine translation, sentiment analysis, and chatbots.",

        "Computer Vision is a field that enables computers to derive meaningful information from digital images, videos, and other visual inputs. It uses deep learning and convolutional neural networks (CNNs) to achieve human-level accuracy in tasks like object detection, facial recognition, and autonomous vehicle navigation.",

        "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for actions and aims to maximize cumulative reward. RL has achieved breakthrough results in game playing (AlphaGo), robotics, and autonomous systems.",

        "Transfer Learning is a machine learning technique where knowledge gained from training a model on one task is leveraged for a different but related task. This approach is especially valuable when labeled data is scarce. Pre-trained models like BERT for NLP and ResNet for computer vision are commonly used starting points.",

        "Generative AI refers to algorithms that can generate new content including text, images, music, and code. These systems use deep learning models like GANs (Generative Adversarial Networks), VAEs (Variational Autoencoders), and transformers. Notable examples include GPT for text generation and DALL-E for image synthesis.",

        "Supervised Learning is a machine learning paradigm where the algorithm learns from labeled training data. The model learns a function that maps inputs to outputs by minimizing prediction error. Common algorithms include linear regression, logistic regression, support vector machines, and decision trees. Applications span classification and regression tasks.",

        "Unsupervised Learning involves training models on unlabeled data to discover hidden patterns or structures. The algorithm learns without explicit feedback or correct answers. Key techniques include clustering (K-means, hierarchical), dimensionality reduction (PCA, t-SNE), and association rule learning. It's used for customer segmentation and anomaly detection.",

        "Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing grid-like data such as images. CNNs use convolutional layers with filters to automatically learn spatial hierarchies of features. They've revolutionized computer vision tasks including image classification, object detection, and semantic segmentation.",

        "Recurrent Neural Networks (RNNs) are neural networks designed to work with sequential data by maintaining internal memory of previous inputs. Variants like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) solve the vanishing gradient problem. RNNs are widely used in time series prediction, speech recognition, and natural language processing.",

        "Transformers are a neural network architecture that relies on self-attention mechanisms to process sequential data in parallel. Introduced in 2017, transformers have become the foundation of modern NLP with models like BERT, GPT, and T5. They excel at capturing long-range dependencies and have also been adapted for computer vision tasks.",

        "Attention Mechanisms allow neural networks to focus on specific parts of the input when producing output. Self-attention computes relationships between all positions in a sequence, enabling the model to weigh the importance of different input elements. Attention is the key innovation behind transformer models and has improved performance across many AI tasks.",

        "Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task by training it further on task-specific data. This technique is central to transfer learning and allows leveraging large pre-trained models like GPT or BERT for specialized applications with limited training data, saving time and computational resources.",
    ]

    # Add metadata for each document
    metadata = [{"topic": "Machine Learning", "complexity": "beginner"}] + \
               [{"topic": "Deep Learning", "complexity": "intermediate"}] + \
               [{"topic": "Neural Networks", "complexity": "intermediate"}] + \
               [{"topic": "NLP", "complexity": "intermediate"}] + \
               [{"topic": "Computer Vision", "complexity": "intermediate"}] + \
               [{"topic": "Reinforcement Learning", "complexity": "advanced"}] + \
               [{"topic": "Transfer Learning", "complexity": "intermediate"}] + \
               [{"topic": "Generative AI", "complexity": "intermediate"}] + \
               [{"topic": "Supervised Learning", "complexity": "beginner"}] + \
               [{"topic": "Unsupervised Learning", "complexity": "beginner"}] + \
               [{"topic": "CNNs", "complexity": "advanced"}] + \
               [{"topic": "RNNs", "complexity": "advanced"}] + \
               [{"topic": "Transformers", "complexity": "advanced"}] + \
               [{"topic": "Attention", "complexity": "advanced"}] + \
               [{"topic": "Fine-tuning", "complexity": "intermediate"}]

    # Add documents to the knowledge base
    rag.add_documents(documents, metadata)

    # Example 1: Query with comparison showing re-ranking
    print("\n" + "="*80)
    print("EXAMPLE 1: Detailed comparison with re-ranking")
    print("="*80)

    rag.query_with_comparison(
        "What are transformers and how do they use attention mechanisms?",
        k=5,
        mmr_k=3
    )

    # Example 2: Simple queries
    print("\n" + "="*80)
    print("EXAMPLE 2 & 3: Simple queries with MMR retrieval")
    print("="*80)

    queries = [
        "Explain the difference between supervised and unsupervised learning.",
        "What is transfer learning and why is it useful?",
    ]

    for query in queries:
        try:
            rag.query_simple(query)
        except Exception as e:
            print(f"❌ Error processing query: {e}\n")

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 Advanced RAG Complete!                        ║
    ║                                                               ║
    ║  The advanced RAG system demonstrated:                        ║
    ║  • MMR (Maximal Marginal Relevance) for diverse retrieval    ║
    ║  • Re-ranking to balance relevance and diversity             ║
    ║  • Before/after comparison of retrieval results              ║
    ║  • LangChain QA chain with custom prompts                    ║
    ║  • Source document tracking and attribution                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
