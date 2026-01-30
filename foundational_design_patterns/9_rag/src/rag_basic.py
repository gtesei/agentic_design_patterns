"""
RAG Pattern: Basic Implementation
This example demonstrates a basic Retrieval-Augmented Generation (RAG) pattern using:
- Qdrant in-memory vector database
- sentence-transformers for embeddings
- OpenAI for generation
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))


class BasicRAG:
    """Basic RAG implementation with Qdrant and sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG system.

        Args:
            model_name: Name of the sentence-transformer model to use for embeddings
        """
        print("\n🔧 Initializing Basic RAG System...")

        # Initialize embedding model
        print(f"   Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize Qdrant in-memory
        print("   Setting up Qdrant vector database (in-memory)")
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = "ai_ml_docs"

        # Initialize OpenAI for generation
        print("   Connecting to OpenAI for generation")
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")

        print("✅ RAG system initialized!\n")

    def create_collection(self):
        """Create a Qdrant collection for storing document embeddings."""
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )
        print(f"📦 Created collection: {self.collection_name}")

    def add_documents(self, documents: List[str]):
        """Add documents to the vector database.

        Args:
            documents: List of text documents to add
        """
        print(f"\n📝 Adding {len(documents)} documents to the knowledge base...")

        # Generate embeddings for all documents
        embeddings = self.embedding_model.encode(documents, show_progress_bar=False)

        # Create points for Qdrant
        points = [
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"text": doc}
            )
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]

        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"✅ Successfully indexed {len(documents)} documents\n")

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with their scores
        """
        print(f"🔍 RETRIEVAL PHASE")
        print(f"   Query: '{query}'")
        print(f"   Retrieving top {top_k} relevant documents...")

        # Embed the query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        # Format results
        retrieved_docs = []
        for idx, result in enumerate(search_results, 1):
            if result.payload is None:
                continue
            doc = {
                "text": result.payload["text"],
                "score": result.score
            }
            retrieved_docs.append(doc)
            print(f"\n   📄 Document {idx} (Score: {result.score:.4f}):")
            print(f"      {result.payload['text'][:100]}...")

        return retrieved_docs

    def augment(self, query: str, retrieved_docs: List[dict]) -> str:
        """Augment the query with retrieved context.

        Args:
            query: The original query
            retrieved_docs: List of retrieved documents

        Returns:
            Augmented prompt with context
        """
        print(f"\n🔗 AUGMENTATION PHASE")
        print(f"   Building prompt with {len(retrieved_docs)} context documents...")

        # Build context from retrieved documents
        context = "\n\n".join([
            f"Context {idx}: {doc['text']}"
            for idx, doc in enumerate(retrieved_docs, 1)
        ])

        # Create augmented prompt
        augmented_prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

        print(f"   ✅ Prompt augmented with {len(context)} characters of context")
        return augmented_prompt

    def generate(self, augmented_prompt: str) -> str:
        """Generate a response using the LLM.

        Args:
            augmented_prompt: The prompt with retrieved context

        Returns:
            Generated response
        """
        print(f"\n💡 GENERATION PHASE")
        print(f"   Generating response with OpenAI...")

        response = self.llm.invoke(augmented_prompt)
        return response.content

    def query(self, query: str, top_k: int = 3) -> str:
        """Execute the full RAG pipeline: retrieve, augment, generate.

        Args:
            query: The user's question
            top_k: Number of documents to retrieve

        Returns:
            Generated answer
        """
        print(f"\n{'='*80}")
        print(f"🎯 RAG QUERY: {query}")
        print(f"{'='*80}\n")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)

        # Step 2: Augment query with context
        augmented_prompt = self.augment(query, retrieved_docs)

        # Step 3: Generate response
        answer = self.generate(augmented_prompt)

        print(f"\n📋 FINAL ANSWER:")
        print(f"   {answer}")
        print(f"\n{'='*80}\n")

        return answer


def main():
    """Main function demonstrating basic RAG usage."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              RAG Pattern - Basic Implementation               ║
    ║                                                               ║
    ║  Demonstrating: Retrieval → Augmentation → Generation        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Initialize RAG system
    rag = BasicRAG()

    # Create collection
    rag.create_collection()

    # Sample documents about AI/ML topics
    documents = [
        "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",

        "Deep Learning is a subset of machine learning based on artificial neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain, allowing it to learn from large amounts of data. Deep learning drives many AI applications like image recognition and natural language processing.",

        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines including computer science and linguistics to bridge the gap between human communication and computer understanding.",

        "Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images and deep learning models, machines can accurately identify and classify objects, and react to what they see.",

        "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions and seeing the results. The agent receives rewards or penalties for its actions and learns to maximize rewards over time. It's used in robotics, game playing, and autonomous vehicles.",

        "Transfer Learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It's a popular approach in deep learning where pre-trained models are used as the foundation for computer vision and NLP tasks.",

        "Generative AI refers to artificial intelligence systems that can generate new content, including text, images, music, and code. These systems use neural networks trained on large datasets to learn patterns and create original outputs. Examples include GPT for text and DALL-E for images.",

        "Neural Networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers. Information passes through the network, with each connection having weights that are adjusted during training to make better predictions.",

        "Supervised Learning is a machine learning approach where the model is trained on labeled data. The algorithm learns from input-output pairs and can then make predictions on new, unseen data. Common applications include spam detection, image classification, and price prediction.",

        "Unsupervised Learning is a type of machine learning that finds hidden patterns or structures in unlabeled data. The system tries to learn without human supervision by discovering patterns on its own. Common techniques include clustering and dimensionality reduction.",
    ]

    # Add documents to the knowledge base
    rag.add_documents(documents)

    # Example queries
    queries = [
        "What is the difference between machine learning and deep learning?",
        "How does reinforcement learning work?",
        "Explain what generative AI is and give examples.",
    ]

    # Run queries
    for query in queries:
        try:
            rag.query(query, top_k=3)
        except Exception as e:
            print(f"❌ Error processing query: {e}\n")

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Basic RAG Complete!                        ║
    ║                                                               ║
    ║  The RAG system demonstrated:                                 ║
    ║  • Embedding documents into a vector database                 ║
    ║  • Retrieving relevant documents based on semantic similarity ║
    ║  • Augmenting prompts with retrieved context                  ║
    ║  • Generating informed answers using LLM                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
