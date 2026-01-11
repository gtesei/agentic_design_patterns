"""
Parallel Chain Processing Example

This module demonstrates parallel execution of multiple LangChain tasks
(summarization, question generation, and key term extraction) followed by
synthesis of results into a comprehensive output.
"""

import asyncio
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7

# Initialize LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# ============================================================================
# CHAIN DEFINITIONS
# ============================================================================

def create_summarize_chain() -> Runnable:
    """Create a chain that generates a concise summary of a topic."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following topic concisely in 2-3 sentences:"),
        ("user", "{topic}")
    ])
    return prompt | llm | StrOutputParser()


def create_questions_chain() -> Runnable:
    """Create a chain that generates interesting questions about a topic."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate three thought-provoking questions about the following topic. "
            "Format each question on a new line with a number."
        ),
        ("user", "{topic}")
    ])
    return prompt | llm | StrOutputParser()


def create_terms_chain() -> Runnable:
    """Create a chain that extracts key terms from a topic."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Identify 5-10 key terms or concepts related to the following topic. "
            "Return them as a comma-separated list."
        ),
        ("user", "{topic}")
    ])
    return prompt | llm | StrOutputParser()


def create_synthesis_chain() -> Runnable:
    """Create a chain that synthesizes parallel results into a comprehensive output."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are synthesizing information about a topic. Based on the following:

Summary: {summary}

Related Questions:
{questions}

Key Terms: {key_terms}

Create a comprehensive, well-structured response that:
1. Integrates the summary with the key terms naturally
2. Addresses or contextualizes the questions raised
3. Provides additional insights or connections
4. Maintains clarity and coherence"""
        ),
        ("user", "Original topic: {topic}")
    ])
    return prompt | llm | StrOutputParser()


# ============================================================================
# PARALLEL CHAIN CONSTRUCTION
# ============================================================================

def build_parallel_chain() -> Runnable:
    """
    Build a complete parallel processing chain.
    
    Returns:
        A Runnable that executes summarization, question generation, and
        term extraction in parallel, then synthesizes the results.
    """
    # Create independent chains
    summarize_chain = create_summarize_chain()
    questions_chain = create_questions_chain()
    terms_chain = create_terms_chain()
    
    # Define parallel execution block
    map_chain = RunnableParallel({
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),
    })
    
    # Create synthesis chain
    synthesis_chain = create_synthesis_chain()
    
    # Combine into full pipeline
    full_chain = map_chain | synthesis_chain | StrOutputParser()
    
    return full_chain


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

async def process_topic_async(topic: str, chain: Optional[Runnable] = None) -> Dict[str, Any]:
    """
    Asynchronously process a topic through the parallel chain.
    
    Args:
        topic: The input topic to be processed
        chain: Optional pre-built chain (creates new one if None)
        
    Returns:
        Dictionary containing the final response and metadata
        
    Raises:
        Exception: If chain execution fails
    """
    if chain is None:
        chain = build_parallel_chain()
    
    print(f"\n{'=' * 70}")
    print(f"Processing Topic: '{topic}'")
    print(f"{'=' * 70}\n")
    
    try:
        # Execute the chain
        response = await chain.ainvoke(topic)
        
        result = {
            "topic": topic,
            "response": response,
            "status": "success"
        }
        
        print("\n--- Final Synthesized Response ---")
        print(response)
        print(f"\n{'=' * 70}\n")
        
        return result
        
    except Exception as e:
        error_msg = f"Chain execution failed: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}\n")
        
        return {
            "topic": topic,
            "response": None,
            "status": "error",
            "error": error_msg
        }


def process_topic_sync(topic: str) -> Dict[str, Any]:
    """
    Synchronously process a topic (wrapper for async function).
    
    Args:
        topic: The input topic to be processed
        
    Returns:
        Dictionary containing the final response and metadata
    """
    return asyncio.run(process_topic_async(topic))


async def process_multiple_topics(topics: list[str]) -> list[Dict[str, Any]]:
    """
    Process multiple topics in parallel.
    
    Args:
        topics: List of topics to process
        
    Returns:
        List of result dictionaries for each topic
    """
    chain = build_parallel_chain()
    tasks = [process_topic_async(topic, chain) for topic in topics]
    return await asyncio.gather(*tasks)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with example usage."""
    
    # Single topic example
    print("\n🚀 Running Single Topic Example")
    single_topic = "The history of space exploration"
    result = process_topic_sync(single_topic)
    
    # Multiple topics example (commented out by default)
    # print("\n🚀 Running Multiple Topics Example")
    # multiple_topics = [
    #     "The history of space exploration",
    #     "Quantum computing fundamentals",
    #     "Climate change mitigation strategies"
    # ]
    # results = asyncio.run(process_multiple_topics(multiple_topics))
    # 
    # # Print summary
    # successful = sum(1 for r in results if r["status"] == "success")
    # print(f"\n✅ Successfully processed {successful}/{len(results)} topics")


if __name__ == "__main__":
    main()