"""
Sequential Prompt Chaining Example

Demonstrates a two-step chain that:
1. Extracts technical specifications from natural language
2. Transforms extracted specs into structured JSON format

This pattern is useful for converting unstructured text into structured data
through multiple LLM processing steps.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables (requires OPENAI_API_KEY in .env file)
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Initialize Language Model
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================

def create_extraction_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for extracting technical specifications from text.
    
    Returns:
        ChatPromptTemplate configured for specification extraction
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a technical specification extraction expert. "
            "Extract all hardware specifications from the provided text. "
            "Focus on CPU, memory (RAM), and storage details. "
            "Be precise and include units of measurement."
        ),
        (
            "user",
            "Extract the technical specifications from the following text:\n\n{text_input}"
        )
    ])


def create_transformation_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for transforming specifications into JSON format.
    
    Returns:
        ChatPromptTemplate configured for JSON transformation
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a data formatting expert. Convert technical specifications "
            "into valid JSON format. Use these exact keys: 'cpu', 'memory', 'storage'. "
            "Include units in the values (e.g., 'GHz', 'GB', 'TB'). "
            "Return ONLY the JSON object, no additional text or markdown formatting."
        ),
        (
            "user",
            "Transform the following specifications into a JSON object:\n\n{specifications}"
        )
    ])


# ============================================================================
# CHAIN CONSTRUCTION
# ============================================================================

def build_extraction_chain() -> Any:
    """
    Build the specification extraction chain.
    
    Returns:
        A Runnable chain that extracts specifications from text
    """
    prompt = create_extraction_prompt()
    return prompt | llm | StrOutputParser()


def build_full_chain() -> Any:
    """
    Build the complete two-step chain: extraction → transformation.
    
    The chain:
    1. Extracts technical specifications from input text
    2. Transforms extracted specs into structured JSON
    
    Returns:
        A Runnable chain that processes text into JSON format
    """
    extraction_chain = build_extraction_chain()
    transformation_prompt = create_transformation_prompt()
    
    # Chain the extraction output into the transformation step
    full_chain = (
        {"specifications": extraction_chain}
        | transformation_prompt
        | llm
        | StrOutputParser()
    )
    
    return full_chain


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def process_text_to_json(text: str, verbose: bool = True) -> Dict[str, str]:
    """
    Process input text through the extraction and transformation chain.
    
    Args:
        text: Input text containing technical specifications
        verbose: Whether to print intermediate steps
        
    Returns:
        Dictionary containing the processing result and metadata
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print("Processing Input Text")
        print(f"{'=' * 70}")
        print(f"\nInput:\n{text}\n")
    
    try:
        # Build and execute the chain
        chain = build_full_chain()
        result = chain.invoke({"text_input": text})
        
        if verbose:
            print("--- Extracted & Transformed JSON ---")
            print(result)
            print(f"\n{'=' * 70}\n")
        
        return {
            "status": "success",
            "input": text,
            "output": result
        }
        
    except Exception as e:
        error_msg = f"Chain execution failed: {str(e)}"
        if verbose:
            print(f"\n❌ ERROR: {error_msg}\n")
        
        return {
            "status": "error",
            "input": text,
            "output": None,
            "error": error_msg
        }


def process_multiple_texts(texts: list[str]) -> list[Dict[str, str]]:
    """
    Process multiple input texts in sequence.
    
    Args:
        texts: List of input texts to process
        
    Returns:
        List of result dictionaries for each input
    """
    chain = build_full_chain()
    results = []
    
    for i, text in enumerate(texts, 1):
        print(f"\n--- Processing Text {i}/{len(texts)} ---")
        try:
            output = chain.invoke({"text_input": text})
            results.append({
                "status": "success",
                "input": text,
                "output": output
            })
        except Exception as e:
            results.append({
                "status": "error",
                "input": text,
                "output": None,
                "error": str(e)
            })
    
    return results


# ============================================================================
# EXAMPLE DATA
# ============================================================================

EXAMPLE_TEXTS = [
    "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD.",
    "This workstation includes an Intel Core i9-13900K running at 5.8 GHz, 64GB DDR5 memory, and 2TB PCIe 4.0 storage.",
    "Budget desktop: AMD Ryzen 5 5600G at 3.9GHz, 8 gigabytes RAM, 512GB solid state drive."
]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with example usage."""
    
    print("\n🚀 Sequential Prompt Chaining Example")
    print("=" * 70)
    print("Extracting specs → Transforming to JSON")
    print("=" * 70)
    
    # Single example
    print("\n📝 Single Text Example:")
    result = process_text_to_json(EXAMPLE_TEXTS[0])
    
    # Multiple examples (commented out by default)
    # print("\n📚 Multiple Texts Example:")
    # results = process_multiple_texts(EXAMPLE_TEXTS)
    # 
    # # Print summary
    # successful = sum(1 for r in results if r["status"] == "success")
    # print(f"\n✅ Successfully processed {successful}/{len(results)} texts")
    # 
    # # Show all results
    # for i, result in enumerate(results, 1):
    #     print(f"\nResult {i}:")
    #     if result["status"] == "success":
    #         print(result["output"])
    #     else:
    #         print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()