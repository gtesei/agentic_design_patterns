import os
import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4")

# --- Step 1: Producer Agent - Generate Initial Code ---
producer_prompt = ChatPromptTemplate.from_template(
    """You are an expert Python developer. Write a Python function that calculates the factorial of a number.

Requirements:
- Function name: factorial
- Input: positive integer n
- Output: factorial of n
- Include docstring

Write clean, working code."""
)

producer_chain = producer_prompt | llm | StrOutputParser()

# --- Step 2: Critic Agent - Evaluate the Code ---
critic_prompt = ChatPromptTemplate.from_template(
    """You are a senior code reviewer. Analyze this Python code and provide specific, actionable feedback.

Code to review:
{draft}

Evaluate based on:
1. **Correctness**: Does it handle all cases correctly?
2. **Edge cases**: Missing validation, error handling
3. **Performance**: Any efficiency concerns?
4. **Code quality**: Readability, documentation, style
5. **Best practices**: Type hints, robust error messages

Provide structured critique with specific suggestions for improvement."""
)

critic_chain = critic_prompt | llm | StrOutputParser()

# --- Step 3: Refinement Agent - Improve Based on Feedback ---
refine_prompt = ChatPromptTemplate.from_template(
    """You are an expert Python developer. Improve the following code based on the critique provided.

Original Code:
{draft}

Critique:
{critique}

Generate an improved version that addresses all feedback. Maintain working functionality while incorporating suggested improvements."""
)

refine_chain = refine_prompt | llm | StrOutputParser()

# --- Build the Reflection Chain ---
# Single reflection step (one critique and refinement cycle)
reflection_chain = (
    {"draft": producer_chain}
    | RunnablePassthrough.assign(
        critique=lambda x: critic_chain.invoke({"draft": x["draft"]})
    )
    | refine_chain
)

# --- Execute the Reflection Workflow ---
print("=" * 80)
print("REFLECTION PATTERN: Factorial Function Generation")
print("=" * 80)

# Generate initial draft
print("\n--- Step 1: Producer - Initial Code Generation ---")
initial_draft = producer_chain.invoke({})
print(initial_draft)

# Get critique
print("\n--- Step 2: Critic - Code Review ---")
critique = critic_chain.invoke({"draft": initial_draft})
print(critique)

# Generate refined version
print("\n--- Step 3: Refinement - Improved Code ---")
refined_code = refine_chain.invoke({
    "draft": initial_draft,
    "critique": critique
})
print(refined_code)

# Alternative: Run full chain in one call
print("\n" + "=" * 80)
print("SINGLE-CHAIN EXECUTION (Automated Reflection)")
print("=" * 80)
final_result = reflection_chain.invoke({})
print(final_result)
