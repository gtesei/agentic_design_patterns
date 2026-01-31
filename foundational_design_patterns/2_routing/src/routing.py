import os
import sys
from typing import Dict, Any

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize LLM
llm = ChatOpenAI(temperature=0)


# ============================================================================
# HANDLER FUNCTIONS
# ============================================================================

def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return (
        f"Booking Handler processed request: '{request}'.\n"
        f"Result: Simulated booking action."
    )


def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return (
        f"Info Handler processed request: '{request}'.\n"
        f"Result: Simulated information retrieval."
    )


def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return (
        f"Coordinator could not delegate request: '{request}'.\n"
        f"Please clarify your request."
    )


# ============================================================================
# COORDINATOR ROUTER CHAIN
# ============================================================================

coordinator_router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Analyze the user's request and determine which specialist handler should process it.

        - If the request is related to booking flights or hotels, output 'booker'.
        - For general information questions, output 'info'.
        - If the request is unclear or doesn't fit either category, output 'unclear'.

        ONLY output one word: 'booker', 'info', or 'unclear'."""
    ),
    ("user", "{request}")
])

coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()


# ============================================================================
# DELEGATION LOGIC
# ============================================================================

def extract_request(x: Dict[str, Any]) -> str:
    """Safely extract the request string from nested dict structure."""
    return x.get('request', {}).get('request', '')


# Define branch handlers
branches = {
    "booker": RunnablePassthrough.assign(
        output=lambda x: booking_handler(extract_request(x))
    ),
    "info": RunnablePassthrough.assign(
        output=lambda x: info_handler(extract_request(x))
    ),
    "unclear": RunnablePassthrough.assign(
        output=lambda x: unclear_handler(extract_request(x))
    ),
}

# Create the RunnableBranch for routing
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip().lower() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip().lower() == 'info', branches["info"]),
    branches["unclear"]  # Default branch
)

# Build the complete coordinator agent pipeline
coordinator_agent = (
    {
        "decision": coordinator_router_chain,
        "request": RunnablePassthrough()
    }
    | delegation_branch
    | (lambda x: x['output'])  # Extract final output
)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_example(description: str, request: str) -> None:
    """Run a single example and print results."""
    print(f"\n{'=' * 70}")
    print(f"--- {description} ---")
    print(f"{'=' * 70}")
    result = coordinator_agent.invoke({"request": request})
    print(f"\nFinal Result: {result}")


if __name__ == "__main__":
    # Test cases
    run_example(
        "Running with a booking request",
        "Book me a flight to London."
    )
    
    run_example(
        "Running with an info request",
        "What is the capital of Italy?"
    )
    
    run_example(
        "Running with an unclear request",
        "Tell me about quantum physics."
    )