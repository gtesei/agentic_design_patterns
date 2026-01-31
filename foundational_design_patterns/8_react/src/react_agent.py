"""
ReAct Pattern: Basic Implementation
This example demonstrates the ReAct (Reasoning and Acting) pattern using LangGraph's
prebuilt create_react_agent function. The agent alternates between reasoning about
what to do and taking actions with tools.
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4")

# --- Define Tools for the ReAct Agent ---

@tool
def search(query: str) -> str:
    """Search for information on the web or in a knowledge base.

    Use this when you need to find facts, current information, or data
    that you don't already know.

    Args:
        query: A clear, specific search query

    Returns:
        Relevant information and facts related to the query
    """
    # In production, integrate with real search APIs (Google, Bing, etc.)
    # This is a mock implementation with simulated knowledge

    knowledge_base = {
        "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. "
                 "It emphasizes code readability and supports multiple programming paradigms including procedural, "
                 "object-oriented, and functional programming.",

        "tokyo population": "Tokyo is the capital of Japan with a population of approximately 14 million people in "
                          "the city proper (as of 2024), and about 37.4 million in the Greater Tokyo Area, making "
                          "it the most populous metropolitan area in the world.",

        "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was designed by "
                       "Gustave Eiffel and completed in 1889. It stands 330 meters (1,083 feet) tall and was the "
                       "world's tallest structure until 1930.",

        "photosynthesis": "Photosynthesis is the process by which plants and other organisms convert light energy "
                         "into chemical energy. The process uses carbon dioxide and water to produce glucose and "
                         "oxygen, with the chemical equation: 6CO2 + 6H2O + light → C6H12O6 + 6O2.",

        "marie curie": "Marie Curie (1867-1934) was a Polish-French physicist and chemist who conducted pioneering "
                      "research on radioactivity. She was the first woman to win a Nobel Prize, the first person to "
                      "win Nobel Prizes in two scientific fields (Physics in 1903 and Chemistry in 1911).",

        "great wall china": "The Great Wall of China is a series of fortifications built across northern China. "
                           "Construction began in the 7th century BC, with major construction during the Ming Dynasty "
                           "(1368-1644). The wall stretches approximately 21,196 kilometers (13,171 miles).",

        "black holes": "Black holes are regions of spacetime with gravitational fields so strong that nothing, not "
                      "even light, can escape. They form when massive stars collapse at the end of their life cycles. "
                      "The boundary of a black hole is called the event horizon.",

        "shakespeare": "William Shakespeare (1564-1616) was an English playwright, poet, and actor, widely regarded "
                      "as the greatest writer in the English language. He wrote approximately 39 plays, 154 sonnets, "
                      "and several poems. Famous works include Hamlet, Romeo and Juliet, and Macbeth.",
    }

    query_lower = query.lower()

    # Search for matching topics
    for topic, info in knowledge_base.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            return f"Search results for '{query}':\n\n{info}"

    # Generic fallback
    return f"Search results for '{query}':\n\nNo specific information found in the knowledge base. " \
           f"In a production system, this would query real search engines or APIs."


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations.

    Use this when you need to compute numerical results, solve equations,
    or perform any mathematical operations.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "15 * 7", "100 / 4")

    Returns:
        The numerical result of the calculation
    """
    try:
        # In production, use a safer evaluation method (e.g., ast.literal_eval with parsing)
        # This is simplified for demonstration
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_word_count(text: str) -> str:
    """Count the number of words in a given text.

    Use this when you need to analyze text length or word frequency.

    Args:
        text: The text to analyze

    Returns:
        Word count and basic text statistics
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = text.count('.') + text.count('!') + text.count('?')

    return f"Text analysis:\n- Words: {word_count}\n- Characters: {char_count}\n- Sentences: ~{sentence_count}"


# --- Create the ReAct Agent ---

tools = [search, calculator, get_word_count]

agent = create_react_agent(llm, tools)


# --- Example Usage ---

def run_example(query: str):
    """Run a single example query through the ReAct agent"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    result = agent.invoke({"messages": [("user", query)]})

    # Display the full reasoning trace
    for message in result["messages"]:
        if hasattr(message, 'content') and message.content:
            print(f"{message.type.upper()}: {message.content}")

        # Show tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\nTOOL CALL: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         ReAct Pattern - Basic Implementation                  ║
    ║                                                               ║
    ║  The agent will Reason, Act, and Observe to answer queries    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Simple research question
    run_example(
        "What is photosynthesis and how many oxygen molecules does it produce per glucose molecule?"
    )

    # Example 2: Multi-step mathematical problem
    run_example(
        "If Tokyo has a population of 14 million and the Eiffel Tower is 330 meters tall, "
        "what is the sum of Tokyo's population plus the Eiffel Tower's height in meters?"
    )

    # Example 3: Research + Analysis
    run_example(
        "Tell me about Marie Curie and count how many words are in her description."
    )

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Examples Complete!                         ║
    ║                                                               ║
    ║  The ReAct agent demonstrated:                                ║
    ║  • Reasoning about what information was needed                ║
    ║  • Using tools to gather facts and perform calculations       ║
    ║  • Observing results and adapting its approach                ║
    ║  • Combining multiple tools to solve complex queries          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
