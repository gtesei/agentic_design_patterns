"""
ReAct Pattern: Advanced Implementation with Explicit Reasoning Traces
This example demonstrates a custom ReAct implementation using LangGraph's StateGraph
with explicit reasoning traces, iteration tracking, and enhanced observability.
"""

import os
import sys
from typing import Annotated, TypedDict, Sequence

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4")


# --- Define State Schema ---

class ReActState(TypedDict):
    """State for the ReAct agent with explicit tracking"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    max_iterations: int


# --- Define Tools ---

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information on a topic.

    Use this when you need authoritative, factual information about
    historical figures, scientific concepts, places, or events.

    Args:
        query: Topic or question to search for

    Returns:
        Relevant Wikipedia-style information
    """
    # Mock Wikipedia knowledge base
    wiki_articles = {
        "albert einstein": """Albert Einstein (1879-1955) was a German-born theoretical physicist who developed
        the theory of relativity, one of the two pillars of modern physics. His work is known for its influence on
        the philosophy of science. He is best known to the general public for his mass–energy equivalence formula
        E = mc². He received the 1921 Nobel Prize in Physics for his services to theoretical physics, especially
        for his discovery of the law of the photoelectric effect.""",

        "great barrier reef": """The Great Barrier Reef is the world's largest coral reef system composed of over
        2,900 individual reefs and 900 islands stretching for over 2,300 kilometres (1,400 mi) over an area of
        approximately 344,400 square kilometres (133,000 sq mi). The reef is located in the Coral Sea, off the
        coast of Queensland, Australia. It was selected as a World Heritage Site in 1981.""",

        "french revolution": """The French Revolution was a period of far-reaching social and political upheaval
        in France that lasted from 1789 until 1799. It led to the fall of the monarchy, the rise of Napoleon
        Bonaparte, and significant social and political changes. Major events include the Storming of the Bastille
        (July 14, 1789), the Declaration of the Rights of Man and of the Citizen, and the Reign of Terror.""",

        "quantum mechanics": """Quantum mechanics is a fundamental theory in physics that provides a description
        of the physical properties of nature at the scale of atoms and subatomic particles. It differs from
        classical physics primarily at the quantum realm of atomic and subatomic length scales. Quantum mechanics
        provides a mathematical description of much of the dual particle-like and wave-like behavior and
        interactions of energy and matter.""",

        "amazon rainforest": """The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical
        rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin
        encompasses 7,000,000 km² (2,700,000 sq mi), of which 5,500,000 km² (2,100,000 sq mi) are covered by
        the rainforest. The majority of the forest is contained within Brazil, with 60% of the rainforest,
        followed by Peru with 13%, and Colombia with 10%.""",
    }

    query_lower = query.lower()
    for topic, article in wiki_articles.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            return f"Wikipedia article on '{topic.title()}':\n\n{article}"

    return f"No Wikipedia article found for '{query}'. Try a different search term."


@tool
def scientific_calculator(operation: str, a: float, b: float = None) -> str:
    """Perform scientific calculations.

    Use this for mathematical operations including basic arithmetic,
    exponents, square roots, and more.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide, power, sqrt, percent)
        a: First number
        b: Second number (not required for sqrt)

    Returns:
        Calculation result with explanation
    """
    try:
        if operation == "add":
            result = a + b
            return f"{a} + {b} = {result}"
        elif operation == "subtract":
            result = a - b
            return f"{a} - {b} = {result}"
        elif operation == "multiply":
            result = a * b
            return f"{a} × {b} = {result}"
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            result = a / b
            return f"{a} ÷ {b} = {result}"
        elif operation == "power":
            result = a ** b
            return f"{a}^{b} = {result}"
        elif operation == "sqrt":
            if a < 0:
                return "Error: Cannot take square root of negative number"
            result = a ** 0.5
            return f"√{a} = {result}"
        elif operation == "percent":
            result = (a / 100) * b
            return f"{a}% of {b} = {result}"
        else:
            return f"Unknown operation: {operation}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def text_analyzer(text: str, analysis_type: str = "full") -> str:
    """Analyze text for various metrics.

    Use this to extract statistics and insights from text.

    Args:
        text: The text to analyze
        analysis_type: Type of analysis (full, words, chars, sentences)

    Returns:
        Text analysis results
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))

    if analysis_type == "words":
        return f"Word count: {word_count}"
    elif analysis_type == "chars":
        return f"Character count: {char_count} (without spaces: {char_count_no_spaces})"
    elif analysis_type == "sentences":
        return f"Sentence count: {sentence_count}"
    else:  # full
        avg_word_length = char_count_no_spaces / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        return f"""Text Analysis:
        - Total words: {word_count}
        - Total characters: {char_count}
        - Characters (no spaces): {char_count_no_spaces}
        - Sentences: {sentence_count}
        - Average word length: {avg_word_length:.1f} characters
        - Average sentence length: {avg_sentence_length:.1f} words"""


# --- Create Tool Node ---

tools = [wikipedia_search, scientific_calculator, text_analyzer]
tool_node = ToolNode(tools)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# --- Define Enhanced System Prompt for Explicit Reasoning ---

REACT_SYSTEM_PROMPT = """You are a helpful research assistant using the ReAct (Reasoning and Acting) framework.

For each task, follow this explicit pattern:

1. **Thought**: Clearly state what you're thinking about the task
   - What do I know so far?
   - What information do I still need?
   - What should I do next?

2. **Action**: If you need more information, use the appropriate tool
   - Choose the right tool for the job
   - Provide clear parameters

3. **Observation**: After receiving tool results, state what you learned
   - What did this tell me?
   - Is it sufficient to answer the question?
   - Do I need more information?

4. **Repeat**: Continue the thought-action-observation cycle until you have enough information

5. **Answer**: When you have all necessary information, provide a complete, accurate final answer

Be systematic, thorough, and show your reasoning process clearly."""


# --- Define Graph Nodes ---

def agent_node(state: ReActState) -> ReActState:
    """Agent reasoning and tool selection node"""

    # Add system prompt if this is the first iteration
    messages = state["messages"]
    if state["iteration"] == 0:
        messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + list(messages)

    # Check iteration limit
    remaining = state["max_iterations"] - state["iteration"]
    if remaining <= 0:
        return {
            "messages": [AIMessage(content="Maximum iterations reached. Unable to complete task.")],
            "iteration": state["iteration"] + 1
        }

    # Add iteration context
    if state["iteration"] > 0:
        iteration_msg = SystemMessage(
            content=f"[Iteration {state['iteration']}/{state['max_iterations']}] "
                   f"Continue reasoning and decide if you need more information."
        )
        messages = list(messages) + [iteration_msg]

    # Invoke LLM
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }


def should_continue(state: ReActState) -> str:
    """Determine if agent should continue or end"""
    last_message = state["messages"][-1]

    # Check if iteration limit reached
    if state["iteration"] >= state["max_iterations"]:
        return "end"

    # Check if agent wants to use tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # No more tools needed - end
    return "end"


# --- Build the Graph ---

workflow = StateGraph(ReActState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Tools always loop back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()


# --- Enhanced Output Display ---

def display_react_trace(result: dict):
    """Display the ReAct reasoning trace in a readable format"""
    print("\n" + "="*100)
    print("REACT REASONING TRACE")
    print("="*100 + "\n")

    iteration = 0
    for i, message in enumerate(result["messages"]):
        # Skip system messages in display
        if isinstance(message, SystemMessage):
            continue

        if isinstance(message, HumanMessage):
            print(f"┌─ USER QUERY")
            print(f"│  {message.content}")
            print(f"└─" + "─"*80 + "\n")

        elif isinstance(message, AIMessage):
            if message.tool_calls:
                iteration += 1
                print(f"┌─ ITERATION {iteration}: THOUGHT + ACTION")
                print(f"│")
                if message.content:
                    print(f"│  💭 Thought: {message.content}")
                    print(f"│")
                print(f"│  🔧 Action: Using tools...")
                for tool_call in message.tool_calls:
                    print(f"│     - Tool: {tool_call['name']}")
                    print(f"│     - Args: {tool_call['args']}")
                print(f"└─" + "─"*80 + "\n")
            else:
                print(f"┌─ FINAL ANSWER")
                print(f"│")
                print(f"│  ✓ {message.content}")
                print(f"└─" + "─"*80 + "\n")

        elif isinstance(message, ToolMessage):
            print(f"┌─ OBSERVATION")
            print(f"│")
            print(f"│  📊 Result:")
            # Format multi-line tool results
            for line in message.content.split('\n'):
                print(f"│     {line}")
            print(f"└─" + "─"*80 + "\n")

    print("="*100)
    print(f"Total Iterations: {result.get('iteration', 0)}")
    print("="*100 + "\n")


# --- Example Usage ---

def run_advanced_example(query: str, max_iterations: int = 10):
    """Run an example with the advanced ReAct agent"""
    print(f"\n{'='*100}")
    print(f"QUERY: {query}")
    print(f"{'='*100}\n")

    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "iteration": 0,
        "max_iterations": max_iterations
    })

    display_react_trace(result)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║              ReAct Pattern - Advanced Implementation                           ║
    ║                                                                                ║
    ║  This version shows explicit Thought → Action → Observation traces             ║
    ║  with iteration tracking and enhanced observability                            ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Multi-step research with calculation
    run_advanced_example(
        "Find information about Albert Einstein, then calculate what 20% of his Nobel Prize year would be."
    )

    # Example 2: Research and text analysis
    run_advanced_example(
        "Look up information about the Amazon rainforest and analyze how many words are in the description."
    )

    # Example 3: Complex multi-tool problem
    run_advanced_example(
        "Search for the Great Barrier Reef, calculate the square root of its area in square kilometers, "
        "and tell me the result."
    )

    print("""
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                          Examples Complete!                                    ║
    ║                                                                                ║
    ║  The Advanced ReAct agent demonstrated:                                        ║
    ║  • Explicit reasoning traces at each step                                      ║
    ║  • Iteration tracking and limits                                               ║
    ║  • Clear Thought → Action → Observation cycles                                 ║
    ║  • Enhanced observability for debugging                                        ║
    ║  • Custom state management with LangGraph                                      ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """)
