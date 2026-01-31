"""
Human-in-the-Loop (HITL) with LangGraph Integration

This module demonstrates HITL using LangGraph's state management and
interrupt capabilities for human checkpoints.
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from datetime import datetime

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*len(title)}{Colors.ENDC}")


def print_state(state_name: str):
    """Print current state."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}[STATE: {state_name}]{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}{Colors.BOLD}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {text}{Colors.ENDC}")


# Define the state structure
class WorkflowState(TypedDict):
    """State for the HITL workflow."""
    task: str
    generated_content: str
    human_feedback: str
    approval_status: str  # pending, approved, rejected, needs_revision
    revision_count: int
    conversation_history: list[dict]
    final_output: str


def generate_content_node(state: WorkflowState) -> WorkflowState:
    """
    Generate content based on the task.
    This is an automated node that doesn't require human input.
    """
    print_state("GENERATE_CONTENT")
    print(f"Generating content for: {state['task']}")

    # Get LLM from state or create new one
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Build prompt based on whether this is a revision
    if state.get('human_feedback'):
        prompt = f"""Task: {state['task']}

Previous attempt: {state.get('generated_content', '')}

Human feedback: {state['human_feedback']}

Please revise the content based on the feedback above."""
    else:
        prompt = f"""Task: {state['task']}

Please generate high-quality content for this task. Be creative and professional."""

    # Generate content
    response = llm.invoke(prompt)
    generated = response.content

    print_success("Content generated successfully")
    print(f"\n{Colors.BOLD}Generated Content:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")
    print(generated)
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")

    # Update state
    state['generated_content'] = generated
    state['approval_status'] = 'pending'
    state['conversation_history'].append({
        'timestamp': datetime.now().isoformat(),
        'node': 'generate_content',
        'content': generated,
        'revision': state.get('revision_count', 0)
    })

    return state


def review_node(state: WorkflowState) -> WorkflowState:
    """
    Human review checkpoint - this node interrupts for human input.
    """
    print_state("HUMAN_REVIEW")
    print_section("🔍 HUMAN REVIEW CHECKPOINT")

    print(f"\n{Colors.BOLD}Task:{Colors.ENDC} {state['task']}")
    print(f"\n{Colors.BOLD}Generated Content:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")
    print(state['generated_content'])
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Revision Count:{Colors.ENDC} {state.get('revision_count', 0)}")

    print(f"\n{Colors.YELLOW}{Colors.BOLD}Review Options:{Colors.ENDC}")
    print(f"  {Colors.GREEN}[A]{Colors.ENDC} Approve - Accept the content")
    print(f"  {Colors.RED}[R]{Colors.ENDC} Reject - Discard and end workflow")
    print(f"  {Colors.YELLOW}[M]{Colors.ENDC} Modify - Request changes")

    # Get human decision
    while True:
        decision = input(f"\n{Colors.BOLD}Your decision [A/R/M]:{Colors.ENDC} ").strip().upper()

        if decision in ['A', 'APPROVE']:
            state['approval_status'] = 'approved'
            state['human_feedback'] = ''
            print_success("Content approved!")
            break
        elif decision in ['R', 'REJECT']:
            state['approval_status'] = 'rejected'
            reason = input(f"{Colors.BOLD}Rejection reason:{Colors.ENDC} ").strip()
            state['human_feedback'] = reason or "No reason provided"
            print_error("Content rejected")
            break
        elif decision in ['M', 'MODIFY']:
            state['approval_status'] = 'needs_revision'
            feedback = input(f"{Colors.BOLD}What changes would you like?{Colors.ENDC} ").strip()
            if feedback:
                state['human_feedback'] = feedback
                state['revision_count'] = state.get('revision_count', 0) + 1
                print_warning(f"Requesting revision (Attempt {state['revision_count']})")
                break
            else:
                print_error("Please provide feedback for modifications")
        else:
            print_error("Invalid option. Please choose A, R, or M")

    # Log the decision
    state['conversation_history'].append({
        'timestamp': datetime.now().isoformat(),
        'node': 'human_review',
        'decision': state['approval_status'],
        'feedback': state.get('human_feedback', '')
    })

    return state


def revision_check_node(state: WorkflowState) -> WorkflowState:
    """
    Check if revision limit has been reached.
    """
    print_state("REVISION_CHECK")

    max_revisions = 3
    current_count = state.get('revision_count', 0)

    if current_count >= max_revisions:
        print_warning(f"Maximum revision limit ({max_revisions}) reached")
        state['approval_status'] = 'rejected'
        state['human_feedback'] = f"Maximum revisions ({max_revisions}) exceeded"

    return state


def finalize_node(state: WorkflowState) -> WorkflowState:
    """
    Finalize the workflow with approved content.
    """
    print_state("FINALIZE")

    if state['approval_status'] == 'approved':
        print_section("✅ FINALIZING APPROVED CONTENT")
        state['final_output'] = state['generated_content']
        print_success("Content has been finalized and is ready for use")
    else:
        print_section("❌ WORKFLOW TERMINATED")
        state['final_output'] = ''
        print_error("No content was finalized")

    # Log finalization
    state['conversation_history'].append({
        'timestamp': datetime.now().isoformat(),
        'node': 'finalize',
        'status': state['approval_status'],
        'final_output': state.get('final_output', '')
    })

    return state


def routing_logic(state: WorkflowState) -> Literal["finalize", "revision_check", "generate_content"]:
    """
    Determine the next node based on approval status.
    """
    status = state.get('approval_status', 'pending')

    if status == 'approved':
        return "finalize"
    elif status == 'rejected':
        return "finalize"
    elif status == 'needs_revision':
        return "revision_check"
    else:
        return "generate_content"


def create_hitl_workflow() -> StateGraph:
    """
    Create a LangGraph workflow with HITL checkpoints.
    """
    # Create the graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("generate_content", generate_content_node)
    workflow.add_node("review", review_node)
    workflow.add_node("revision_check", revision_check_node)
    workflow.add_node("finalize", finalize_node)

    # Define edges
    workflow.set_entry_point("generate_content")

    # After generation, go to review
    workflow.add_edge("generate_content", "review")

    # After review, use conditional routing
    workflow.add_conditional_edges(
        "review",
        routing_logic,
        {
            "finalize": "finalize",
            "revision_check": "revision_check",
            "generate_content": "generate_content"  # This won't be used from review
        }
    )

    # After revision check, either regenerate or finalize
    workflow.add_conditional_edges(
        "revision_check",
        lambda state: "generate_content" if state['approval_status'] == 'needs_revision' else "finalize",
        {
            "generate_content": "generate_content",
            "finalize": "finalize"
        }
    )

    # Finalize is the end
    workflow.add_edge("finalize", END)

    return workflow


def display_conversation_history(history: list):
    """Display the conversation history."""
    print_section("📜 CONVERSATION HISTORY")

    for i, entry in enumerate(history, 1):
        print(f"\n{Colors.BOLD}Entry {i}:{Colors.ENDC}")
        print(f"  Timestamp: {entry['timestamp']}")
        print(f"  Node: {entry['node']}")

        if 'content' in entry:
            print(f"  Content: [Generated content - revision {entry.get('revision', 0)}]")
        if 'decision' in entry:
            print(f"  Decision: {entry['decision']}")
        if 'feedback' in entry and entry['feedback']:
            print(f"  Feedback: {entry['feedback']}")
        if 'status' in entry:
            print(f"  Status: {entry['status']}")


def run_workflow_example(task: str):
    """Run a complete workflow example."""
    print_header(f"LANGGRAPH HITL WORKFLOW")
    print(f"\n{Colors.BOLD}Task:{Colors.ENDC} {task}\n")

    # Create the workflow
    workflow = create_hitl_workflow()

    # Compile with memory saver for state persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # Initialize state
    initial_state = {
        'task': task,
        'generated_content': '',
        'human_feedback': '',
        'approval_status': 'pending',
        'revision_count': 0,
        'conversation_history': [],
        'final_output': ''
    }

    # Configuration for thread
    config = {"configurable": {"thread_id": "hitl_demo_1"}}

    # Run the workflow
    print_section("🚀 STARTING WORKFLOW")

    try:
        # Execute the workflow
        final_state = None
        for state in app.stream(initial_state, config):
            final_state = state

        # Display results
        if final_state:
            # Get the last state value (could be from any node)
            last_node = list(final_state.keys())[0]
            state_data = final_state[last_node]

            print_section("📊 WORKFLOW RESULTS")

            print(f"\n{Colors.BOLD}Final Status:{Colors.ENDC} {state_data['approval_status']}")
            print(f"{Colors.BOLD}Total Revisions:{Colors.ENDC} {state_data.get('revision_count', 0)}")

            if state_data.get('final_output'):
                print(f"\n{Colors.BOLD}Final Output:{Colors.ENDC}")
                print(f"{Colors.GREEN}{'-'*80}{Colors.ENDC}")
                print(state_data['final_output'])
                print(f"{Colors.GREEN}{'-'*80}{Colors.ENDC}")
            else:
                print_error("\nNo final output produced")

            # Display conversation history
            display_conversation_history(state_data['conversation_history'])

            # Summary statistics
            print_section("📈 SUMMARY STATISTICS")
            total_events = len(state_data['conversation_history'])
            generation_events = sum(1 for e in state_data['conversation_history'] if e['node'] == 'generate_content')
            review_events = sum(1 for e in state_data['conversation_history'] if e['node'] == 'human_review')

            print(f"Total events: {total_events}")
            print(f"Content generations: {generation_events}")
            print(f"Human reviews: {review_events}")
            print(f"Final status: {state_data['approval_status']}")

    except Exception as e:
        print_error(f"Workflow error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to demonstrate LangGraph HITL."""
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment variables.")
        print(f"Please ensure .env file exists at: {env_path}")
        sys.exit(1)

    print_header("WELCOME TO LANGGRAPH HITL DEMO")
    print("\nThis demo shows human-in-the-loop workflow using LangGraph.")
    print("The workflow includes state management, conditional routing, and human checkpoints.\n")

    # Example tasks
    tasks = [
        "Write a professional email announcing a new product feature",
        "Create a social media post about company culture",
        "Draft a customer support response for a billing inquiry",
        "Generate a blog post introduction about AI ethics"
    ]

    print_section("Available Tasks")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    print(f"\n{Colors.BOLD}Or enter a custom task{Colors.ENDC}")

    # Get task selection
    while True:
        choice = input(f"\n{Colors.BOLD}Select task (1-{len(tasks)}) or enter custom task:{Colors.ENDC} ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(tasks):
            task = tasks[int(choice) - 1]
            break
        elif choice:
            task = choice
            break
        else:
            print_error("Please enter a valid choice")

    # Run the workflow
    run_workflow_example(task)

    print_section("✨ DEMO COMPLETE")
    print("\nKey LangGraph Features Demonstrated:")
    print("  • State management across workflow nodes")
    print("  • Human-in-the-loop checkpoints")
    print("  • Conditional routing based on decisions")
    print("  • Conversation history tracking")
    print("  • Iterative refinement with revision limits")


if __name__ == "__main__":
    main()
