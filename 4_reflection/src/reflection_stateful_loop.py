"""
Reflection Pattern with Stateful Loops: Blog Post Writer
This example demonstrates using LangGraph for iterative reflection to write high-quality blog posts.
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
import operator
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0.7, model="gpt-5.2")  # Higher temperature for creativity

# --- Define State Schema ---
class BlogPostState(TypedDict):
    """State for the blog post writing workflow"""
    topic: str  # Blog post topic
    target_audience: str  # Target audience
    tone: str  # Desired tone (professional, casual, technical, etc.)
    word_count_target: int  # Target word count
    draft: str  # Current draft
    critique: str  # Feedback from critic
    iteration: Annotated[int, operator.add]  # Iteration counter
    issues_found: list[str]  # List of issues identified
    quality_score: float  # Overall quality score (0-10)
    is_approved: bool  # Whether post meets quality standards
    max_iterations: int  # Maximum allowed iterations
    improvement_history: Annotated[list[dict], operator.add]  # Track improvements

# --- Pydantic Models for Structured Critique ---
class BlogCritique(BaseModel):
    """Structured critique of blog post"""
    overall_quality: float = Field(ge=0, le=10, description="Overall quality score 0-10")
    
    # Flow & Structure
    flow_score: float = Field(ge=0, le=10, description="Logical flow and structure")
    flow_issues: list[str] = Field(default_factory=list, description="Specific flow problems")
    
    # Tone & Voice
    tone_score: float = Field(ge=0, le=10, description="Tone appropriateness")
    tone_issues: list[str] = Field(default_factory=list, description="Tone mismatches")
    
    # Clarity & Readability
    clarity_score: float = Field(ge=0, le=10, description="Clarity and readability")
    clarity_issues: list[str] = Field(default_factory=list, description="Unclear sections")
    
    # Content Quality
    content_score: float = Field(ge=0, le=10, description="Content depth and accuracy")
    content_issues: list[str] = Field(default_factory=list, description="Content gaps")
    
    # Engagement
    engagement_score: float = Field(ge=0, le=10, description="Reader engagement potential")
    engagement_issues: list[str] = Field(default_factory=list, description="Engagement problems")
    
    # Specific recommendations
    strengths: list[str] = Field(default_factory=list, description="What works well")
    priority_improvements: list[str] = Field(default_factory=list, description="Top 3-5 improvements needed")
    suggested_additions: list[str] = Field(default_factory=list, description="Content to add")
    suggested_removals: list[str] = Field(default_factory=list, description="Content to remove/reduce")
    
    is_publication_ready: bool = Field(description="Ready to publish as-is")

# --- Prompts ---
WRITER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert blog writer specializing in {tone} content for {target_audience}.

{context}

Topic: {topic}
Target Word Count: {word_count_target}
Tone: {tone}
Audience: {target_audience}

Requirements:
- Write engaging, well-structured content
- Use clear, accessible language
- Include relevant examples and insights
- Create compelling introduction and conclusion
- Use subheadings for better readability
- Maintain consistent voice throughout

{additional_instructions}

Write the blog post:"""
)

CRITIC_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert content editor and writing coach. Evaluate this blog post comprehensively.

Topic: {topic}
Target Audience: {target_audience}
Desired Tone: {tone}
Target Word Count: {word_count_target}

Blog Post Draft:
{draft}

Evaluate on these dimensions:
1. **Flow & Structure**: Is it logically organized? Do paragraphs transition smoothly?
2. **Tone & Voice**: Does it match the {tone} tone? Is voice consistent?
3. **Clarity & Readability**: Is it easy to understand? Any jargon or unclear passages?
4. **Content Quality**: Is it informative, accurate, and valuable?
5. **Engagement**: Will it hook and retain readers?

Provide detailed, actionable feedback. Be specific about what to fix and how."""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert blog writer. Rewrite this blog post based on the critique.

Original Draft:
{draft}

Detailed Critique:
{critique}

Priority Improvements:
{priority_improvements}

Instructions:
- Address ALL identified issues
- Maintain what works well
- Implement suggested improvements
- Keep the same topic and core message
- Match the {tone} tone for {target_audience}
- Aim for {word_count_target} words

Write the improved version:"""
)

# --- Node Functions ---

def writer_node(state: BlogPostState) -> BlogPostState:
    """Producer: Generate or rewrite blog post"""
    print(f"\n{'='*80}")
    print(f"ITERATION {state['iteration']}: WRITER")
    print(f"{'='*80}")
    
    # Build context based on iteration
    if state['iteration'] == 0:
        context = "This is your first draft. Write freely and comprehensively."
        additional_instructions = "Focus on getting ideas down. We'll refine later."
    else:
        context = f"""This is iteration {state['iteration']}.

Previous quality score: {state['quality_score']:.1f}/10

Previous critique summary:
{state['critique'][:500]}...

Issues to address:
{chr(10).join(f'- {issue}' for issue in state['issues_found'][:5])}"""
        
        additional_instructions = f"""CRITICAL: Address the specific feedback above.
Focus on the priority improvements identified by the editor.
Build on what's working; fix what's not."""
    
    # Choose prompt based on iteration
    if state['iteration'] == 0:
        # Initial draft
        draft = (
            WRITER_PROMPT 
            | llm 
            | StrOutputParser()
        ).invoke({
            "topic": state['topic'],
            "target_audience": state['target_audience'],
            "tone": state['tone'],
            "word_count_target": state['word_count_target'],
            "context": context,
            "additional_instructions": additional_instructions
        })
    else:
        # Rewrite based on critique
        critique_data = state.get('critique', '')
        priority_improvements = '\n'.join(state.get('issues_found', [])[:5])
        
        draft = (
            REWRITE_PROMPT 
            | llm 
            | StrOutputParser()
        ).invoke({
            "draft": state['draft'],
            "critique": critique_data,
            "priority_improvements": priority_improvements,
            "tone": state['tone'],
            "target_audience": state['target_audience'],
            "word_count_target": state['word_count_target']
        })
    
    word_count = len(draft.split())
    print(f"\nDraft generated: {word_count} words")
    print(f"\nFirst 300 characters:")
    print(draft[:300] + "...")
    
    return {
        #**state,
        "draft": draft,
        "iteration": 1
    }

def critic_node(state: BlogPostState) -> BlogPostState:
    """Critic: Evaluate blog post with structured feedback"""
    print(f"\n{'='*80}")
    print(f"ITERATION {state['iteration']}: CRITIC")
    print(f"{'='*80}")
    
    # Use structured output for reliable critique
    llm_structured = ChatOpenAI(temperature=0, model="gpt-5.2").with_structured_output(BlogCritique)
    
    critique_result: BlogCritique = llm_structured.invoke(
        CRITIC_PROMPT.format(
            topic=state['topic'],
            target_audience=state['target_audience'],
            tone=state['tone'],
            word_count_target=state['word_count_target'],
            draft=state['draft']
        )
    )
    
    # Format critique as readable text
    critique_text = f"""
OVERALL QUALITY: {critique_result.overall_quality}/10

DETAILED SCORES:
- Flow & Structure: {critique_result.flow_score}/10
- Tone & Voice: {critique_result.tone_score}/10
- Clarity & Readability: {critique_result.clarity_score}/10
- Content Quality: {critique_result.content_score}/10
- Engagement: {critique_result.engagement_score}/10

STRENGTHS:
{chr(10).join(f'✓ {s}' for s in critique_result.strengths)}

PRIORITY IMPROVEMENTS:
{chr(10).join(f'! {p}' for p in critique_result.priority_improvements)}

FLOW ISSUES:
{chr(10).join(f'- {i}' for i in critique_result.flow_issues)}

TONE ISSUES:
{chr(10).join(f'- {i}' for i in critique_result.tone_issues)}

CLARITY ISSUES:
{chr(10).join(f'- {i}' for i in critique_result.clarity_issues)}

CONTENT ISSUES:
{chr(10).join(f'- {i}' for i in critique_result.content_issues)}

ENGAGEMENT ISSUES:
{chr(10).join(f'- {i}' for i in critique_result.engagement_issues)}

SUGGESTED ADDITIONS:
{chr(10).join(f'+ {a}' for a in critique_result.suggested_additions)}

SUGGESTED REMOVALS:
{chr(10).join(f'- {r}' for r in critique_result.suggested_removals)}

PUBLICATION READY: {'YES' if critique_result.is_publication_ready else 'NO'}
"""
    
    print(f"\nCritique Summary:")
    print(f"  Overall Quality: {critique_result.overall_quality}/10")
    print(f"  Flow: {critique_result.flow_score}/10")
    print(f"  Tone: {critique_result.tone_score}/10")
    print(f"  Clarity: {critique_result.clarity_score}/10")
    print(f"  Content: {critique_result.content_score}/10")
    print(f"  Engagement: {critique_result.engagement_score}/10")
    print(f"  Publication Ready: {critique_result.is_publication_ready}")
    
    # Collect all issues
    all_issues = (
        critique_result.flow_issues +
        critique_result.tone_issues +
        critique_result.clarity_issues +
        critique_result.content_issues +
        critique_result.engagement_issues
    )
    
    # Track improvement
    improvement_entry = {
        "iteration": state['iteration'], 
        "quality_score": critique_result.overall_quality,
        "issues_count": len(all_issues),
        "publication_ready": critique_result.is_publication_ready
    }
    
    return {
        "critique": critique_text,
        "quality_score": critique_result.overall_quality,
        "issues_found": all_issues,
        "is_approved": critique_result.is_publication_ready,
        "improvement_history": [improvement_entry]
    }

def decision_node(state: BlogPostState) -> BlogPostState:
    """Decide if blog post is ready or needs more iterations"""
    print(f"\n{'='*80}")
    print(f"DECISION NODE")
    print(f"{'='*80}")
    
    print(f"\nQuality Score: {state['quality_score']}/10")
    print(f"Approved: {state['is_approved']}")
    print(f"Issues Found: {len(state['issues_found'])}")
    print(f"Current Iteration: {state['iteration']}/{state['max_iterations']}")
    
    # Check if quality threshold met
    quality_threshold = 8.0
    meets_threshold = state['quality_score'] >= quality_threshold
    
    print(f"Meets Quality Threshold ({quality_threshold}): {meets_threshold}")
    
    return state

def should_continue(state: BlogPostState) -> Literal["writer", "end"]:
    """Routing function: Continue iteration or end"""
    
    # End if approved and quality is high
    if state['is_approved'] and state['quality_score'] >= 8.0:
        print("\n✓ Blog post approved for publication!")
        return "end"
    
    # End if max iterations reached
    if state['iteration'] >= state['max_iterations']:
        if state['quality_score'] >= 7.0:
            print(f"\n⚠ Max iterations reached. Quality acceptable ({state['quality_score']}/10)")
        else:
            print(f"\n✗ Max iterations reached. Quality below target ({state['quality_score']}/10)")
        return "end"
    
    # Continue if quality improving and issues remain
    print(f"\n→ Continuing to iteration {state['iteration'] + 1}")
    print(f"   Focus: {state['issues_found'][:3]}")
    return "writer"

# --- Build the Graph ---

def create_blog_writer_graph():
    """Create the LangGraph workflow for blog post writing with reflection"""
    
    # Initialize graph
    workflow = StateGraph(BlogPostState)
    
    # Add nodes
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("decision", decision_node)
    
    # Define edges
    workflow.set_entry_point("writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "decision")
    
    # Conditional edge: continue or end
    workflow.add_conditional_edges(
        "decision",
        should_continue,
        {
            "writer": "writer",
            "end": END
        }
    )
    
    return workflow.compile()

# --- Main Execution ---

def write_blog_post(
    topic: str,
    target_audience: str = "general readers",
    tone: str = "professional yet accessible",
    word_count_target: int = 800,
    max_iterations: int = 3
):
    """
    Write a blog post using the reflection pattern with stateful loops.
    
    Args:
        topic: Blog post topic
        target_audience: Who the post is for
        tone: Desired writing tone
        word_count_target: Target word count
        max_iterations: Maximum number of reflection cycles
    """
    
    print("\n" + "="*80)
    print("BLOG POST WRITER WITH REFLECTION PATTERN")
    print("="*80)
    
    print(f"\nTopic: {topic}")
    print(f"Audience: {target_audience}")
    print(f"Tone: {tone}")
    print(f"Target Length: {word_count_target} words")
    print(f"Max Iterations: {max_iterations}")
    
    # Initialize state
    initial_state: BlogPostState = {
        "topic": topic,
        "target_audience": target_audience,
        "tone": tone,
        "word_count_target": word_count_target,
        "draft": "",
        "critique": "",
        "iteration": 0,
        "issues_found": [],
        "quality_score": 0.0,
        "is_approved": False,
        "max_iterations": max_iterations,
        "improvement_history": []
    }
    
    # Create and run graph
    graph = create_blog_writer_graph()
    
    # Execute workflow
    final_state = graph.invoke(initial_state)
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\nTotal Iterations: {final_state['iteration']}")
    print(f"Final Quality Score: {final_state['quality_score']}/10")
    print(f"Status: {'✓ APPROVED' if final_state['is_approved'] else '⚠ NEEDS WORK'}")
    print(f"Word Count: {len(final_state['draft'].split())} words")
    
    # Show improvement trajectory
    print(f"\nImprovement Trajectory:")
    for entry in final_state['improvement_history']:
        status = "✓" if entry['publication_ready'] else "✗"
        print(f"  Iteration {entry['iteration']}: {entry['quality_score']:.1f}/10 {status} ({entry['issues_count']} issues)")
    
    if final_state['issues_found']:
        print(f"\nRemaining Issues ({len(final_state['issues_found'])}):")
        for i, issue in enumerate(final_state['issues_found'][:5], 1):
            print(f"  {i}. {issue}")
    
    print(f"\n{'='*80}")
    print("FINAL BLOG POST")
    print("="*80)
    print(final_state['draft'])
    
    print(f"\n{'='*80}")
    print("FINAL CRITIQUE")
    print("="*80)
    print(final_state['critique'])
    
    return final_state

# --- Example Topics ---

EXAMPLE_TOPICS = {
    "technical": {
        "topic": "Understanding Agentic AI Frameworks: A Practical Guide",
        "target_audience": "software developers and ML engineers",  # CORRECT KEY
        "tone": "technical but accessible",
        "word_count_target": 1200  # ALSO FIX THIS
    },
    "business": {
        "topic": "Why Remote Work is Transforming Company Culture",
        "target_audience": "business leaders and HR professionals",  # CORRECT KEY
        "tone": "professional and analytical",
        "word_count_target": 1000  # ALSO FIX THIS
    },
    "casual": {
        "topic": "5 Simple Habits That Changed My Morning Routine",
        "target_audience": "lifestyle enthusiasts",  # CORRECT KEY
        "tone": "casual and conversational",
        "word_count_target": 800  # ALSO FIX THIS
    },
    "educational": {
        "topic": "How Machine Learning is Revolutionizing Healthcare",
        "target_audience": "healthcare professionals with limited tech background",  # CORRECT KEY
        "tone": "educational and encouraging",
        "word_count_target": 1000  # ALSO FIX THIS
    }
}

# --- Visualization Helper ---

def visualize_improvement(state: BlogPostState):
    print("\n" + "="*80)
    print("QUALITY IMPROVEMENT VISUALIZATION")
    print("="*80)
    
    # Deduplicate by keeping only last entry per iteration
    seen_iterations = {}
    for entry in state['improvement_history']:
        iteration = entry['iteration']
        seen_iterations[iteration] = entry
    
    # Display in order
    for iteration in sorted(seen_iterations.keys()):
        entry = seen_iterations[iteration]
        score = entry['quality_score']
        ready = entry['publication_ready']
        
        bar_length = int(score)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        status = "✓" if ready else "✗"
        
        print(f"Iteration {iteration}: {bar} {score:.1f}/10 {status}")

# --- Run Examples ---

if __name__ == "__main__":
    
    # EXAMPLE 1 ------------------------------
    print("\n" + "#"*80)
    print("# EXAMPLE 1: Technical Blog Post")
    print("#"*80)
    
    result = write_blog_post(
        topic="Understanding Agentic AI Frameworks: A Practical Guide",
        target_audience="software developers and ML engineers",
        tone="technical but accessible",
        word_count_target=1200,
        max_iterations=3
    )
    
    visualize_improvement(result)
    
    # EXAMPLE 2 ------------------------------
    print("\n" + "#"*80)
    print("# EXAMPLE 2: Business Blog Post (Uncomment to run)")
    print("#"*80)

    result = write_blog_post(**EXAMPLE_TOPICS["business"], max_iterations=3)

    visualize_improvement(result)