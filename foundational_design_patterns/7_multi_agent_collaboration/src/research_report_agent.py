"""
Agentic Workflows

Keeps the same roles and orchestration:
- planner_agent: returns a Python list[str] plan
- research_agent: tool-using agent (arXiv, Tavily, Wikipedia) via LangChain tools
- writer_agent: drafts structured content
- editor_agent: critiques / suggests improvements
- executor_agent: routes plan steps to the right agent, accumulates history

Assumptions:
- utils.py provides:
    arxiv_search_tool, tavily_search_tool, wikipedia_search_tool
    (optionally) tool_mapping and tool defs
- OPENAI_API_KEY is set in env (or .env loaded elsewhere)
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import utils
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "../", ".env"))


# =============================================================================
# LLM factory
# =============================================================================

def _make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


# =============================================================================
# Tools (LangChain wrappers around utils.py)
# =============================================================================

@tool
def arxiv_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search arXiv for papers by query string."""
    return utils.arxiv_search_tool(query=query, max_results=max_results)


@tool
def tavily_tool(query: str, max_results: int = 5, include_images: bool = False) -> List[Dict[str, Any]]:
    """Web search using the Tavily API."""
    return utils.tavily_search_tool(query=query, max_results=max_results, include_images=include_images)


@tool
def wikipedia_tool(query: str, sentences: int = 5) -> List[Dict[str, Any]]:
    """Fetch a short Wikipedia summary and URL for a query."""
    return utils.wikipedia_search_tool(query=query, sentences=sentences)


RESEARCH_TOOLS = [arxiv_tool, tavily_tool, wikipedia_tool]
TOOLS_MAP: Dict[str, Any] = {t.name: t for t in RESEARCH_TOOLS}


# =============================================================================
# Shared prompts
# =============================================================================

PLANNER_SYSTEM = """You are a planning agent responsible for organizing a research workflow with multiple intelligent agents.

🧠 Available agents:
- A research agent who can search the web, Wikipedia, and arXiv.
- A writer agent who can draft research summaries.
- An editor agent who can reflect and revise drafts.

🎯 Your job is to write a clear, step-by-step research plan as a valid Python list,
where each step is a string. Steps should be atomic and executable.

🚫 DO NOT include irrelevant tasks like "create CSV", "set up a repo", "install packages", etc.
✅ DO include real research-related tasks (e.g., search, summarize, draft, revise).
✅ DO NOT include explanation text — return ONLY the Python list.
✅ The final step should be to generate a Markdown document containing the complete research report.
"""

WRITER_SYSTEM = """You are a professional writing assistant specialized in producing clear, well-structured,
and rigorous academic and technical content.

Your role is to draft, expand, summarize, or refine text with high clarity and precision.
- Organize content logically using headings and coherent flow
- Avoid filler language and repetition
- Preserve technical correctness
- Do not invent facts; use only provided info
Produce polished, publication-quality text.
"""

EDITOR_SYSTEM = """You are a professional editor specialized in reviewing, critiquing, and improving
academic and technical writing.

Your role is NOT to rewrite from scratch, but to:
- Identify weaknesses in clarity, structure, logic, and flow
- Suggest concrete improvements and improved wording when helpful
- Preserve intent; do not introduce new facts
Provide actionable feedback and (when appropriate) revised versions of problematic sections.
"""

RESEARCH_SYSTEM = """You are an expert research assistant designed to execute complex research tasks using external tools.

You have access to:
- arxiv_tool: academic papers / technical research
- tavily_tool: current web context / recent info
- wikipedia_tool: definitions / background

Instructions:
- Use arxiv_tool for scientific/technical claims.
- Use tavily_tool for current events, stats, recent sources.
- Use wikipedia_tool for definitions and background.
- Cite sources by including the tool name and returned URLs for key claims.
- If tools are insufficient, say what's missing; never fabricate.
Return a structured research result.
"""


# =============================================================================
# Agent implementations
# =============================================================================

def planner_agent(topic: str, model: str = "gpt-4o-mini") -> List[str]:
    """
    Generates a plan as a Python list[str] (returned as text by the LLM).
    Handles Markdown code block formatting automatically.
    """
    llm = _make_llm(model=model, temperature=1.0)
    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f'Topic: "{topic}"\n\nReturn ONLY the Python list.'),
    ]
    resp = llm.invoke(messages)
    steps_str = (resp.content or "").strip()

    # --- FIX: Clean Markdown code blocks ---
    # This regex looks for text between ```python and ``` or just ``` and ```
    match = re.search(r"```(?:python)?\s*(.*?)```", steps_str, re.DOTALL)
    if match:
        cleaned_str = match.group(1).strip()
    else:
        cleaned_str = steps_str

    try:
        # Use the cleaned string instead of the raw LLM output
        steps = ast.literal_eval(cleaned_str)
    except Exception as e:
        raise ValueError(f"planner_agent returned non-literal plan: {e}\nRaw:\n{steps_str}") from e

    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError("planner_agent must return a Python list[str] (as text).")
    
    print("\n🚀 --- Research Execution Plan ---")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")
    print("----------------------------------\n")

    return steps


def _run_tool_loop(
    llm_with_tools: ChatOpenAI,
    messages: List[Any],
    max_turns: int = 6,
) -> Tuple[str, List[Any]]:
    """
    Generic tool-calling loop for LangChain tool calls.
    Returns (final_text, full_messages).
    """
    for _ in range(max_turns):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            return (resp.content or ""), messages

        # Execute tools in order and append ToolMessage(s)
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            tool_obj = TOOLS_MAP.get(name)

            if tool_obj is None:
                tool_out = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    tool_out = tool_obj.invoke(args)
                except Exception as e:
                    tool_out = {"error": f"{type(e).__name__}: {e}"}

            messages.append(
                ToolMessage(
                    content=json.dumps(tool_out),
                    tool_call_id=call.get("id"),
                )
            )

    # If we hit max turns without a final response
    last = messages[-1]
    if isinstance(last, AIMessage) and last.content:
        return last.content, messages
    return "Max tool iterations reached. Please simplify the request.", messages


def research_agent(
    task: str,
    model: str = "gpt-4o",
    return_messages: bool = False,
) -> Any:
    """
    Executes a research task using LangChain tool calling.
    Returns str, or (str, messages) if return_messages=True.
    """
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    current_time = datetime.now().strftime("%Y-%m-%d")
    llm = _make_llm(model=model, temperature=0.2).bind_tools(RESEARCH_TOOLS)

    messages: List[Any] = [
        SystemMessage(content=f"{RESEARCH_SYSTEM}\n\nCurrent time: {current_time}"),
        HumanMessage(content=f"TASK:\n{task}\n\nBegin now."),
    ]

    content, msgs = _run_tool_loop(llm, messages, max_turns=6)

    print("✅ Output:\n", content)
    return (content, msgs) if return_messages else content


def writer_agent(task: str, model: str = "gpt-4o") -> str:
    """
    Drafts/expands/summarizes text (no tools).
    """
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")

    llm = _make_llm(model=model, temperature=1.0)
    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=task),
    ]
    resp = llm.invoke(messages)
    return resp.content or ""


def editor_agent(task: str, model: str = "gpt-4o") -> str:
    """
    Critiques / suggests improvements (no tools).
    """
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")

    llm = _make_llm(model=model, temperature=0.7)
    messages = [
        SystemMessage(content=EDITOR_SYSTEM),
        HumanMessage(content=task),
    ]
    resp = llm.invoke(messages)
    return resp.content or ""


# =============================================================================
# Executor (orchestrator)
# =============================================================================

AgentFn = Callable[..., Any]

agent_registry: Dict[str, AgentFn] = {
    "research_agent": research_agent,
    "editor_agent": editor_agent,
    "writer_agent": writer_agent,
}


def clean_json_block(raw: str) -> str:
    """Clean JSON that may be wrapped with Markdown backticks."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


@dataclass(frozen=True)
class AgentDecision:
    agent: str
    task: str


def _decide_agent(step: str, model: str = "gpt-4o") -> AgentDecision:
    """
    Uses an LLM to route a plan step to one of: research_agent, writer_agent, editor_agent.
    """
    llm = _make_llm(model=model, temperature=0.0)
    prompt = f"""
You are an execution manager for a multi-agent research team.

Given the following instruction, identify which agent should perform it and extract the clean task.

Return only a valid JSON object with two keys:
- "agent": one of ["research_agent", "editor_agent", "writer_agent"]
- "task": a string with the instruction that the agent should follow

Only respond with a valid JSON object. Do not include explanations or markdown formatting.

Instruction: "{step}"
""".strip()

    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = resp.content or ""
    info = json.loads(clean_json_block(raw))
    return AgentDecision(agent=info["agent"], task=info["task"])


def executor_agent(
    topic: str,
    model: str = "gpt-4o",
    limit_steps: bool = True,
    max_steps:int = 10,
) -> List[Tuple[str, str, str]]:
    """
    Executes a planner-produced workflow by routing each step to the right agent,
    building context from history.
    """
    plan_steps = planner_agent(topic)
    if limit_steps:
        plan_steps = plan_steps[: min(len(plan_steps), max_steps)]

    history: List[Tuple[str, str, str]] = []

    print("==================================")
    print("🎯 Executor Agent")
    print("==================================")

    for i, step in enumerate(plan_steps, start=1):
        decision = _decide_agent(step, model=model)
        agent_name, task = decision.agent, decision.task

        context = "\n".join(
            [f"Step {j+1} executed by {a}:\n{r}" for j, (_, a, r) in enumerate(history)]
        )

        enriched_task = f"""
You are {agent_name}.

Here is the context of what has been done so far:
{context}

Your next task is:
{task}
""".strip()

        print(f"\n🛠️ Step {i}: agent=`{agent_name}` task={task}")

        if agent_name not in agent_registry:
            output = f"⚠️ Unknown agent: {agent_name}"
        else:
            output = agent_registry[agent_name](enriched_task)

        history.append((step, agent_name, output))
        print(f"✅ Output:\n{output}")

    return history


# =============================================================================
# Example run (optional)
# =============================================================================

if __name__ == "__main__":
    topic = "The ensemble Kalman filter for time series forecasting"
    print()
    print("\n=== User topic ===\n")
    print(topic)

    h = executor_agent(topic, limit_steps=True)
    print("\n================ FINAL ================\n")
    print(h[-1][-1])
