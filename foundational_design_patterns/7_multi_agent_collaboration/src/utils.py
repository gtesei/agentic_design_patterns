# --- Standard library ---
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# --- Third-party ---
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
import wikipedia


# =============================================================================
# Environment + HTTP session
# =============================================================================

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_DEFAULT_USER_AGENT = "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"
_DEFAULT_TIMEOUT_S = 60


def _llm_error(msg: str) -> List[Dict[str, str]]:
    """Return a consistent LLM-friendly error format."""
    return [{"error": msg}]


def _make_session(user_agent: str = _DEFAULT_USER_AGENT) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


_SESSION = _make_session()


# =============================================================================
# arXiv tool
# =============================================================================

def arxiv_search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv for research papers matching the given query.

    Returns a list of dicts with keys:
      - title, authors, published, url, summary, link_pdf
    """
    url = f"{_ARXIV_API_URL}?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        resp = _SESSION.get(url, timeout=_DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return _llm_error(str(e))

    try:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = ((entry.find("atom:title", ns).text or "") if entry.find("atom:title", ns) is not None else "").strip()

            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                authors.append((name_el.text or "") if name_el is not None else "")

            published_el = entry.find("atom:published", ns)
            published = ((published_el.text or "") if published_el is not None else "")[:10]

            id_el = entry.find("atom:id", ns)
            url_abstract = ((id_el.text or "") if id_el is not None else "").strip()

            summary_el = entry.find("atom:summary", ns)
            summary = ((summary_el.text or "") if summary_el is not None else "").strip()

            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break

            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "url": url_abstract,
                    "summary": summary,
                    "link_pdf": link_pdf,
                }
            )

        return results
    except Exception as e:
        return _llm_error(f"Parsing failed: {e}")


arxiv_tool_def: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches for research papers on arXiv by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords for research papers."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# =============================================================================
# Tavily tool (refactored: dependency injection + lazy client)
# =============================================================================

@dataclass(frozen=True)
class TavilyConfig:
    api_key_env: str = "TAVILY_API_KEY"
    base_url_env: str = "DLAI_TAVILY_BASE_URL"  # optional
    default_max_results: int = 5


_TAVILY_CLIENT: Optional[TavilyClient] = None
_TAVILY_CONFIG = TavilyConfig()


def _get_env(name: str) -> Optional[str]:
    val = os.getenv(name)
    return val if val and val.strip() else None


def _get_tavily_client(config: TavilyConfig = _TAVILY_CONFIG) -> TavilyClient:
    """
    Create TavilyClient once and reuse it.
    Uses env vars by default:
      - TAVILY_API_KEY (required)
      - DLAI_TAVILY_BASE_URL (optional)
    """
    global _TAVILY_CLIENT

    if _TAVILY_CLIENT is not None:
        return _TAVILY_CLIENT

    api_key = _get_env(config.api_key_env)
    if not api_key:
        raise ValueError(f"{config.api_key_env} not found in environment variables.")

    api_base_url = _get_env(config.base_url_env)  # optional
    _TAVILY_CLIENT = TavilyClient(api_key=api_key, api_base_url=api_base_url)
    return _TAVILY_CLIENT


def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> List[Dict[str, Any]]:
    """
    Perform a search using the Tavily API.

    Returns list of dicts:
      - title, content, url
    and optionally image_url entries.
    """
    try:
        client = _get_tavily_client()
        response = client.search(query=query, max_results=max_results, include_images=include_images)

        results: List[Dict[str, Any]] = [
            {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
            for r in response.get("results", [])
        ]

        if include_images:
            results.extend({"image_url": img_url} for img_url in response.get("images", []))

        return results
    except Exception as e:
        return _llm_error(str(e))


tavily_tool_def: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords for retrieving information from the web."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}


# =============================================================================
# Wikipedia tool (refactored: tighter error handling)
# =============================================================================

def wikipedia_search_tool(query: str, sentences: int = 5) -> List[Dict[str, Any]]:
    """Search Wikipedia and return a short summary + URL."""
    try:
        titles = wikipedia.search(query)
        if not titles:
            return _llm_error(f"No Wikipedia results for query: {query}")

        page_title = titles[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)

        return [{"title": page.title, "summary": summary, "url": page.url}]
    except wikipedia.DisambiguationError as e:
        # Provide a helpful hint without failing hard
        options = e.options[:5]
        return _llm_error(f"Disambiguation for '{query}'. Try one of: {options}")
    except wikipedia.PageError:
        return _llm_error(f"Wikipedia page not found for query: {query}")
    except Exception as e:
        return _llm_error(str(e))


wikipedia_tool_def: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords for the Wikipedia article."},
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# =============================================================================
# Tool mapping (refactored: typed mapping)
# =============================================================================

ToolFn = Callable[..., List[Dict[str, Any]]]

tool_mapping: Dict[str, ToolFn] = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}


# =============================================================================
# tests_agents.py (split into its own module ideally; kept as-is but cleaned)
# =============================================================================

from dlai_grader.grading import test_case, print_feedback
from types import FunctionType

_TOPIC = "The ensemble Kalman filter for time series forecasting"
_TASK = "Draft a concise summary (150-250 words) explaining the core idea and typical applications."
_TASK_EDIT = "Reflect on the draft and suggest improvements in structure, clarity, and citations."


def test_planner_agent(learner_func):
    def g():
        function_name = "planner_agent"
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            out = learner_func(_TOPIC)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__}: {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        t = test_case()
        if not isinstance(out, list):
            t.failed = True
            t.msg = f"{function_name} must return a list[str]"
            t.want = list
            t.got = type(out)
            return [t]
        cases.append(t)

        t = test_case()
        elem_types_ok = all(isinstance(s, str) for s in out)
        if not elem_types_ok or len(out) < 3:
            t.failed = True
            t.msg = "plan should be a list of >=3 string steps"
            t.want = "list[str] with length >= 3"
            t.got = {"length": len(out), "bad_types": [type(s) for s in out if not isinstance(s, str)]}
        cases.append(t)

        t = test_case()
        last = (out[-1] if out else "").lower()
        if not any(k in last for k in ["markdown", "md"]):
            t.failed = True
            t.msg = "final step should mention generating a Markdown document"
            t.want = "mention of 'Markdown' or 'md'"
            t.got = last
        cases.append(t)

        return cases

    print_feedback(g())


def test_research_agent(learner_func):
    def g():
        function_name = "research_agent"
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            out_text = learner_func("Find 3 key references and summarize them briefly.")
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__} (return_messages=False): {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        t = test_case()
        if not isinstance(out_text, str):
            t.failed = True
            t.msg = f"{function_name} must return a str when return_messages=False"
            t.want = str
            t.got = type(out_text)
            return [t]
        cases.append(t)

        t = test_case()
        if len(out_text.strip()) <= 50:
            t.failed = True
            t.msg = "output should be non-trivial (length > 50) for research summary"
            t.want = "> 50 chars"
            t.got = len(out_text.strip())
        cases.append(t)

        try:
            out = learner_func("Briefly summarize two seminal papers in one paragraph.", return_messages=True)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__} (return_messages=True): {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        t = test_case()
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], str) and isinstance(out[1], list)):
            t.failed = True
            t.msg = f"{function_name} must return (str, messages_list) when return_messages=True"
            t.want = "(str, list)"
            t.got = type(out)
            return [t]
        cases.append(t)

        return cases

    print_feedback(g())


def test_writer_agent(learner_func):
    def g():
        function_name = "writer_agent"
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            out = learner_func(_TASK)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__}: {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        t = test_case()
        if not isinstance(out, str):
            t.failed = True
            t.msg = f"{function_name} must return a str"
            t.want = str
            t.got = type(out)
            return [t]
        cases.append(t)

        t = test_case()
        if len(out.strip()) <= 50:
            t.failed = True
            t.msg = "draft should be non-trivial (length > 50)"
            t.want = "> 50 chars"
            t.got = len(out.strip())
        cases.append(t)

        return cases

    print_feedback(g())


def test_editor_agent(learner_func):
    def g():
        function_name = "editor_agent"
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            out = learner_func(_TASK_EDIT)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__}: {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        t = test_case()
        if not isinstance(out, str):
            t.failed = True
            t.msg = f"{function_name} must return a str"
            t.want = str
            t.got = type(out)
            return [t]
        cases.append(t)

        t = test_case()
        if len(out.strip()) <= 50:
            t.failed = True
            t.msg = "editor output should be non-trivial (length > 50)"
            t.want = "> 50 chars"
            t.got = len(out.strip())
        cases.append(t)

        return cases

    print_feedback(g())
