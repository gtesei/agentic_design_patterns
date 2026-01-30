"""
Tree of Thoughts Pattern Implementation

This package implements the Tree of Thoughts reasoning pattern, which enables
LLMs to explore multiple reasoning paths simultaneously through systematic
branching, evaluation, and pruning.
"""

from .tot_basic import (
    ThoughtNode,
    generate_thoughts,
    evaluate_thought,
    tree_of_thoughts_bfs,
)

from .tot_advanced import (
    AdvancedThoughtNode,
    generate_creative_thoughts,
    evaluate_creative_thought,
    tree_of_thoughts_beam_search,
)

__all__ = [
    # Basic ToT
    'ThoughtNode',
    'generate_thoughts',
    'evaluate_thought',
    'tree_of_thoughts_bfs',
    # Advanced ToT
    'AdvancedThoughtNode',
    'generate_creative_thoughts',
    'evaluate_creative_thought',
    'tree_of_thoughts_beam_search',
]
