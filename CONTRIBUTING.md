# Contributing to Agentic Design Patterns

Thank you for your interest in contributing! This repository aims to be a comprehensive, high-quality resource for AI agent design patterns. We welcome contributions of all kinds.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Submitting a New Pattern](#submitting-a-new-pattern)
- [Pattern Quality Criteria](#pattern-quality-criteria)
- [Code Style Guide](#code-style-guide)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## 🤝 Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to gteseil@yahoo.com.

## 🎯 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

- **Pattern name** and file path
- **Python version** and dependencies (from `pyproject.toml`)
- **LangChain/LangGraph versions**
- **Expected vs. actual behavior**
- **Minimal reproducible example**
- **Error messages** and stack traces

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) when available.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear description** of the enhancement
- **Use case** and motivation
- **Expected behavior** or outcome
- **Alternative solutions** you've considered

### Proposing New Patterns

We're always looking for new, valuable patterns! When proposing a pattern:

1. **Search existing patterns** to avoid duplicates
2. **Create an issue** using the pattern proposal template
3. **Include**:
   - Pattern name and category (foundational/reasoning/reliability/orchestration/observability/memory/learning)
   - Problem it solves
   - When to use (and when not to use)
   - Academic references (papers, if applicable)
   - Implementation approach
   - Example use cases

### Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or use cases
- Improve code comments
- Translate documentation
- Create tutorial videos or blog posts

### Contributing Code

Code contributions include:

- Implementing new patterns
- Adding examples to existing patterns
- Improving error handling
- Adding type hints
- Writing tests
- Performance optimizations

## 🛠️ Development Setup

### Prerequisites

- **Python 3.11+** (3.13 has known issues with some dependencies)
- **uv** (fast Python package manager)
- **Git**
- **OpenAI API key** (for testing patterns)

### Initial Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/agentic_design_patterns.git
cd agentic_design_patterns

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Install dependencies for a specific pattern
cd foundational_design_patterns/8_react
uv sync

# Run the pattern to verify setup
bash run.sh
```

### Project Structure

```
agentic_design_patterns/
├── foundational_design_patterns/  # Core patterns (1-10)
├── reasoning/                     # Advanced reasoning patterns
├── reliability/                   # Safety and resilience patterns
├── orchestration/                 # Multi-agent coordination
├── observability/                 # Monitoring and optimization
├── memory/                        # Context and history management
├── learning/                      # Continuous improvement
├── .github/                       # GitHub templates and workflows
├── docs/                          # Additional documentation
└── README.md                      # Main documentation
```

## 📝 Submitting a New Pattern

### Pattern Structure

Each pattern should follow this structure:

```
pattern_category/pattern_name/
├── README.md              # Comprehensive documentation (560+ lines)
├── QUICK_START.md         # Quick start guide
├── pyproject.toml         # Dependencies
├── run.sh                 # Execution script (must be executable)
└── src/
    ├── __init__.py        # Package initialization
    ├── pattern_basic.py   # Basic implementation
    └── pattern_advanced.py # Advanced implementation (if applicable)
```

### README.md Template

Your pattern README should include:

```markdown
# Pattern Name

## Overview
[Clear explanation of the pattern]

## Why Use This Pattern?
[Problems it solves, limitations it addresses]

## How It Works
[Step-by-step explanation with diagrams]

## When to Use
[Ideal use cases and anti-patterns]

## Core Components
[Key building blocks]

## Implementation Approaches
[Different ways to implement]

## Key Benefits
[What you gain]

## Trade-offs
[Costs and limitations]

## Best Practices
[Dos and don'ts with code examples]

## Performance Metrics
[How to measure success]

## Example Scenarios
[Real-world applications]

## Advanced Patterns
[Extensions and variations]

## Comparison with Related Patterns
[How it differs from similar patterns]

## Common Pitfalls
[Mistakes to avoid]

## Conclusion
[Summary with implementation checklist]
```

### Code Requirements

All code contributions must:

1. **Follow Python best practices**
   - PEP 8 style guide
   - Type hints for function parameters and returns
   - Docstrings for classes and functions

2. **Include comprehensive comments**
   - Explain non-obvious logic
   - Describe design decisions
   - Reference academic papers when applicable

3. **Handle errors gracefully**
   - Use try-except blocks appropriately
   - Provide informative error messages
   - Fail safely

4. **Be educational**
   - Prioritize clarity over cleverness
   - Include print statements for visibility
   - Show intermediate steps

5. **Work out of the box**
   - Load API keys from `../../.env` (or appropriate path)
   - Include all required dependencies in `pyproject.toml`
   - Provide working examples

## ✅ Pattern Quality Criteria

To maintain high quality, patterns must meet these criteria:

### 1. Academic Grounding (when applicable)
- Cite relevant research papers
- Link to academic sources
- Explain theoretical foundations

### 2. Comprehensive Documentation
- README.md: 560+ lines
- QUICK_START.md: Getting started guide
- Clear explanations of when/why to use

### 3. Working Code Examples
- At least one basic implementation
- Optionally, an advanced implementation
- All examples must run successfully

### 4. Educational Value
- Clear variable names
- Extensive comments
- Step-by-step walkthroughs
- Expected outputs shown

### 5. Production Considerations
- Error handling
- Resource management
- Performance metrics
- Security best practices

### 6. Proper Attribution
- Credit original authors
- Link to source papers/repos
- Maintain MIT license compatibility

## 🎨 Code Style Guide

### Python Style

We use **Ruff** for linting and formatting. Configuration is in `pyproject.toml`.

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Key Style Points

```python
# ✅ Good: Type hints, docstrings, clear names
def calculate_priority_score(
    urgency: float,
    impact: float,
    effort: float
) -> float:
    """Calculate task priority score.

    Args:
        urgency: Time sensitivity (0-10)
        impact: Business value (0-10)
        effort: Implementation cost (0-10)

    Returns:
        Priority score (higher is more important)
    """
    return (urgency * impact) / max(effort, 1)


# ❌ Bad: No types, unclear names, no docstring
def calc(a, b, c):
    return (a * b) / max(c, 1)
```

### Import Organization

```python
# Standard library imports
import os
from typing import List, Optional

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Local imports
from .utils import format_response
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ReActAgent`)
- **Functions**: `snake_case` (e.g., `generate_thoughts`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- **Private**: `_leading_underscore` (e.g., `_internal_method`)

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes**
   ```bash
   # Run the pattern
   bash run.sh

   # Verify output is correct
   ```

2. **Check code style**
   ```bash
   ruff check .
   ruff format .
   ```

3. **Update documentation**
   - Update README if adding features
   - Add entries to CHANGELOG.md
   - Update QUICK_START if needed

4. **Commit with clear messages**
   ```bash
   # Good commit messages
   git commit -m "Add error recovery pattern with circuit breaker"
   git commit -m "Fix type hints in routing example"
   git commit -m "Improve ReAct pattern documentation"

   # Bad commit messages
   git commit -m "updates"
   git commit -m "fix"
   ```

### PR Guidelines

1. **Create a descriptive PR title**
   - `feat: Add Graph of Thoughts pattern`
   - `fix: Correct token counting in context management`
   - `docs: Improve RAG pattern documentation`
   - `refactor: Simplify multi-agent communication`

2. **Fill out the PR template** (when available)
   - Description of changes
   - Motivation and context
   - Type of change (bugfix, feature, docs)
   - Testing performed
   - Screenshots (if applicable)

3. **Link related issues**
   - Use `Closes #123` to auto-close issues
   - Reference related discussions

4. **Request review**
   - Tag relevant maintainers
   - Be responsive to feedback
   - Make requested changes promptly

5. **Keep PRs focused**
   - One feature/fix per PR
   - Avoid unrelated changes
   - Split large PRs into smaller ones

### Review Process

1. **Automated checks** run (linting, tests)
2. **Maintainer review** (1-3 days typically)
3. **Feedback addressed** by contributor
4. **Approval and merge** by maintainer

## 🌐 Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Q&A, ideas, show & tell
- **Twitter/X**: [@gtesei](https://twitter.com/gtesei) (if applicable)

### Getting Help

- Check [existing issues](https://github.com/gtesei/agentic_design_patterns/issues)
- Search [discussions](https://github.com/gtesei/agentic_design_patterns/discussions)
- Review [documentation](https://github.com/gtesei/agentic_design_patterns/blob/main/README.md)
- Ask in [Q&A discussions](https://github.com/gtesei/agentic_design_patterns/discussions/categories/q-a)

### Recognition

Contributors are recognized in:
- **README.md acknowledgments section**
- **Release notes** for significant contributions
- **GitHub contributors page**

## 📚 Resources

### Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601) (Yao et al., 2023)
- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136) (Singh et al., 2025)

### Related Projects

- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Questions?** Open a [discussion](https://github.com/gtesei/agentic_design_patterns/discussions/new) or reach out to the maintainers.

**Ready to contribute?** Check out [good first issues](https://github.com/gtesei/agentic_design_patterns/labels/good-first-issue)!
