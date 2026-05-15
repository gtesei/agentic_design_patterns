"""Shared runtime helpers for example scripts in this repository."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

SSL_BYPASS_ENV_VAR = "AGENTIC_DISABLE_SSL"
DEFAULT_MODEL_ENV_VAR = "OPENAI_MODEL"
REPO_SENTINELS = ("ssl_fix.py", "README.md")


@lru_cache(maxsize=None)
def find_repo_root(start: str | Path) -> Path:
    """Find the repository root from an example file or directory."""
    candidate = Path(start).resolve()
    search_from = candidate if candidate.is_dir() else candidate.parent

    for parent in (search_from, *search_from.parents):
        if all((parent / sentinel).exists() for sentinel in REPO_SENTINELS):
            return parent

    raise RuntimeError(f"Could not locate repo root from {start!r}")


def ensure_repo_on_path(start: str | Path) -> Path:
    """Add the repository root to ``sys.path`` if needed."""
    repo_root = find_repo_root(start)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def load_repo_env(start: str | Path) -> Path:
    """Load the repository ``.env`` file when python-dotenv is available."""
    repo_root = find_repo_root(start)
    env_path = repo_root / ".env"

    if not env_path.exists():
        return env_path

    try:
        from dotenv import load_dotenv
    except ImportError:
        return env_path

    load_dotenv(env_path, override=False)
    return env_path


def ssl_bypass_requested(env_var: str = SSL_BYPASS_ENV_VAR) -> bool:
    """Return ``True`` when insecure SSL mode was explicitly requested."""
    value = os.getenv(env_var, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_example(start: str | Path) -> Path:
    """Prepare a script to run from anywhere in the repository."""
    repo_root = ensure_repo_on_path(start)
    load_repo_env(start)

    if ssl_bypass_requested():
        import ssl_fix

        ssl_fix.apply_ssl_bypass()

    return repo_root


def get_default_model(default: str = "gpt-4o-mini", env_var: str = DEFAULT_MODEL_ENV_VAR) -> str:
    """Resolve the default chat model for examples."""
    return os.getenv(env_var, default)
