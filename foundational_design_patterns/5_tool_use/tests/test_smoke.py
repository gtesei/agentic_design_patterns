import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def load_module(filename: str):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_tool_use_basic_symbols() -> None:
    mod = load_module("tool_use.py")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_parallel_enrichment")
    assert hasattr(mod, "run_agentic_response")


def test_tool_use_advanced_symbols() -> None:
    mod = load_module("tool_use_advanced.py")
    assert hasattr(mod, "main")
    assert hasattr(mod, "parallel_context")
    assert hasattr(mod, "compose_ops_plan")
