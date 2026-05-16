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


def test_rag_basic_symbols() -> None:
    mod = load_module("rag_basic.py")
    assert hasattr(mod, "HybridRetriever")
    assert hasattr(mod, "main")


def test_rag_agentic_symbols() -> None:
    mod = load_module("rag_agentic.py")
    assert hasattr(mod, "build_graph")
    assert hasattr(mod, "main")
