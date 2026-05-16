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


def test_mcp_basic_symbols() -> None:
    mod = load_module("mcp_basic.py")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_client_demo")


def test_mcp_advanced_symbols() -> None:
    mod = load_module("mcp_advanced.py")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_agent_orchestration_demo")
