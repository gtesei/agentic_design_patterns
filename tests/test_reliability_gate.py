import os
import py_compile
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [
    REPO_ROOT / "foundational_design_patterns",
    REPO_ROOT / "reasoning",
    REPO_ROOT / "reliability",
    REPO_ROOT / "orchestration",
    REPO_ROOT / "observability",
    REPO_ROOT / "memory",
    REPO_ROOT / "learning",
    REPO_ROOT / "tests",
]


class ReliabilityGateTests(unittest.TestCase):
    def _repo_python_files(self):
        for path in [REPO_ROOT / "ssl_fix.py", REPO_ROOT / "repo_support.py"]:
            if path.exists():
                yield path

        for source_root in SOURCE_ROOTS:
            if not source_root.exists():
                continue

            for path in source_root.rglob("*.py"):
                if {".venv", "site-packages", "__pycache__"} & set(path.parts):
                    continue
                yield path

    def test_repo_support_finds_repo_root(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from repo_support import find_repo_root

        script_path = REPO_ROOT / "reliability/guardrails/src/guardrails_basic.py"
        self.assertEqual(find_repo_root(script_path), REPO_ROOT)

    def test_ssl_fix_is_opt_in_by_default(self) -> None:
        code = (
            "import ssl; "
            "original = ssl._create_default_https_context; "
            "import ssl_fix; "
            "print('same' if ssl._create_default_https_context is original else 'changed')"
        )
        env = os.environ.copy()
        env.pop("AGENTIC_DISABLE_SSL", None)

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        self.assertEqual(result.stdout.strip(), "same")

    def test_ssl_fix_can_be_enabled_explicitly(self) -> None:
        code = (
            "import ssl; "
            "original = ssl._create_default_https_context; "
            "import ssl_fix; "
            "print('changed' if ssl._create_default_https_context is not original else 'same')"
        )
        env = os.environ.copy()
        env["AGENTIC_DISABLE_SSL"] = "1"

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        self.assertEqual(result.stdout.strip(), "changed")

    def test_all_python_files_compile(self) -> None:
        for path in self._repo_python_files():
            py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main()
