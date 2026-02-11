from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from tests import test_support


ROOT = test_support.ROOT


def _find_cpp_solver() -> Path | None:
    env_path = os.environ.get("GOPS_CPP_SOLVER")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    defaults = [
        ROOT / "solver" / "build" / "vcpkg" / "Release" / "cpp_solver.exe",
        ROOT / "solver" / "build" / "vcpkg" / "Release" / "cpp_solver",
        ROOT / "solver" / "build" / "cpp_solver.exe",
        ROOT / "solver" / "build" / "cpp_solver",
    ]
    for candidate in defaults:
        if candidate.exists():
            return candidate

    build_dir = ROOT / "solver" / "build"
    if not build_dir.exists():
        return None
    for candidate in sorted(build_dir.rglob("cpp_solver*")):
        if candidate.is_file() and candidate.stem == "cpp_solver":
            return candidate
    return None


CPP_SOLVER = _find_cpp_solver()


@unittest.skipUnless(
    CPP_SOLVER is not None,
    "requires built solver binary (set GOPS_CPP_SOLVER or build solver/cpp_solver)",
)
class TestCppSolverCli(unittest.TestCase):
    def _run_solver(self, *args: str) -> tuple[int, str]:
        result = subprocess.run(
            [str(CPP_SOLVER), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        return result.returncode, combined

    def test_help_succeeds(self) -> None:
        code, out = self._run_solver("--help")
        self.assertEqual(code, 0)
        self.assertIn("Usage:", out)

    def test_invalid_objective_fails(self) -> None:
        code, out = self._run_solver("--objective", "not-valid")
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid --objective value", out)

    def test_missing_n_value_fails(self) -> None:
        code, out = self._run_solver("--n")
        self.assertNotEqual(code, 0)
        self.assertIn("Missing value for --n", out)

    def test_non_numeric_n_fails(self) -> None:
        code, out = self._run_solver("--n", "abc")
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid --n value", out)

    def test_zero_n_fails(self) -> None:
        code, out = self._run_solver("--n", "0")
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid --n", out)

    def test_too_large_n_fails(self) -> None:
        code, out = self._run_solver("--n", "17")
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid --n", out)

    def test_lp_bench_non_numeric_arg_fails(self) -> None:
        code, out = self._run_solver("lp-bench", "abc")
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid minN for lp-bench", out)


if __name__ == "__main__":
    unittest.main()
