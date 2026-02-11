from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path

from tests import test_support

from common import (
    cache_objective,
    cmp,
    compress_cards,
    decode_key,
    list_cards,
    load_evc,
    lowest_card,
    popcount,
    remove_card,
)


ROOT = test_support.ROOT
EPS = 1e-12


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


def _solve_linear_system(system: list[list[float]], rhs: list[float]) -> list[float] | None:
    n = len(system)
    augmented = [list(system[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot_row = None
        pivot_abs = 0.0
        for row in range(col, n):
            value = abs(augmented[row][col])
            if value > pivot_abs:
                pivot_abs = value
                pivot_row = row
        if pivot_row is None or pivot_abs < EPS:
            return None
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < EPS:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def _solve_zero_sum_game_value(payoff: list[list[float]]) -> float | None:
    n = len(payoff)
    best = -1.0e300
    found = False
    for support_size in range(1, n + 1):
        row_supports = itertools.combinations(range(n), support_size)
        for rows in row_supports:
            col_supports = itertools.combinations(range(n), support_size)
            for cols in col_supports:
                system: list[list[float]] = []
                rhs: list[float] = []
                for col in cols:
                    system.append([payoff[row][col] for row in rows] + [-1.0])
                    rhs.append(0.0)
                system.append([1.0] * support_size + [0.0])
                rhs.append(1.0)
                solution = _solve_linear_system(system, rhs)
                if solution is None:
                    continue
                probs = solution[:-1]
                value = solution[-1]
                if any(p < -EPS for p in probs):
                    continue
                min_payoff = None
                for col in range(n):
                    expected = 0.0
                    for i, row in enumerate(rows):
                        expected += probs[i] * payoff[row][col]
                    if min_payoff is None or expected < min_payoff:
                        min_payoff = expected
                if min_payoff is None or min_payoff + EPS < value:
                    continue
                if value > best:
                    best = value
                    found = True
    if not found:
        return None
    return best


class _ReferenceSolver:
    def __init__(self, objective: str) -> None:
        if objective not in {"win", "points"}:
            raise ValueError(f"unsupported objective: {objective}")
        self.objective = objective
        self.memo: dict[tuple[int, int, int, int, int], float] = {}

    def _canonicalize(self, A: int, B: int, P: int, diff: int, cur_p: int) -> tuple[tuple[int, int, int, int, int], float]:
        sign = 1.0
        if self.objective == "points":
            diff = 0
            if A < B:
                A, B = B, A
                sign = -sign
        else:
            if diff < 0:
                A, B = B, A
                diff = -diff
                sign = -sign
            if diff == 0 and A < B:
                A, B = B, A
                sign = -sign
        A, B = compress_cards(A, B)
        if self.objective == "points":
            diff = 0
        return (A, B, P, diff, cur_p), sign

    def solve(self, A: int, B: int, P: int, diff: int, cur_p: int) -> float:
        if self.objective == "points":
            diff = 0
        key, sign = self._canonicalize(A, B, P, diff, cur_p)
        value = self.memo.get(key)
        if value is None:
            value = self._solve_canonical(*key)
            self.memo[key] = value
        return sign * value

    def _solve_canonical(self, A: int, B: int, P: int, diff: int, cur_p: int) -> float:
        if A == B and (self.objective == "points" or diff == 0):
            return 0.0
        if popcount(A) == 1 and popcount(B) == 1 and P == 0:
            delta = cmp(lowest_card(A), lowest_card(B)) * cur_p
            if self.objective == "points":
                return float(delta)
            return float(cmp(diff + delta, 0))

        cards_a = list_cards(A)
        cards_b = list_cards(B)
        prizes = list_cards(P)
        size = len(cards_a)
        if size != len(cards_b) or len(prizes) != size - 1:
            raise ValueError(f"invalid canonical state: A={A} B={B} P={P} diff={diff} curP={cur_p}")

        payoff = [[0.0 for _ in range(size)] for _ in range(size)]
        for i, card_a in enumerate(cards_a):
            for j, card_b in enumerate(cards_b):
                new_a = remove_card(A, card_a)
                new_b = remove_card(B, card_b)
                delta = cmp(card_a, card_b) * cur_p
                new_diff = diff + delta
                total = 0.0
                for next_prize in prizes:
                    new_p = remove_card(P, next_prize)
                    child_diff = new_diff if self.objective == "win" else 0
                    total += self.solve(new_a, new_b, new_p, child_diff, next_prize)
                avg = total / len(prizes)
                if self.objective == "points":
                    avg += delta
                payoff[i][j] = avg

        value = _solve_zero_sum_game_value(payoff)
        if value is None:
            raise RuntimeError(f"support-enumeration LP failed for state A={A} B={B} P={P} diff={diff} curP={cur_p}")
        return float(value)


CPP_SOLVER = _find_cpp_solver()


@unittest.skipUnless(
    CPP_SOLVER is not None,
    "requires built solver binary (set GOPS_CPP_SOLVER or build solver/cpp_solver)",
)
class TestCppPythonParity(unittest.TestCase):
    N = 5
    TOL = 1e-9

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = ROOT / f"gops_cpp_parity_{uuid.uuid4().hex}"
        cls._tmp_dir.mkdir()
        tmp_dir = cls._tmp_dir
        cls.win_cache_path = tmp_dir / "parity_win_n5.evc"
        cls.points_cache_path = tmp_dir / "parity_points_n5.evc"
        cls._run_cpp_solver("win", cls.win_cache_path)
        cls._run_cpp_solver("points", cls.points_cache_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmp_dir"):
            shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @classmethod
    def _run_cpp_solver(cls, objective: str, cache_path: Path) -> None:
        cmd = [
            str(CPP_SOLVER),
            "--n",
            str(cls.N),
            "--objective",
            objective,
            "--cache-out",
            str(cache_path),
        ]
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cpp_solver failed ({objective}) with exit code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def _assert_cache_matches_reference(self, cache_path: Path, objective: str) -> None:
        cache = load_evc(str(cache_path))
        self.assertEqual(cache_objective(cache), objective)
        self.assertGreater(len(cache), 0)

        reference = _ReferenceSolver(objective)
        max_diff = 0.0
        worst_key = 0
        worst_state: tuple[int, int, int, int, int] | None = None

        for key in cache:
            A, B, P, diff, cur_p = decode_key(int(key))
            cpp_ev = float(cache[key])
            py_ev = reference.solve(A, B, P, diff, cur_p)
            delta = abs(cpp_ev - py_ev)
            if delta > max_diff:
                max_diff = delta
                worst_key = int(key)
                worst_state = (A, B, P, diff, cur_p)

        self.assertLessEqual(
            max_diff,
            self.TOL,
            msg=f"max abs diff={max_diff} at key={worst_key} state={worst_state}",
        )

    def test_win_objective_cache_matches_reference_solver(self) -> None:
        self._assert_cache_matches_reference(self.win_cache_path, "win")

    def test_points_objective_cache_matches_reference_solver(self) -> None:
        self._assert_cache_matches_reference(self.points_cache_path, "points")


if __name__ == "__main__":
    unittest.main()
