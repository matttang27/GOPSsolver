from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests import test_support  # noqa: F401 - ensures ai/ is on sys.path

import linprog


class _FakeVar:
    def __init__(self, value: float = 0.0) -> None:
        self._value = float(value)

    def solution_value(self) -> float:
        return self._value


class _FakeConstraint:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.coeffs: dict[object, float] = {}

    def Clear(self) -> None:
        self.clear_calls += 1
        self.coeffs.clear()

    def SetCoefficient(self, var: object, value: float) -> None:
        self.coeffs[var] = float(value)


class _FakeSolver:
    def __init__(
        self,
        *,
        num_rows: int,
        num_cols: int,
        status: int,
        probs: list[float] | None = None,
        game_value: float = 0.0,
    ) -> None:
        probs = probs if probs is not None else [0.0] * num_rows
        self.p = [_FakeVar(v) for v in probs]
        self.v = _FakeVar(game_value)
        self.constraints = [_FakeConstraint() for _ in range(num_cols)]
        self._status = status
        self.solve_calls = 0

    def Solve(self) -> int:
        self.solve_calls += 1
        return self._status


class TestFindBestStrategyControlFlow(unittest.TestCase):
    def setUp(self) -> None:
        linprog.solver_cache.clear()

    def tearDown(self) -> None:
        linprog.solver_cache.clear()

    def test_or_tools_optimal_does_not_call_fallback(self) -> None:
        payoff = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        fake_solver = _FakeSolver(
            num_rows=2,
            num_cols=2,
            status=linprog.pywraplp.Solver.OPTIMAL,
            probs=[0.25, 0.75],
            game_value=0.125,
        )

        with (
            mock.patch.object(linprog, "get_solver_for_size", return_value=fake_solver) as mocked_get_solver,
            mock.patch.object(linprog, "findBestStrategy_scipy_fallback") as mocked_fallback,
        ):
            probs, value = linprog.findBestStrategy(payoff)

        mocked_get_solver.assert_called_once_with(2, 2)
        mocked_fallback.assert_not_called()
        self.assertEqual(fake_solver.solve_calls, 1)

        np.testing.assert_allclose(probs, np.array([0.25, 0.75], dtype=np.float64), rtol=0.0, atol=0.0)
        self.assertAlmostEqual(value, 0.125, places=12)

        for j, constraint in enumerate(fake_solver.constraints):
            self.assertEqual(constraint.clear_calls, 1)
            self.assertAlmostEqual(constraint.coeffs[fake_solver.p[0]], payoff[0, j], places=12)
            self.assertAlmostEqual(constraint.coeffs[fake_solver.p[1]], payoff[1, j], places=12)
            self.assertAlmostEqual(constraint.coeffs[fake_solver.v], -1.0, places=12)

    def test_non_optimal_status_calls_scipy_fallback(self) -> None:
        payoff = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        fake_solver = _FakeSolver(
            num_rows=2,
            num_cols=2,
            status=linprog.pywraplp.Solver.INFEASIBLE,
        )
        expected_probs = np.array([0.6, 0.4], dtype=np.float64)
        expected_value = -0.2

        with (
            mock.patch.object(linprog, "get_solver_for_size", return_value=fake_solver),
            mock.patch.object(
                linprog,
                "findBestStrategy_scipy_fallback",
                return_value=(expected_probs, expected_value),
            ) as mocked_fallback,
        ):
            probs, value = linprog.findBestStrategy(payoff)

        mocked_fallback.assert_called_once()
        call_matrix = mocked_fallback.call_args.args[0]
        np.testing.assert_allclose(call_matrix, payoff, rtol=0.0, atol=0.0)
        self.assertIs(probs, expected_probs)
        self.assertEqual(value, expected_value)


class TestScipyFallbackMath(unittest.TestCase):
    def _assert_valid_strategy(
        self,
        payoff: np.ndarray,
        probs: np.ndarray | None,
        value: float | None,
        *,
        tol: float = 1e-8,
    ) -> None:
        self.assertIsNotNone(probs)
        self.assertIsNotNone(value)
        assert probs is not None
        assert value is not None

        self.assertEqual(probs.ndim, 1)
        self.assertEqual(probs.size, payoff.shape[0])
        self.assertTrue(bool(np.all(probs >= -tol)))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=7)

        column_values = probs @ payoff
        self.assertGreaterEqual(float(np.min(column_values)), float(value) - 1e-7)

    def test_matching_pennies_returns_symmetric_mixed_strategy(self) -> None:
        payoff = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        probs, value = linprog.findBestStrategy_scipy_fallback(payoff)
        self._assert_valid_strategy(payoff, probs, value)
        assert probs is not None
        assert value is not None
        np.testing.assert_allclose(probs, np.array([0.5, 0.5], dtype=np.float64), atol=1e-6, rtol=0.0)
        self.assertAlmostEqual(value, 0.0, places=6)

    def test_dominated_row_game_prefers_best_row(self) -> None:
        payoff = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        probs, value = linprog.findBestStrategy_scipy_fallback(payoff)
        self._assert_valid_strategy(payoff, probs, value)
        assert probs is not None
        assert value is not None
        self.assertGreaterEqual(float(probs[0]), 1.0 - 1e-7)
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_single_row_game_has_trivial_distribution(self) -> None:
        payoff = np.array([[2.0, -3.0, 1.0]], dtype=np.float64)
        probs, value = linprog.findBestStrategy_scipy_fallback(payoff)
        self._assert_valid_strategy(payoff, probs, value)
        assert probs is not None
        assert value is not None
        np.testing.assert_allclose(probs, np.array([1.0], dtype=np.float64), atol=1e-9, rtol=0.0)
        self.assertAlmostEqual(value, -3.0, places=6)


class TestSolverCache(unittest.TestCase):
    def setUp(self) -> None:
        linprog.solver_cache.clear()

    def tearDown(self) -> None:
        linprog.solver_cache.clear()

    def test_get_solver_for_size_reuses_by_shape(self) -> None:
        solver_a = linprog.get_solver_for_size(2, 3)
        solver_b = linprog.get_solver_for_size(2, 3)
        solver_c = linprog.get_solver_for_size(3, 2)

        self.assertIs(solver_a, solver_b)
        self.assertIsNot(solver_a, solver_c)
        self.assertIn((2, 3), linprog.solver_cache)
        self.assertIn((3, 2), linprog.solver_cache)

    def test_reused_solver_updates_coefficients_for_new_matrix(self) -> None:
        payoff1 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        probs1, value1 = linprog.findBestStrategy(payoff1)
        self.assertIsNotNone(probs1)
        self.assertIsNotNone(value1)

        solver_before = linprog.get_solver_for_size(2, 2)
        payoff2 = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        probs2, value2 = linprog.findBestStrategy(payoff2)
        solver_after = linprog.get_solver_for_size(2, 2)

        self.assertIs(solver_before, solver_after)
        self.assertIsNotNone(probs2)
        self.assertIsNotNone(value2)
        assert probs2 is not None
        assert value2 is not None
        self.assertGreaterEqual(float(probs2[0]), 1.0 - 1e-6)
        self.assertAlmostEqual(float(value2), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
