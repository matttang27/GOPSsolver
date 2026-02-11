from __future__ import annotations

import unittest

import numpy as np

from tests import test_support

from common import build_matrix, encode_key, get_ev


class PointsCache(dict):
    objective = "points"


class TestCommonEvFallbacks(unittest.TestCase):
    def test_terminal_fallback_uses_diff_plus_last_delta(self) -> None:
        a = test_support.mask([1])
        b = test_support.mask([2])
        p = 0
        cur_p = 1

        # diff + delta = 1 + (-1) = 0
        self.assertEqual(get_ev({}, a, b, p, 1, cur_p), 0.0)
        # diff + delta = 2 + (-1) = 1
        self.assertEqual(get_ev({}, a, b, p, 2, cur_p), 1.0)
        # diff + delta = 0 + (-1) = -1
        self.assertEqual(get_ev({}, a, b, p, 0, cur_p), -1.0)

    def test_guarantee_fallback_includes_current_prize(self) -> None:
        # This state is not terminal and key is missing, but with curP included
        # there is a guaranteed win for A.
        a = test_support.mask([2, 3])
        b = test_support.mask([1, 2])
        self.assertEqual(get_ev({}, a, b, 0, 0, 3), 1.0)
        self.assertEqual(get_ev({}, b, a, 0, 0, 3), -1.0)

    def test_missing_non_guaranteed_state_raises(self) -> None:
        a = test_support.mask([1, 2])
        b = test_support.mask([1, 2])
        p = test_support.mask([3])
        with self.assertRaises(KeyError):
            get_ev({}, a, b, p, 1, 4)


class TestBuildMatrix(unittest.TestCase):
    def test_terminal_win_matrix(self) -> None:
        a = test_support.mask([1])
        b = test_support.mask([2])
        mat = build_matrix({}, a, b, 0, 1, 1)
        self.assertEqual(mat, [[0.0]])

    def test_terminal_points_matrix(self) -> None:
        a = test_support.mask([1])
        b = test_support.mask([2])
        mat = build_matrix(PointsCache(), a, b, 0, 999, 3)
        self.assertEqual(mat, [[-3.0]])

    def test_nonterminal_points_matrix_with_known_child_values(self) -> None:
        cache = PointsCache()
        # Child state used by mismatched plays: A=[2], B=[1], P=[], curP=4
        key = encode_key(test_support.mask([2]), test_support.mask([1]), 0, 0, 4)
        cache[key] = 2.0

        a = test_support.mask([1, 2])
        b = test_support.mask([1, 2])
        p = test_support.mask([4])
        mat = build_matrix(cache, a, b, p, 0, 3)

        expected = np.array(
            [
                [0.0, -1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(np.array(mat, dtype=np.float64), expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
