from __future__ import annotations

import random
import unittest

from tests import test_support

from common import State
from play import run_game
from strategies import (
    build_strategy,
    make_current_strategy,
    make_exploit_current_strategy,
    sample_action,
    strategy_choices,
    strategy_label,
    strategy_requires_cache,
)


class _FixedRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value


class TestStrategies(unittest.TestCase):
    def test_strategy_registry(self) -> None:
        choices = strategy_choices()
        self.assertIn("evc-ne", choices)
        self.assertTrue(strategy_requires_cache("evc-ne"))
        self.assertFalse(strategy_requires_cache("random"))

    def test_build_strategy_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_strategy("not-a-strategy")

    def test_sample_action_uses_last_action_for_probability_slack(self) -> None:
        action = sample_action([1, 2], probs=[0.3, 0.3], rng=_FixedRng(0.95))
        self.assertEqual(action, 2)

    def test_current_strategy_rng_fallback_is_deterministic(self) -> None:
        state = State(
            A=test_support.mask([1, 3]),
            B=test_support.mask([1, 3]),
            P=0,
            diff=0,
            curP=2,  # not in hand, so fallback path is used
        )
        s1 = make_current_strategy(rng=random.Random(123))
        s2 = make_current_strategy(rng=random.Random(123))
        self.assertEqual(s1(state), s2(state))

    def test_exploit_current_strategy_prefers_curp_plus_one(self) -> None:
        state = State(
            A=test_support.mask([1, 3]),
            B=test_support.mask([1, 3]),
            P=0,
            diff=0,
            curP=2,
        )
        strat = make_exploit_current_strategy(rng=random.Random(1))
        self.assertEqual(strat(state), 3)

    def test_strategy_label(self) -> None:
        self.assertEqual(strategy_label("random", None), "random")
        self.assertEqual(strategy_label("random", 0), "random(seed=random)")
        self.assertEqual(strategy_label("random", 7), "random(seed=7)")


class TestRunGame(unittest.TestCase):
    def test_invalid_action_raises(self) -> None:
        def bad_choose(state_a: State, state_b: State):
            del state_b
            return (99, 1)  # invalid for A

        with self.assertRaises(ValueError):
            run_game(3, 123, bad_choose)

    def test_highest_vs_highest_is_always_draw_for_n_up_to_5(self) -> None:
        for n in range(1, 6):
            with self.subTest(n=n):
                sa = build_strategy("highest")
                sb = build_strategy("highest")

                def choose(state_a: State, state_b: State):
                    return sa(state_a), sb(state_b)

                diff = run_game(n, 42, choose)
                self.assertEqual(diff, 0)

    def test_random_vs_random_diff_in_valid_range_for_n_up_to_5(self) -> None:
        for n in range(1, 6):
            with self.subTest(n=n):
                max_abs = n * (n + 1) // 2
                sa = build_strategy("random", rng=random.Random(100 + n))
                sb = build_strategy("random", rng=random.Random(200 + n))

                def choose(state_a: State, state_b: State):
                    return sa(state_a), sb(state_b)

                diff = run_game(n, 500 + n, choose)
                self.assertLessEqual(abs(diff), max_abs)


if __name__ == "__main__":
    unittest.main()
