from __future__ import annotations

import unittest
from pathlib import Path

from tests import test_support

from common import load_evc, remove_card
from exploitability import ExploitabilityEvaluator, evaluate_average_exploitability


ROOT = test_support.ROOT
FULL6_CACHE = ROOT / "reports" / "full6.evc"


@unittest.skipUnless(FULL6_CACHE.exists(), "requires reports/full6.evc")
class TestIntegrationN5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache = load_evc(str(FULL6_CACHE))

    def test_evc_ne_exploitability_near_zero_at_n5(self) -> None:
        evaluator = ExploitabilityEvaluator(
            policy="evc-ne",
            eval_objective="win",
            policy_eps=1e-12,
            strategy_cache=self.cache,
            n=5,
        )
        result = evaluate_average_exploitability(n=5, evaluator=evaluator)
        self.assertLess(abs(result["avg_max_exploitability"]), 1e-9)
        self.assertEqual(len(result["per_initial_prize_value_a"]), 5)

    def test_policy_distribution_well_formed_for_each_initial_prize_n5(self) -> None:
        evaluator = ExploitabilityEvaluator(
            policy="evc-ne",
            eval_objective="win",
            policy_eps=1e-12,
            strategy_cache=self.cache,
            n=5,
        )
        full = (1 << 5) - 1
        for cur_p in range(1, 6):
            with self.subTest(cur_p=cur_p):
                p_mask = remove_card(full, cur_p)
                actions, probs = evaluator.policy_distribution(full, full, p_mask, 0, cur_p)
                self.assertEqual(len(actions), len(probs))
                self.assertGreater(len(actions), 0)
                self.assertAlmostEqual(float(sum(probs)), 1.0, places=12)
                self.assertTrue(all(1 <= a <= 5 for a in actions))
                self.assertTrue(all(float(p) >= 0.0 for p in probs))


if __name__ == "__main__":
    unittest.main()
