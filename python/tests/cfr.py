# FOR ANYONE READING THIS - THIS WAS A TEST IMPLEMENTATION, CFR IS NOT A VIABLE SOLUTION
# It implements CFR and CFR+, evaluation helpers, and a CLI to run experiments.
"""
Goofspiel CFR / CFR+ demo
=========================

- Solves N-card Goofspiel (public, fixed prize order 1..N) with terminal utility:
    +1 if P1 total prize points > P2
    -1 if P1 total prize points < P2
     0 if tie
  (Round ties split prize; they don't change score that round.)

- Two algorithms:
    * CFR (vanilla): standard regret matching, average from iter 1
    * CFR+       : regret clipping (>= 0) + linear averaging after a delay

- Reports root average strategy and exploitability over checkpoints.
- Optional plotting if matplotlib is installed.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple
import argparse
import math
import random
import sys

try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -------------------- Game definition --------------------

def util_from_score(sd: float) -> float:
    """Terminal utility for P1 given score difference (P1 - P2)."""
    if sd > 0: return 1.0
    if sd < 0: return -1.0
    return 0.0


def round_delta(a: int, b: int, prize: int) -> float:
    """Score change for the round with actions a (P1), b (P2), prize value."""
    if a > b: return prize
    if b > a: return -prize
    return 0.0  # tie splits; cancels


def key_p1(p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...], k: int):
    """Information set key for P1 (what P1 can condition on)."""
    return (p1_hand, p2_hand, k)


def key_p2(p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...], k: int):
    """Information set key for P2 (symmetrized)."""
    return (p2_hand, p1_hand, k)


def regret_matching(regrets: Dict[int, float], actions: Iterable[int]) -> Dict[int, float]:
    pos = [max(regrets.get(a, 0.0), 0.0) for a in actions]
    s = sum(pos)
    acts = list(actions)
    if s <= 1e-16:
        # uniform
        return {a: 1.0/len(acts) for a in acts}
    return {a: r/s for a, r in zip(acts, pos)}


# -------------------- CFR base class --------------------

class CFRBase:
    def __init__(self, prizes: List[int]):
        self.prizes = prizes
        self.regret1 = defaultdict(lambda: defaultdict(float))  # I -> a -> R
        self.regret2 = defaultdict(lambda: defaultdict(float))
        self.strat_sum1 = defaultdict(lambda: defaultdict(float))
        self.strat_sum2 = defaultdict(lambda: defaultdict(float))

    def _accumulate_strategy(self, k1, k2, sigma1, sigma2, reach1, reach2, weight=1.0):
        if weight <= 0: return
        for a, p in sigma1.items():
            self.strat_sum1[k1][a] += weight * reach2 * p
        for b, q in sigma2.items():
            self.strat_sum2[k2][b] += weight * reach1 * q

    def _sigma_from_regrets(self, k1, k2, A1, A2):
        return regret_matching(self.regret1[k1], A1), regret_matching(self.regret2[k2], A2)

    # To be implemented in subclass:
    def cfr(self, p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...], k: int, sd: float, reach1: float, reach2: float):
        raise NotImplementedError

    def average_strategies(self):
        def norm(table):
            out = {}
            for I, w in table.items():
                tot = sum(w.values())
                if tot <= 1e-16:
                    acts = list(w.keys())
                    out[I] = {a: 1.0/len(acts) for a in acts} if acts else {}
                else:
                    out[I] = {a: w[a]/tot for a in w}
            return out
        return norm(self.strat_sum1), norm(self.strat_sum2)


class CFR(CFRBase):
    """Vanilla CFR with immediate averaging."""
    def cfr(self, p1_hand, p2_hand, k, sd, reach1, reach2):
        if k == len(self.prizes):
            return util_from_score(sd)

        p1 = tuple(sorted(p1_hand))
        p2 = tuple(sorted(p2_hand))
        A1 = list(p1); A2 = list(p2)
        I1 = key_p1(p1, p2, k)
        I2 = key_p2(p1, p2, k)
        prize = self.prizes[k]

        sigma1, sigma2 = self._sigma_from_regrets(I1, I2, A1, A2)

        # average strategy accumulation (opponent reach weighting)
        self._accumulate_strategy(I1, I2, sigma1, sigma2, reach1, reach2, weight=1.0)

        # Recurse for all joint actions
        u_child = {}
        for a in A1:
            for b in A2:
                new_p1 = list(p1); new_p1.remove(a)
                new_p2 = list(p2); new_p2.remove(b)
                sd2 = sd + round_delta(a, b, prize)
                u_child[(a, b)] = self.cfr(tuple(new_p1), tuple(new_p2), k+1, sd2,
                                           reach1 * sigma1[a], reach2 * sigma2[b])

        v_sigma = sum(sigma1[a] * sigma2[b] * u_child[(a,b)] for a in A1 for b in A2)
        v1_a = {a: sum(sigma2[b] * u_child[(a,b)] for b in A2) for a in A1}
        v2_b = {b: sum(sigma1[a] * (-u_child[(a,b)]) for a in A1) for b in A2}

        # Regret updates (can be negative)
        for a in A1:
            self.regret1[I1][a] += reach2 * (v1_a[a] - v_sigma)
        for b in A2:
            self.regret2[I2][b] += reach1 * (v2_b[b] - (-v_sigma))

        return v_sigma


class CFRPlus(CFRBase):
    """CFR+ with regret clipping and linear averaging after a delay."""
    def __init__(self, prizes: List[int], avg_delay_frac: float = 0.5, total_iters: int = 1000):
        super().__init__(prizes)
        self.t = 0
        self.avg_start = max(1, int(total_iters * avg_delay_frac))

    def cfr(self, p1_hand, p2_hand, k, sd, reach1, reach2):
        if k == len(self.prizes):
            return util_from_score(sd)

        p1 = tuple(sorted(p1_hand))
        p2 = tuple(sorted(p2_hand))
        A1 = list(p1); A2 = list(p2)
        I1 = key_p1(p1, p2, k)
        I2 = key_p2(p1, p2, k)
        prize = self.prizes[k]

        sigma1, sigma2 = self._sigma_from_regrets(I1, I2, A1, A2)

        u_child = {}
        for a in A1:
            for b in A2:
                new_p1 = list(p1); new_p1.remove(a)
                new_p2 = list(p2); new_p2.remove(b)
                sd2 = sd + round_delta(a, b, prize)
                u_child[(a, b)] = self.cfr(tuple(new_p1), tuple(new_p2), k+1, sd2,
                                           reach1 * sigma1[a], reach2 * sigma2[b])

        v_sigma = sum(sigma1[a] * sigma2[b] * u_child[(a,b)] for a in A1 for b in A2)
        v1_a = {a: sum(sigma2[b] * u_child[(a,b)] for b in A2) for a in A1}
        v2_b = {b: sum(sigma1[a] * (-u_child[(a,b)]) for a in A1) for b in A2}

        # CFR+ regret clipping
        for a in A1:
            self.regret1[I1][a] = max(self.regret1[I1][a] + reach2 * (v1_a[a] - v_sigma), 0.0)
        for b in A2:
            self.regret2[I2][b] = max(self.regret2[I2][b] + reach1 * (v2_b[b] - (-v_sigma)), 0.0)

        return v_sigma

    def accumulate_average_strategy(self, p1_hand, p2_hand, k, reach1, reach2):
        """Accumulate average strategy for CFR+ after delay."""
        if k == len(self.prizes):
            return
        p1 = tuple(sorted(p1_hand))
        p2 = tuple(sorted(p2_hand))
        A1 = list(p1); A2 = list(p2)
        I1 = key_p1(p1, p2, k)
        I2 = key_p2(p1, p2, k)
        sigma1, sigma2 = self._sigma_from_regrets(I1, I2, A1, A2)
        # Only accumulate after averaging delay
        self._accumulate_strategy(I1, I2, sigma1, sigma2, reach1, reach2, weight=1.0)
        for a in A1:
            for b in A2:
                new_p1 = list(p1); new_p1.remove(a)
                new_p2 = list(p2); new_p2.remove(b)
                self.accumulate_average_strategy(tuple(new_p1), tuple(new_p2), k+1,
                                                reach1 * sigma1[a], reach2 * sigma2[b])


# -------------------- Evaluation helpers --------------------

def eval_policy_value(prizes: List[int], avg1: Dict, avg2: Dict) -> float:
    """Compute expected utility of (avg1, avg2) from the root."""
    @lru_cache(maxsize=None)
    def V(p1: Tuple[int, ...], p2: Tuple[int, ...], k: int, sd: float) -> float:
        if k == len(prizes):
            return util_from_score(sd)
        p1s = tuple(sorted(p1)); p2s = tuple(sorted(p2))
        I1 = key_p1(p1s, p2s, k)
        I2 = key_p2(p1s, p2s, k)
        A1 = list(p1s); A2 = list(p2s)
        sigma1 = avg1.get(I1, {a: 1.0/len(A1) for a in A1})
        sigma2 = avg2.get(I2, {b: 1.0/len(A2) for b in A2})
        prize = prizes[k]
        val = 0.0
        for a, pa in sigma1.items():
            for b, pb in sigma2.items():
                new_p1 = list(p1s); new_p1.remove(a)
                new_p2 = list(p2s); new_p2.remove(b)
                sd2 = sd + round_delta(a, b, prize)
                val += pa * pb * V(tuple(new_p1), tuple(new_p2), k+1, sd2)
        return val

    N = len(prizes)
    return V(tuple(range(1, N+1)), tuple(range(1, N+1)), 0, 0.0)


def best_response_value_vs_p2(prizes: List[int], avg2: Dict) -> float:
    """P1's best response EV against fixed P2 policy avg2."""
    @lru_cache(maxsize=None)
    def BR(p1: Tuple[int, ...], p2: Tuple[int, ...], k: int, sd: float) -> float:
        if k == len(prizes):
            return util_from_score(sd)
        p1s = tuple(sorted(p1)); p2s = tuple(sorted(p2))
        I2 = key_p2(p1s, p2s, k)
        A1 = list(p1s); A2 = list(p2s)
        sigma2 = avg2.get(I2, {b: 1.0/len(A2) for b in A2})
        prize = prizes[k]
        best = -1e9
        for a in A1:
            ev = 0.0
            for b, pb in sigma2.items():
                new_p1 = list(p1s); new_p1.remove(a)
                new_p2 = list(p2s); new_p2.remove(b)
                sd2 = sd + round_delta(a, b, prize)
                ev += pb * BR(tuple(new_p1), tuple(new_p2), k+1, sd2)
            best = max(best, ev)
        return best

    N = len(prizes)
    return BR(tuple(range(1, N+1)), tuple(range(1, N+1)), 0, 0.0)


def best_response_value_vs_p1(prizes: List[int], avg1: Dict) -> float:
    """P2's best response (i.e., - P1's worst case) vs fixed P1 policy avg1."""
    @lru_cache(maxsize=None)
    def BR2(p1: Tuple[int, ...], p2: Tuple[int, ...], k: int, sd: float) -> float:
        if k == len(prizes):
            return -util_from_score(sd)  # P2 utility
        p1s = tuple(sorted(p1)); p2s = tuple(sorted(p2))
        I1 = key_p1(p1s, p2s, k)
        A1 = list(p1s); A2 = list(p2s)
        sigma1 = avg1.get(I1, {a: 1.0/len(A1) for a in A1})
        prize = prizes[k]
        best = -1e9
        for b in A2:
            ev = 0.0
            for a, pa in sigma1.items():
                new_p1 = list(p1s); new_p1.remove(a)
                new_p2 = list(p2s); new_p2.remove(b)
                sd2 = sd + round_delta(a, b, prize)
                ev += pa * BR2(tuple(new_p1), tuple(new_p2), k+1, sd2)
            best = max(best, ev)
        return best

    N = len(prizes)
    return BR2(tuple(range(1, N+1)), tuple(range(1, N+1)), 0, 0.0)


def current_policies_from_regrets(regret1: Dict, regret2: Dict):
    avg1 = {}
    for I, reg in regret1.items():
        acts = list(reg.keys())
        if acts:
            avg1[I] = regret_matching(reg, acts)
    avg2 = {}
    for I, reg in regret2.items():
        acts = list(reg.keys())
        if acts:
            avg2[I] = regret_matching(reg, acts)
    return avg1, avg2


# -------------------- Experiment runner --------------------

@dataclass
class ResultRow:
    iter: int
    root_p1_1: float
    root_p1_2: float
    root_p1_3: float
    EV_avg: float
    Exploit_avg: float
    Exploit_last: float


def run_experiment(N: int, iters: int, checkpoints: List[int], use_cfr_plus: bool,
                   avg_delay_frac: float) -> List[ResultRow]:
    prizes = list(range(1, N+1))
    if use_cfr_plus:
        solver = CFRPlus(prizes, avg_delay_frac=avg_delay_frac, total_iters=iters)
    else:
        solver = CFR(prizes)

    rows: List[ResultRow] = []
    root_I = key_p1(tuple(range(1,N+1)), tuple(range(1,N+1)), 0)

    for t in range(1, iters+1):
        if isinstance(solver, CFRPlus):
            solver.t = t
        solver.cfr(tuple(range(1,N+1)), tuple(range(1,N+1)), 0, 0.0, 1.0, 1.0)
        # For CFR+, accumulate average strategy after delay
        if isinstance(solver, CFRPlus) and t > solver.avg_start:
            solver.accumulate_average_strategy(tuple(range(1,N+1)), tuple(range(1,N+1)), 0, 1.0, 1.0)

        if t in checkpoints:
            avg1, avg2 = solver.average_strategies()
            cur1, cur2 = current_policies_from_regrets(solver.regret1, solver.regret2)

            root_sigma = avg1.get(root_I, {})
            p1 = root_sigma.get(1, 0.0)
            p2 = root_sigma.get(2, 0.0)
            p3 = root_sigma.get(3, 0.0) if N >= 3 else 0.0

            v_avg = eval_policy_value(prizes, avg1, avg2)
            br1_avg = best_response_value_vs_p2(prizes, avg2)
            br2_avg = best_response_value_vs_p1(prizes, avg1)
            expl_avg = (br1_avg - v_avg) + (br2_avg - (-v_avg))

            v_cur = eval_policy_value(prizes, cur1, cur2)
            br1_cur = best_response_value_vs_p2(prizes, cur2)
            br2_cur = best_response_value_vs_p1(prizes, cur1)
            expl_cur = (br1_cur - v_cur) + (br2_cur - (-v_cur))

            rows.append(ResultRow(t, p1, p2, p3, v_avg, expl_avg, expl_cur))
    return rows


def maybe_plot(rows_cfr, rows_cfrp, title):
    if not HAS_MPL:
        print("(matplotlib not installed; skipping plots)")
        return
    import matplotlib.pyplot as plt

    it_cfr = [r.iter for r in rows_cfr]
    ex_cfr = [r.Exploit_avg for r in rows_cfr]
    it_p  = [r.iter for r in rows_cfrp]
    ex_p  = [r.Exploit_avg for r in rows_cfrp]

    plt.figure()
    plt.plot(it_cfr, ex_cfr, marker='o', label='CFR (avg)')
    plt.plot(it_p, ex_p, marker='o', label='CFR+ (avg)')
    plt.xlabel("Iterations")
    plt.ylabel("Exploitability (sum of BR gaps)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="CFR / CFR+ for Goofspiel (binary terminal payoff)")
    ap.add_argument("--N", type=int, default=3, help="Number of cards (default: 3)")
    ap.add_argument("--iters", type=int, default=1000, help="Iterations (default: 1000)")
    ap.add_argument("--checkpoints", type=str, default="1,2,5,10,20,50,100,200,500,1000",
                    help="Comma-separated iteration checkpoints to report")
    ap.add_argument("--avg-delay-frac", type=float, default=0.5, help="CFR+ averaging delay fraction (default: 0.5)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed (unused here, reserved)")
    ap.add_argument("--no-plot", action="store_true", help="Disable matplotlib plots")
    args = ap.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    N = args.N
    iters = args.iters

    # Run both solvers
    rows_cfr  = run_experiment(N, iters, checkpoints, use_cfr_plus=False, avg_delay_frac=0.0)
    rows_cfrp = run_experiment(N, iters, checkpoints, use_cfr_plus=True,  avg_delay_frac=args.avg_delay_frac)

    # Print a compact table
    print(f"\nCFR vs CFR+ on Goofspiel N={N}, iters={iters}")
    hdr = ("iter", "root1_CFR", "root2_CFR", "root3_CFR", "EV_CFR",
           "Exploit_avg_CFR", "Exploit_last_CFR",
           "root1_CFR+", "root2_CFR+", "root3_CFR+", "EV_CFR+",
           "Exploit_avg_CFR+", "Exploit_last_CFR+")
    print("\t".join(hdr))
    # Build index for CFR+ rows
    rows_p_by_iter = {r.iter: r for r in rows_cfrp}
    for r in rows_cfr:
        rp = rows_p_by_iter.get(r.iter, None)
        if rp is None:
            continue
        print(f"{r.iter}\t"
              f"{r.root_p1_1:.3f}\t{r.root_p1_2:.3f}\t{r.root_p1_3:.3f}\t{r.EV_avg:+.3f}\t"
              f"{r.Exploit_avg:.3f}\t{r.Exploit_last:.3f}\t"
              f"{rp.root_p1_1:.3f}\t{rp.root_p1_2:.3f}\t{rp.root_p1_3:.3f}\t{rp.EV_avg:+.3f}\t"
              f"{rp.Exploit_avg:.3f}\t{rp.Exploit_last:.3f}")

    if not args.no_plot:
        maybe_plot(rows_cfr, rows_cfrp, f"CFR vs CFR+ (N={N})")


if __name__ == "__main__":
    main()