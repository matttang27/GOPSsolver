from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
REPORTS_DIR = ROOT / "reports"
DEFAULT_CACHE = REPORTS_DIR / "full9.evc"

ai_path = str(AI_DIR)
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

from common import build_matrix, decode_key, list_cards, load_evc, popcount
from linprog import findBestStrategy


def iter_filtered_states(
    cache: Any,
    *,
    curp_min: int,
    curp_max: int | None,
    hand_min: int,
    hand_max: int | None,
    diff_abs_max: int | None,
):
    for key in cache:
        A, B, P, diff, curP = decode_key(key)
        nA = popcount(A)
        if nA == 0 or nA != popcount(B):
            continue
        if popcount(P) != nA - 1:
            continue
        if curP < curp_min:
            continue
        if curp_max is not None and curP > curp_max:
            continue
        if nA < hand_min:
            continue
        if hand_max is not None and nA > hand_max:
            continue
        if diff_abs_max is not None and abs(diff) > diff_abs_max:
            continue
        yield int(key), A, B, P, int(diff), int(curP), nA


def reservoir_sample_states(
    states,
    sample_size: int,
    rng: random.Random,
    eligible_limit: int | None,
) -> tuple[list[tuple[int, int, int, int, int, int, int]], int]:
    sample: list[tuple[int, int, int, int, int, int, int]] = []
    seen = 0
    for item in states:
        if seen < sample_size:
            sample.append(item)
        else:
            j = rng.randint(0, seen)
            if j < sample_size:
                sample[j] = item
        seen += 1
        if eligible_limit is not None and seen >= eligible_limit:
            break
    return sample, seen


def trend_scan(
    cache: Any,
    *,
    sample_size: int,
    seed: int,
    curp_min: int,
    curp_max: int | None,
    hand_min: int,
    hand_max: int | None,
    diff_abs_max: int | None,
    prob_eps: float,
    top_k: int,
    cutoff_above: int | None,
    show_examples: int,
    eligible_limit: int | None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    filtered_states = iter_filtered_states(
        cache,
        curp_min=curp_min,
        curp_max=curp_max,
        hand_min=hand_min,
        hand_max=hand_max,
        diff_abs_max=diff_abs_max,
    )
    sample, eligible_count = reservoir_sample_states(
        filtered_states,
        sample_size,
        rng,
        eligible_limit,
    )

    solved_count = 0
    matrix_failures = 0
    lp_failures = 0

    states_by_curp: Counter[int] = Counter()
    delta_hits: dict[int, Counter[int]] = defaultdict(Counter)
    delta_mass: dict[int, Counter[int]] = defaultdict(Counter)
    pattern_counts: dict[int, Counter[tuple[int, tuple[int, ...]]]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    cutoff_hits = 0
    cutoff_total = 0

    for key, A, B, P, diff, curP, nA in sample:
        cardsA = list_cards(A)
        mat = build_matrix(cache, A, B, P, diff, curP)
        if not mat:
            matrix_failures += 1
            continue
        pA, _v = findBestStrategy(np.array(mat, dtype=np.float64))
        if pA is None:
            lp_failures += 1
            continue

        solved_count += 1
        states_by_curp[curP] += 1

        support = [(int(card), float(prob)) for card, prob in zip(cardsA, pA) if prob > prob_eps]
        support_deltas = tuple(sorted(card - curP for card, _ in support))
        pattern_counts[curP][(nA, support_deltas)] += 1

        for card, prob in support:
            delta = card - curP
            delta_hits[curP][delta] += 1
            delta_mass[curP][delta] += prob

        if cutoff_above is not None:
            cutoff_total += 1
            if any(card > curP + cutoff_above for card, _ in support):
                cutoff_hits += 1
                if len(examples) < show_examples:
                    examples.append(
                        {
                            "key": key,
                            "A": cardsA,
                            "B": list_cards(B),
                            "P": list_cards(P),
                            "diff": diff,
                            "curP": curP,
                            "violating_support": [
                                {"card": card, "prob": round(prob, 6)}
                                for card, prob in support
                                if card > curP + cutoff_above
                            ],
                        }
                    )

    by_curp: dict[str, Any] = {}
    for curP in sorted(states_by_curp):
        n_states = states_by_curp[curP]
        delta_table = []
        for delta, count in delta_hits[curP].most_common(top_k):
            avg_mass_when_present = delta_mass[curP][delta] / count
            hit_rate = count / n_states if n_states else 0.0
            delta_table.append(
                {
                    "delta": delta,
                    "hit_rate": hit_rate,
                    "avg_mass_when_present": avg_mass_when_present,
                    "state_hits": count,
                }
            )

        pattern_table = []
        for (hand_size, deltas), count in pattern_counts[curP].most_common(top_k):
            pattern_table.append(
                {
                    "hand_size": hand_size,
                    "support_deltas": list(deltas),
                    "hit_rate": count / n_states if n_states else 0.0,
                    "state_hits": count,
                }
            )

        by_curp[str(curP)] = {
            "states": n_states,
            "top_deltas": delta_table,
            "top_support_patterns": pattern_table,
        }

    return {
        "eligible_states": eligible_count,
        "sample_size": len(sample),
        "solved_states": solved_count,
        "matrix_failures": matrix_failures,
        "lp_failures": lp_failures,
        "settings": {
            "curp_min": curp_min,
            "curp_max": curp_max,
            "hand_min": hand_min,
            "hand_max": hand_max,
            "diff_abs_max": diff_abs_max,
            "prob_eps": prob_eps,
            "top_k": top_k,
            "cutoff_above": cutoff_above,
            "seed": seed,
            "eligible_limit": eligible_limit,
        },
        "cutoff": (
            {
                "threshold": cutoff_above,
                "violating_states": cutoff_hits,
                "states_checked": cutoff_total,
                "violation_rate": (cutoff_hits / cutoff_total) if cutoff_total else 0.0,
                "examples": examples,
            }
            if cutoff_above is not None
            else None
        ),
        "by_curP": by_curp,
    }


def print_summary(result: dict[str, Any], top_k: int) -> None:
    print(f"Eligible states: {result['eligible_states']}")
    print(f"Sampled states: {result['sample_size']}")
    print(f"Solved states: {result['solved_states']}")
    print(f"Matrix failures: {result['matrix_failures']}")
    print(f"LP failures: {result['lp_failures']}")

    cutoff = result.get("cutoff")
    if cutoff:
        print(
            f"Cutoff > curP+{cutoff['threshold']}: "
            f"{cutoff['violating_states']}/{cutoff['states_checked']} "
            f"({cutoff['violation_rate'] * 100:.2f}%)"
        )
        for ex in cutoff.get("examples", []):
            print(
                "  example key="
                f"{ex['key']} curP={ex['curP']} diff={ex['diff']} "
                f"violating_support={ex['violating_support']}"
            )

    by_curp = result.get("by_curP", {})
    for curp_key in sorted(by_curp, key=lambda x: int(x)):
        entry = by_curp[curp_key]
        print(f"\ncurP={curp_key} states={entry['states']}")
        print("  top support deltas (card-curP):")
        for row in entry["top_deltas"][:top_k]:
            print(
                f"    delta={row['delta']:>3d} "
                f"hit_rate={row['hit_rate']:.3f} "
                f"avg_mass={row['avg_mass_when_present']:.3f} "
                f"hits={row['state_hits']}"
            )
        print("  top support patterns (deltas):")
        for row in entry["top_support_patterns"][:top_k]:
            print(
                f"    n={row['hand_size']} deltas={row['support_deltas']} "
                f"hit_rate={row['hit_rate']:.3f} hits={row['state_hits']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine NE support trends from a GOPS EVC cache.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Path to .evc cache")
    parser.add_argument("--sample-size", type=int, default=8000, help="Number of filtered states to solve")
    parser.add_argument("--seed", type=int, default=0, help="Reservoir-sample RNG seed")
    parser.add_argument("--curp-min", type=int, default=1, help="Minimum current prize")
    parser.add_argument("--curp-max", type=int, default=None, help="Maximum current prize")
    parser.add_argument("--hand-min", type=int, default=2, help="Minimum cards per player")
    parser.add_argument("--hand-max", type=int, default=None, help="Maximum cards per player")
    parser.add_argument("--diff-abs-max", type=int, default=None, help="Optional |diff| filter")
    parser.add_argument("--eps", type=float, default=1e-6, help="Support probability threshold")
    parser.add_argument("--top-k", type=int, default=8, help="Rows per summary table")
    parser.add_argument(
        "--cutoff-above",
        type=int,
        default=3,
        help="Track states that place support on cards > curP + cutoff; disable with -1",
    )
    parser.add_argument("--show-examples", type=int, default=5, help="Counterexample rows to print")
    parser.add_argument(
        "--eligible-limit",
        type=int,
        default=None,
        help="Only scan the first N filtered states before sampling/solving (faster exploratory runs)",
    )
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    if args.curp_min < 1:
        raise ValueError("--curp-min must be >= 1")
    if args.hand_min < 1:
        raise ValueError("--hand-min must be >= 1")
    if args.show_examples < 0:
        raise ValueError("--show-examples must be >= 0")
    if args.eligible_limit is not None and args.eligible_limit < 1:
        raise ValueError("--eligible-limit must be >= 1")

    cache_path = Path(args.cache)
    if not cache_path.exists():
        alt = REPORTS_DIR / args.cache
        if alt.exists():
            cache_path = alt
        else:
            raise FileNotFoundError(f"Cache file not found: {args.cache}")

    cutoff_above = None if args.cutoff_above < 0 else int(args.cutoff_above)

    t0 = time.perf_counter()
    cache = load_evc(str(cache_path))
    t1 = time.perf_counter()
    result = trend_scan(
        cache,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        curp_min=int(args.curp_min),
        curp_max=None if args.curp_max is None else int(args.curp_max),
        hand_min=int(args.hand_min),
        hand_max=None if args.hand_max is None else int(args.hand_max),
        diff_abs_max=None if args.diff_abs_max is None else int(args.diff_abs_max),
        prob_eps=float(args.eps),
        top_k=int(args.top_k),
        cutoff_above=cutoff_above,
        show_examples=int(args.show_examples),
        eligible_limit=None if args.eligible_limit is None else int(args.eligible_limit),
    )
    t2 = time.perf_counter()

    print(f"Loaded cache in {t1 - t0:.3f}s: {cache_path}")
    print(f"Trend scan in {t2 - t1:.3f}s")
    print_summary(result, int(args.top_k))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
