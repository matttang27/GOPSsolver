import os
import sys
import random
from typing import Dict, List, Tuple

import numpy as np

REPORTS_DIR = os.path.dirname(__file__)
if REPORTS_DIR not in sys.path:
    sys.path.insert(0, REPORTS_DIR)

from common import build_matrix, cmp, get_ev, list_cards, load_evc, load_meta, remove_card

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from linprog import findBestStrategy


def sample_action(actions: List[int], probs: np.ndarray) -> int:
    r = random.random()
    total = 0.0
    for action, p in zip(actions, probs):
        total += p
        if r <= total + 1e-15:
            return action
    return actions[-1]


def format_probs(actions: List[int], probs: np.ndarray) -> str:
    parts = []
    for action, p in zip(actions, probs):
        if p > 1e-6:
            parts.append(f"{action}:{p:.3f}")
    return "{" + ", ".join(parts) + "}"


def play_game(cache: Dict[int, float], n: int, seed: int) -> None:
    rng = random.Random(seed)
    prizes = list(range(1, n + 1))
    rng.shuffle(prizes)

    A = (1 << n) - 1
    B = (1 << n) - 1
    curP = prizes[0]
    remaining = prizes[1:]
    P = 0
    for p in remaining:
        P |= 1 << (p - 1)
    diff = 0

    round_idx = 1
    while A:
        print("\nRound", round_idx)
        print("Prize:", curP, "Remaining prizes:", sorted(remaining))
        print("Current diff:", diff)

        print("Your hand:", list_cards(A))
        print("Bot hand:", list_cards(B))

        cardsA = list_cards(A)
        cardsB = list_cards(B)
        mat = build_matrix(cache, A, B, P, diff, curP)
        if not mat:
            print("Matrix build failed for current state.")
            return
        mat = np.array(mat, dtype=np.float64)
        pA, v = findBestStrategy(mat)
        if pA is None:
            print("LP failed for player strategy")
            return
        pB, vB = findBestStrategy(-mat.T)
        if pB is None:
            print("LP failed for bot strategy")
            return

        ev_per_action = []
        for i, _cardA in enumerate(cardsA):
            ev = float(np.dot(mat[i, :], pB))
            ev_per_action.append(ev)

        best_ev = max(ev_per_action)
        print("NE probs (you):", format_probs(cardsA, pA))
        print("NE probs (bot):", format_probs(cardsB, pB))
        print("EV by action vs bot mix:")
        for cardA, ev in zip(cardsA, ev_per_action):
            mark = " <==" if abs(ev - best_ev) < 1e-9 else ""
            print(f"  {cardA:2d} -> {ev:+.4f}{mark}")

        choice = None
        while choice not in cardsA:
            try:
                choice = int(input("Choose your card: ").strip())
            except ValueError:
                choice = None
        bot_choice = sample_action(cardsB, pB)

        print("Bot plays:", bot_choice)
        diff += cmp(choice, bot_choice) * curP

        A = remove_card(A, choice)
        B = remove_card(B, bot_choice)

        if not remaining:
            break
        curP = remaining[0]
        remaining = remaining[1:]
        P = remove_card(P, curP)
        round_idx += 1

    print("\nFinal diff:", diff)
    if diff > 0:
        print("You win!")
    elif diff < 0:
        print("Bot wins!")
    else:
        print("Draw.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python play.py <cache.evc> [n] [seed]")
        return
    cache_path = sys.argv[1]
    if not os.path.exists(cache_path):
        alt_path = os.path.join(REPORTS_DIR, cache_path)
        if os.path.exists(alt_path):
            cache_path = alt_path
        else:
            alt_path = os.path.abspath(os.path.join(REPORTS_DIR, "..", cache_path))
            if os.path.exists(alt_path):
                cache_path = alt_path
            else:
                print(f"Cache file not found: {sys.argv[1]}")
                return
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else 6
    seed = int(sys.argv[3]) if len(sys.argv) >= 4 else 12345
    cache = load_evc(cache_path)
    play_game(cache, n, seed)


if __name__ == "__main__":
    main()
