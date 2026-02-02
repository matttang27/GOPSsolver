import argparse
import os
import sys
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

REPORTS_DIR = os.path.dirname(__file__)
if REPORTS_DIR not in sys.path:
    sys.path.insert(0, REPORTS_DIR)

from common import State, build_matrix, cmp, list_cards, load_evc, remove_card
from strategies import ActionFn, build_strategy, sample_action, strategy_choices, strategy_label, strategy_requires_cache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from linprog import findBestStrategy

ChooseFn = Callable[[State, State], Tuple[int, int]]


class AbortGame(RuntimeError):
    pass


def format_probs(actions: List[int], probs: np.ndarray) -> str:
    parts = []
    for action, p in zip(actions, probs):
        if p > 1e-6:
            parts.append(f"{action}:{p:.3f}")
    return "{" + ", ".join(parts) + "}"


def print_game_state(state: State, *, label_a: str, label_b: str) -> None:
    print("\nPrize:", state.curP, "Remaining prizes:", sorted(list_cards(state.P)))
    print("Current diff:", state.diff)
    print(f"{label_a} hand:", list_cards(state.A))
    print(f"{label_b} hand:", list_cards(state.B))


def run_game(n: int,
             seed: int,
             choose_actions: ChooseFn,
             *,
             rng: Optional[random.Random] = None) -> int:
    rng = rng or (random.Random(seed) if seed != 0 else random)
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

    while A:
        state = State(A=A, B=B, P=P, diff=diff, curP=curP)
        state_for_b = State(A=B, B=A, P=P, diff=-diff, curP=curP)
        cardsA = list_cards(A)
        cardsB = list_cards(B)
        choiceA, choiceB = choose_actions(state, state_for_b)
        if choiceA not in cardsA:
            raise ValueError(f"Strategy A returned invalid card: {choiceA}")
        if choiceB not in cardsB:
            raise ValueError(f"Strategy B returned invalid card: {choiceB}")

        diff += cmp(choiceA, choiceB) * curP

        A = remove_card(A, choiceA)
        B = remove_card(B, choiceB)

        if not remaining:
            break
        curP = remaining[0]
        remaining = remaining[1:]
        P = remove_card(P, curP)

    return diff


def make_auto_choose_actions(stratA: ActionFn,
                             stratB: ActionFn,
                             *,
                             verbose: bool) -> ChooseFn:
    def _choose(state: State, state_for_b: State) -> Tuple[int, int]:
        if verbose:
            print_game_state(state, label_a="Player A", label_b="Player B")
        choiceA = stratA(state)
        choiceB = stratB(state_for_b)
        if verbose:
            print("Player A plays:", choiceA)
            print("Player B plays:", choiceB)
        return choiceA, choiceB

    return _choose


def auto_play_game(n: int,
                   seed: int,
                   stratA: ActionFn,
                   stratB: ActionFn,
                   verbose: bool = True,
                   rng: Optional[random.Random] = None) -> int:
    choose_actions = make_auto_choose_actions(stratA, stratB, verbose=verbose)
    diff = run_game(n, seed, choose_actions, rng=rng)
    if verbose:
        print("\nFinal diff:", diff)
        if diff > 0:
            print("Player A wins!")
        elif diff < 0:
            print("Player B wins!")
        else:
            print("Draw.")
    return diff


def auto_play_random(n: int, seed: int, verbose: bool = True) -> int:
    rng = random.Random(seed) if seed != 0 else random
    strat = build_strategy("random", rng=rng)
    return auto_play_game(n, seed, strat, strat, verbose=verbose, rng=rng)


def play_game(cache: Dict[int, float], n: int, seed: int) -> None:
    def choose_actions(state: State, _state_for_b: State) -> Tuple[int, int]:
        print_game_state(state, label_a="Your", label_b="Bot")

        cardsA = list_cards(state.A)
        cardsB = list_cards(state.B)

        mat = build_matrix(cache, state.A, state.B, state.P, state.diff, state.curP)
        if not mat:
            print("Matrix build failed for current state.")
            raise AbortGame()
        mat = np.array(mat, dtype=np.float64)
        pA, _v = findBestStrategy(mat)
        if pA is None:
            print("LP failed for player strategy")
            raise AbortGame()
        pB, _vB = findBestStrategy(-mat.T)
        if pB is None:
            print("LP failed for bot strategy")
            raise AbortGame()

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
        return choice, bot_choice

    try:
        diff = run_game(n, seed, choose_actions)
    except AbortGame:
        return

    print("\nFinal diff:", diff)
    if diff > 0:
        print("You win!")
    elif diff < 0:
        print("Bot wins!")
    else:
        print("Draw.")


def resolve_cache_path(raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None
    if os.path.exists(raw_path):
        return raw_path
    alt_path = os.path.join(REPORTS_DIR, raw_path)
    if os.path.exists(alt_path):
        return alt_path
    alt_path = os.path.abspath(os.path.join(REPORTS_DIR, "..", raw_path))
    if os.path.exists(alt_path):
        return alt_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play GOPS using cached EVs or auto-play.")
    parser.add_argument("cache", nargs="?", help="Path to .evc cache (required for interactive play)")
    parser.add_argument("n", nargs="?", type=int, help="Number of cards (positional)")
    parser.add_argument("seed", nargs="?", type=int, help="RNG seed (positional, 0 for non-deterministic)")
    parser.add_argument("--n", dest="n_flag", type=int, help="Number of cards")
    parser.add_argument("--seed", dest="seed_flag", type=int, help="RNG seed (0 for non-deterministic)")
    parser.add_argument("--auto", action="store_true", help="Auto-play with random strategies (no cache needed)")
    parser.add_argument("--random", action="store_true", help="Alias for --auto")
    parser.add_argument("--count", type=int, default=1, help="Number of auto-play games to run")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Verbose output for auto-play")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Reduce output for auto-play")
    parser.set_defaults(verbose=False)
    strat_choices = strategy_choices()
    group_a = parser.add_argument_group("Strategy A")
    group_a.add_argument("--sa", default="random", choices=strat_choices, help="Strategy for player A")
    group_a.add_argument("--sa-seed", type=int, help="RNG seed for player A strategy (0 for non-deterministic)")
    group_b = parser.add_argument_group("Strategy B")
    group_b.add_argument("--sb", default="random", choices=strat_choices, help="Strategy for player B")
    group_b.add_argument("--sb-seed", type=int, help="RNG seed for player B strategy (0 for non-deterministic)")
    args = parser.parse_args()

    n = args.n_flag if args.n_flag is not None else (args.n if args.n is not None else 6)
    seed = args.seed_flag if args.seed_flag is not None else (args.seed if args.seed is not None else random.randint(0, 1000000))

    if args.auto or args.random:
        if args.count < 1:
            print("--count must be >= 1")
            return
        needs_cache = strategy_requires_cache(args.sa) or strategy_requires_cache(args.sb)
        cache = None
        if needs_cache:
            cache_path = resolve_cache_path(args.cache)
            if not cache_path:
                print("Cache file required for the selected strategy (evc-ne).")
                return
            cache = load_evc(cache_path)
        wins_a = 0
        wins_b = 0
        draws = 0
        total_diff = 0
        for i in range(args.count):
            seed_i = seed if seed == 0 else seed + i
            rng_game = random.Random(seed_i) if seed_i != 0 else random
            rng_a = rng_game
            if args.sa_seed is not None:
                rng_a = random.Random(args.sa_seed) if args.sa_seed != 0 else random
            rng_b = rng_game
            if args.sb_seed is not None:
                rng_b = random.Random(args.sb_seed) if args.sb_seed != 0 else random
            stratA = build_strategy(args.sa, cache=cache, rng=rng_a)
            stratB = build_strategy(args.sb, cache=cache, rng=rng_b)
            diff = auto_play_game(n, seed_i, stratA, stratB, verbose=args.verbose, rng=rng_game)
            total_diff += diff
            if diff > 0:
                wins_a += 1
            elif diff < 0:
                wins_b += 1
            else:
                draws += 1
        total_games = args.count
        avg_diff = total_diff / total_games
        avg_ev = (wins_a - wins_b) / total_games
        label_a = strategy_label(args.sa, args.sa_seed)
        label_b = strategy_label(args.sb, args.sb_seed)
        if total_games > 1 or not args.verbose:
            print("\nSeries summary")
            print(f"Games: {total_games}")
            print(f"Strategy A: {label_a}")
            print(f"Strategy B: {label_b}")
            print(f"A wins: {wins_a} ({wins_a / total_games * 100:.1f}%)")
            print(f"B wins: {wins_b} ({wins_b / total_games * 100:.1f}%)")
            print(f"Draws: {draws} ({draws / total_games * 100:.1f}%)")
            print(f"Average point diff (A-B): {avg_diff:+.3f}")
            print(f"Average EV (A=+1, B=-1): {avg_ev:+.3f}")
        return

    if not args.cache:
        print("Usage: python play.py <cache.evc> [n] [seed / 0 for random]")
        return

    cache_path = resolve_cache_path(args.cache)
    if not cache_path:
        print(f"Cache file not found: {args.cache}")
        return

    cache = load_evc(cache_path)
    play_game(cache, n, seed)


if __name__ == "__main__":
    main()
