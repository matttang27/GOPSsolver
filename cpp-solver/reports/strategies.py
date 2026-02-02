import random
from typing import Callable, Dict, List, Optional

import numpy as np

from common import State, build_matrix, list_cards
from linprog import findBestStrategy

ActionFn = Callable[[State], int]

_REQUIRES_CACHE = {"evc-ne"}


def sample_action(actions: List[int], probs: np.ndarray, rng: Optional[random.Random] = None) -> int:
    rng = rng or random
    r = rng.random()
    total = 0.0
    for action, p in zip(actions, probs):
        total += p
        if r <= total + 1e-15:
            return action
    return actions[-1]


def make_random_strategy(rng: Optional[random.Random] = None) -> ActionFn:
    rng = rng or random

    def _strategy(state: State) -> int:
        return rng.choice(list_cards(state.A))

    return _strategy


def make_highest_strategy() -> ActionFn:
    def _strategy(state: State) -> int:
        return max(list_cards(state.A))

    return _strategy


def make_lowest_strategy() -> ActionFn:
    def _strategy(state: State) -> int:
        return min(list_cards(state.A))

    return _strategy


def make_current_strategy() -> ActionFn:
    def _strategy(state: State) -> int:
        cards = list_cards(state.A)
        if state.curP in cards:
            return state.curP
        return min(cards)

    return _strategy


def make_exploit_current_strategy() -> ActionFn:
    def _strategy(state: State) -> int:
        cards = sorted(list_cards(state.A))
        for card in cards:
            if card > state.curP:
                return card
        return cards[0]

    return _strategy


def make_evc_ne_strategy(cache: Dict[int, float], rng: Optional[random.Random] = None) -> ActionFn:
    rng = rng or random

    def _strategy(state: State) -> int:
        cardsA = list_cards(state.A)
        mat = build_matrix(cache, state.A, state.B, state.P, state.diff, state.curP)
        if not mat:
            return max(cardsA)
        mat = np.array(mat, dtype=np.float64)
        pA, _v = findBestStrategy(mat)
        if pA is None:
            return max(cardsA)
        return sample_action(cardsA, pA, rng=rng)

    return _strategy


def strategy_choices() -> List[str]:
    return ["random", "highest", "lowest", "current", "exploit-current", "evc-ne"]


def strategy_requires_cache(name: str) -> bool:
    return name in _REQUIRES_CACHE


def build_strategy(name: str,
                   *,
                   cache: Optional[Dict[int, float]] = None,
                   rng: Optional[random.Random] = None) -> ActionFn:
    if name == "random":
        return make_random_strategy(rng=rng)
    if name == "highest":
        return make_highest_strategy()
    if name == "lowest":
        return make_lowest_strategy()
    if name == "current":
        return make_current_strategy()
    if name == "exploit-current":
        return make_exploit_current_strategy()
    if name == "evc-ne":
        if cache is None:
            raise ValueError("evc-ne strategy requires a cache")
        return make_evc_ne_strategy(cache, rng=rng)
    raise ValueError(f"Unknown strategy: {name}")


def strategy_label(name: str, seed: Optional[int]) -> str:
    if seed is None:
        return name
    if seed == 0:
        return f"{name}(seed=random)"
    return f"{name}(seed={seed})"
