import random
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
reports_path = str(REPORTS_DIR)
if reports_path not in sys.path:
    sys.path.insert(0, reports_path)

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
    r = rng or random
    return lambda state: r.choice(list_cards(state.A))

def make_highest_strategy() -> ActionFn:
    return lambda state: max(list_cards(state.A))

def make_lowest_strategy() -> ActionFn:
    return lambda state: min(list_cards(state.A))

def make_current_strategy(rng: Optional[random.Random] = None) -> ActionFn:
    r = rng or random
    return lambda state: state.curP if state.curP in list_cards(state.A) else r.choice(list_cards(state.A))

def make_exploit_current_strategy(rng: Optional[random.Random] = None) -> ActionFn:
    r = rng or random
    return lambda state: state.curP + 1 if state.curP + 1 in list_cards(state.A) else r.choice(list_cards(state.A))



def make_evc_ne_strategy(cache: Mapping[int, float], rng: Optional[random.Random] = None) -> ActionFn:
    r = rng or random

    def _strategy(state: State) -> int:
        cardsA = list_cards(state.A)
        mat = build_matrix(cache, state.A, state.B, state.P, state.diff, state.curP)
        if not mat:
            return max(cardsA)
        mat = np.array(mat, dtype=np.float64)
        pA, _v = findBestStrategy(mat)
        if pA is None:
            return max(cardsA)
        return sample_action(cardsA, pA, rng=r)

    return _strategy

def strategy_choices() -> List[str]:
    return ["random", "highest", "lowest", "current", "exploit-current", "evc-ne"]


def strategy_requires_cache(name: str) -> bool:
    return name in _REQUIRES_CACHE


def build_strategy(name: str,
                   *,
                   cache: Optional[Mapping[int, float]] = None,
                   rng: Optional[random.Random] = None) -> ActionFn:
    if name == "random":
        return make_random_strategy(rng=rng)
    if name == "highest":
        return make_highest_strategy()
    if name == "lowest":
        return make_lowest_strategy()
    if name == "current":
        return make_current_strategy(rng=rng)
    if name == "exploit-current":
        return make_exploit_current_strategy(rng=rng)
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
