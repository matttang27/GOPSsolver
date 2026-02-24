import random
import sys
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
reports_path = str(REPORTS_DIR)
if reports_path not in sys.path:
    sys.path.insert(0, reports_path)

import numpy as np

from common import State, build_matrix, list_cards, list_to_mask
from linprog import findBestStrategy

ActionFn = Callable[[State], int]
StrategyDistributionFn = Callable[[State], tuple[List[int], np.ndarray]]
StrategyDistributionFactory = Callable[[Optional[Mapping[int, float]]], StrategyDistributionFn]
BuiltinDistributionFn = Callable[[List[int], int], tuple[List[int], np.ndarray]]


@dataclass(frozen=True)
class _StrategySpec:
    build_distribution: StrategyDistributionFactory
    requires_cache: bool = False
    compression_safe: bool = True


def sample_action(actions: List[int], probs: np.ndarray, rng: Optional[random.Random] = None) -> int:
    rng = rng or random
    r = rng.random()
    total = 0.0
    for action, p in zip(actions, probs):
        total += p
        if r <= total + 1e-15:
            return action
    return actions[-1]


def distribution_to_action_strategy(
    distribution: StrategyDistributionFn,
    *,
    rng: Optional[random.Random] = None,
) -> ActionFn:
    r = rng or random

    def _strategy(state: State) -> int:
        actions, probs = distribution(state)
        return sample_action(actions, probs, rng=r)

    return _strategy


def _make_builtin_distribution_strategy(distribution: BuiltinDistributionFn) -> StrategyDistributionFn:
    def _strategy(state: State) -> tuple[List[int], np.ndarray]:
        return distribution(list_cards(state.A), state.curP)

    return _strategy


def make_evc_ne_distribution_strategy(cache: Mapping[int, float]) -> StrategyDistributionFn:
    def _strategy(state: State) -> tuple[List[int], np.ndarray]:
        cardsA = list_cards(state.A)
        mat = build_matrix(cache, state.A, state.B, state.P, state.diff, state.curP)
        if not mat:
            return [max(cardsA)], np.array([1.0], dtype=np.float64)
        mat = np.array(mat, dtype=np.float64)
        pA, _v = findBestStrategy(mat)
        if pA is None:
            return [max(cardsA)], np.array([1.0], dtype=np.float64)
        return cardsA, pA

    return _strategy


def _uniform_distribution(cards: List[int]) -> tuple[List[int], np.ndarray]:
    if not cards:
        raise ValueError("Empty hand.")
    return cards, np.full(len(cards), 1.0 / len(cards), dtype=np.float64)


def _random_distribution(cards: List[int], _cur_p: int) -> tuple[List[int], np.ndarray]:
    return _uniform_distribution(cards)


def _highest_distribution(cards: List[int], _cur_p: int) -> tuple[List[int], np.ndarray]:
    if not cards:
        raise ValueError("Empty hand.")
    return [max(cards)], np.array([1.0], dtype=np.float64)


def _lowest_distribution(cards: List[int], _cur_p: int) -> tuple[List[int], np.ndarray]:
    if not cards:
        raise ValueError("Empty hand.")
    return [min(cards)], np.array([1.0], dtype=np.float64)


def _current_distribution(cards: List[int], cur_p: int) -> tuple[List[int], np.ndarray]:
    if cur_p in cards:
        return [cur_p], np.array([1.0], dtype=np.float64)
    return _uniform_distribution(cards)


def _exploit_current_distribution(cards: List[int], cur_p: int) -> tuple[List[int], np.ndarray]:
    target = cur_p + 1
    if target in cards:
        return [target], np.array([1.0], dtype=np.float64)
    return _uniform_distribution(cards)


def _builtin_factory(distribution: BuiltinDistributionFn) -> StrategyDistributionFactory:
    def _build(_cache: Optional[Mapping[int, float]]) -> StrategyDistributionFn:
        return _make_builtin_distribution_strategy(distribution)

    return _build


def _with_required_cache_distribution(
    builder: Callable[[Mapping[int, float]], StrategyDistributionFn],
    name: str,
) -> StrategyDistributionFactory:
    def _build(cache: Optional[Mapping[int, float]]) -> StrategyDistributionFn:
        if cache is None:
            raise ValueError(f"{name} strategy requires a cache")
        return builder(cache)

    return _build


_STRATEGIES: dict[str, _StrategySpec] = {
    "random": _StrategySpec(build_distribution=_builtin_factory(_random_distribution)),
    "highest": _StrategySpec(build_distribution=_builtin_factory(_highest_distribution)),
    "lowest": _StrategySpec(build_distribution=_builtin_factory(_lowest_distribution)),
    "current": _StrategySpec(
        build_distribution=_builtin_factory(_current_distribution),
        compression_safe=False,
    ),
    "exploit-current": _StrategySpec(
        build_distribution=_builtin_factory(_exploit_current_distribution),
        compression_safe=False,
    ),
    "evc-ne": _StrategySpec(
        build_distribution=_with_required_cache_distribution(make_evc_ne_distribution_strategy, "evc-ne"),
        requires_cache=True,
    ),
}

def strategy_choices() -> List[str]:
    return list(_STRATEGIES)


def strategy_requires_cache(name: str) -> bool:
    spec = _STRATEGIES.get(name)
    return bool(spec and spec.requires_cache)


def strategy_is_compression_safe(name: str) -> bool:
    spec = _STRATEGIES.get(name)
    return bool(spec and spec.compression_safe)


def strategy_distribution_builtin(name: str, cards: List[int], cur_p: int) -> tuple[List[int], np.ndarray]:
    if strategy_requires_cache(name):
        raise ValueError(f"Strategy does not expose a built-in distribution: {name}")
    distribution = build_strategy_distribution(name)
    state = State(A=list_to_mask(cards), B=0, P=0, diff=0, curP=cur_p)
    return distribution(state)


def build_strategy_distribution(name: str,
                                *,
                                cache: Optional[Mapping[int, float]] = None) -> StrategyDistributionFn:
    spec = _STRATEGIES.get(name)
    if spec is None:
        raise ValueError(f"Unknown strategy: {name}")
    return spec.build_distribution(cache)


def build_strategy(name: str,
                   *,
                   cache: Optional[Mapping[int, float]] = None,
                   rng: Optional[random.Random] = None) -> ActionFn:
    return distribution_to_action_strategy(build_strategy_distribution(name, cache=cache), rng=rng)


def strategy_label(name: str, seed: Optional[int]) -> str:
    if seed is None:
        return name
    if seed == 0:
        return f"{name}(seed=random)"
    return f"{name}(seed={seed})"
