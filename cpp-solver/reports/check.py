from __future__ import annotations

from typing import List, Tuple

from common import canonicalize, cmp, guaranteed, list_cards, load_evc, popcount, remove_card

N = 8
CACHE_PATH = "full8.evc"

ExpansionCache = dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int, int]]]


def is_present(cache: dict[int, float], A: int, B: int, P: int, diff: int, curP: int) -> bool:
    if (A == B and diff == 0):
        return True
    key, _sign = canonicalize(A, B, P, diff, curP)
    if key in cache:
        return True
    cardsA = list_cards(A)
    cardsB = list_cards(B)
    if not cardsA or not cardsB:
        return True
    prizes = list_cards(P)
    prizes.append(curP)
    try:
        return guaranteed(tuple(cardsA), tuple(cardsB), diff, tuple(prizes)) != 0
    except IndexError:
        return False


def expand_state(A: int,
                 B: int,
                 P: int,
                 diff: int,
                 curP: int,
                 expansion_cache: ExpansionCache) -> List[Tuple[int, int, int, int, int]]:
    key = (A, B, P, curP)
    base = expansion_cache.get(key)
    if base is None:
        cardsA = list_cards(A)
        cardsB = list_cards(B)
        prizes = list_cards(P)
        base = []
        for cardA in cardsA:
            newA = remove_card(A, cardA)
            for cardB in cardsB:
                newB = remove_card(B, cardB)
                delta = cmp(cardA, cardB) * curP
                for nextPrize in prizes:
                    newP = remove_card(P, nextPrize)
                    base.append((newA, newB, newP, delta, nextPrize))
        expansion_cache[key] = base
    next_states = []
    for newA, newB, newP, delta, nextPrize in base:
        next_states.append((newA, newB, newP, diff + delta, nextPrize))
    return next_states


def main() -> None:
    cache = load_evc(CACHE_PATH)
    expansion_cache: ExpansionCache = {}
    checked_cache: set[int] = set()

    fullN = (1 << N) - 1
    start_states = []
    for prize in range(1, N + 1):
        remaining = remove_card(fullN, prize)
        start_states.append((fullN, fullN, remaining, 0, prize))

    visited: set[Tuple[int, int, int, int, int]] = set()
    stack = start_states[:]
    missing = 0
    total = 0

    while stack:
        A, B, P, diff, curP = stack.pop()
        state = (A, B, P, diff, curP)
        if state in visited:
            continue
        visited.add(state)
        total += 1
        if not is_present(cache, A, B, P, diff, curP):
            missing += 1
        if popcount(A) <= 1 or popcount(B) <= 1:
            continue
        if popcount(P) != popcount(A) - 1:
            continue
        key, _sign = canonicalize(A, B, P, diff, curP)
        if key in checked_cache:
            continue
        checked_cache.add(key)
        stack.extend(expand_state(A, B, P, diff, curP, expansion_cache))

    print("Total raw states:", total)
    print("Missing states:", missing)


if __name__ == "__main__":
    main()
