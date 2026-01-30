from __future__ import annotations

from typing import List, Tuple

from common import canonicalize, cmp, guaranteed, list_cards, load_evc, popcount, remove_card

N = 6
CACHE_PATH = "full6.evc"


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


def expand_state(A: int, B: int, P: int, diff: int, curP: int) -> List[Tuple[int, int, int, int, int]]:
    cardsA = list_cards(A)
    cardsB = list_cards(B)
    prizes = list_cards(P)
    next_states = []
    for cardA in cardsA:
        newA = remove_card(A, cardA)
        for cardB in cardsB:
            newB = remove_card(B, cardB)
            newDiff = diff + cmp(cardA, cardB) * curP
            for nextPrize in prizes:
                newP = remove_card(P, nextPrize)
                next_states.append((newA, newB, newP, newDiff, nextPrize))
    return next_states


def main() -> None:
    cache = load_evc(CACHE_PATH)

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
        stack.extend(expand_state(A, B, P, diff, curP))

    print("Total raw states:", total)
    print("Missing states:", missing)


if __name__ == "__main__":
    main()
