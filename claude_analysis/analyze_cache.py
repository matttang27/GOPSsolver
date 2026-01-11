"""Analyze what's in the cache to find symmetries"""
from solver import calculateEV
from utils import full, compress_cards
from collections import defaultdict

calculateEV.cache_clear()
ev = calculateEV(full(5), full(5), 0, full(5), 4, "v")

# Analyze the cache
print(f"Cache size: {calculateEV.cache_info().currsize}")

# Group by (cardsA, cardsB) ignoring pointDiff and prizes
by_cards = defaultdict(list)
by_shape = defaultdict(int)

for key, value in calculateEV.cache.items():
    cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
    by_cards[(cardsA, cardsB)].append((pointDiff, prizes, prizeIndex, value))
    by_shape[len(cardsA)] += 1

print("\nStates by card count:")
for k in sorted(by_shape.keys()):
    print(f"  {k} cards: {by_shape[k]} states")

# See how many unique (cardsA, cardsB) pairs there are
print(f"\nUnique (cardsA, cardsB) pairs: {len(by_cards)}")

# Look at some examples
print("\nSample states with same cards but different pointDiff:")
for (cardsA, cardsB), states in list(by_cards.items())[:3]:
    if len(states) > 1:
        print(f"  {cardsA} vs {cardsB}: {len(states)} states")
        for s in states[:5]:
            print(f"    pointDiff={s[0]}, prizes={s[1]}, prizeIdx={s[2]}, EV={s[3]:.4f}")
