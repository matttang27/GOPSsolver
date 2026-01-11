"""
Analysis of potential optimizations for GOPS solver

Key observations:
1. The state space explosion comes from: (cardsA, cardsB, pointDiff, prizes, prizeIndex)
2. prizes is a permutation tracking which prize cards are left
3. prizeIndex just picks which remaining prize to reveal next

Key insight: In the *original GOPS game*, prizes are revealed in a FIXED order (1,2,3,...,N)
But in this solver, we're averaging over ALL possible prize orders.

Let's verify: when prizes is full (e.g., (1,2,3,4,5)), all orderings are equally likely,
so we loop over prizeIndex=0..4 and average.

The key question: can we reduce the state space by normalizing prizes differently?
"""

from solver import calculateEV
from utils import full
from collections import defaultdict

calculateEV.cache_clear()
ev = calculateEV(full(4), full(4), 0, full(4), 3, "v")

# Group states by number of remaining cards
by_depth = defaultdict(list)
for key, value in calculateEV.cache.items():
    cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
    by_depth[len(cardsA)].append({
        'cardsA': cardsA, 'cardsB': cardsB, 'pointDiff': pointDiff,
        'prizes': prizes, 'prizeIndex': prizeIndex, 'ev': value
    })

print("States at each depth:")
for depth in sorted(by_depth.keys(), reverse=True):
    states = by_depth[depth]
    unique_prizes = len(set(s['prizes'] for s in states))
    unique_cards = len(set((s['cardsA'], s['cardsB']) for s in states))
    unique_diff = len(set(s['pointDiff'] for s in states))
    print(f"  {depth} cards left: {len(states)} states, {unique_prizes} unique prize sets, {unique_cards} card pairs, {unique_diff} diffs")
    
# Check if prizes always match cardsA/cardsB
print("\nPrizes vs cards relationship:")
for depth in sorted(by_depth.keys(), reverse=True):
    if depth <= 3:
        for s in by_depth[depth][:3]:
            print(f"  cards={s['cardsA']} vs {s['cardsB']}, prizes={s['prizes']}, diff={s['pointDiff']}")
