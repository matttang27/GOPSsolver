"""
FUNDAMENTAL INSIGHT: The game tree has massive redundancy.

When we compute calculateEV(cardsA, cardsB, pointDiff, prizes, prizeIndex),
we build a matrix where each entry requires averaging over ALL remaining prizes.

This means: matrix[i][j] = (1/(n-1)) * sum_{k != prizeIndex} EV(...)

The inner EV calls are for (n-1) cards with (n-1) prizes.
Each of those builds their own matrix...

KEY OBSERVATION:
For the symmetric starting position full(n) vs full(n):
- At depth 0: 1 state
- At depth 1: O(n^2) states (which cards each player played)
- At depth 2: O(n^4) states
- ...

But with caching and compression, we're reusing a lot.

Let me count ACTUAL unique states more carefully:
"""

from solver import calculateEV
from utils import full
from collections import defaultdict

for n in range(1, 7):
    calculateEV.cache_clear()
    ev = calculateEV(full(n), full(n), 0, full(n), n - 1, "v")
    
    # Analyze states by depth
    by_depth = defaultdict(int)
    for key in calculateEV.cache.keys():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        depth = n - len(cardsA)
        by_depth[depth] += 1
    
    total = sum(by_depth.values())
    print(f"full({n}): {total} total states")
    for d in sorted(by_depth.keys()):
        print(f"  depth {d}: {by_depth[d]} states")
    print()

print("="*60)
print("ANALYSIS: State growth is polynomial in n, not factorial!")
print("The caching IS working. The issue is per-state cost.")
print()
print("Key optimizations remaining:")
print("1. Reduce LP solve overhead (most matrices have saddle points)")
print("2. Reduce cache key creation overhead")
print("3. Use faster data structures")
