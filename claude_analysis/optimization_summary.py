"""
Summary of optimization opportunities for GOPS solver
=====================================================

CURRENT STATE:
- full(6): 1.8s, 14,664 cache entries
- full(7): ~15s, 121,194 cache entries  
- Scaling: ~10x per additional card

BOTTLENECKS (from profiling):
1. Linear programming (findBestStrategy): ~25% of time
2. Cache key creation (_make_key): ~20% of time
3. guaranteed() check: ~10% of time
4. NumPy operations (min, max): ~10% of time

OPTIMIZATION OPPORTUNITIES:

1. PRIZE NORMALIZATION [HIGH IMPACT]
   - Sort prizes tuple to canonical form
   - Currently (1,2,3), (2,1,3), (3,1,2) are cached separately
   - Should be the same since we average over all orderings anyway!
   - Potential reduction: factor of k! for k remaining prizes

2. ELIMINATE prizeIndex FROM CACHE KEY [HIGH IMPACT]
   - Since we ALWAYS average over all prize orderings in subsequent rounds,
   - the prizeIndex only affects THIS round's prize value
   - Could restructure: compute matrix once, then branch on prize value

3. NUMBA/CYTHON ACCELERATION [MEDIUM IMPACT]
   - The core recursion is compute-bound
   - JIT compilation could help significantly

4. PARALLEL PROCESSING [MEDIUM IMPACT]
   - Matrix entries are independent
   - Could parallelize the double loop

5. BETTER LP SOLVER [MEDIUM IMPACT]  
   - Current: OR-Tools GLOP with SciPy fallback
   - For 2x2 and 3x3 games, analytical solutions exist

6. ALPHA-BETA PRUNING [HIGH IMPACT]
   - If we've established EV >= 1 or EV <= -1, we can prune
   - The guaranteed() function does some of this

7. STATE COMPRESSION [HIGH IMPACT]
   - After compress_cards(), cards are normalized
   - Could also normalize prizes relative to cards
   - Key insight: only RELATIVE orderings matter

Let me implement the most impactful: PRIZE NORMALIZATION
"""

print("Testing prize normalization impact...")

from solver import calculateEV
from utils import full

# Current behavior
calculateEV.cache_clear()
ev1 = calculateEV(full(5), full(5), 0, full(5), 4, "v")
size1 = calculateEV.cache_info().currsize
print(f"Current: {size1} cache entries")

# What if prizes were always sorted?
# The solver calls calculateEV with newPrizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
# This maintains the ORDER of prizes
# But since we AVERAGE over all orderings, we should be able to sort them!

print("\nLet's count how many unique (cardsA, cardsB, sorted_prizes, pointDiff) there are:")

unique_states = set()
for key in calculateEV.cache.keys():
    cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
    # Normalize prizes by sorting
    sorted_prizes = tuple(sorted(prizes))
    unique_states.add((cardsA, cardsB, pointDiff, sorted_prizes, returnType))

print(f"With prize sorting: {len(unique_states)} unique states (vs {size1} current)")
print(f"Reduction factor: {size1 / len(unique_states):.1f}x")
