"""
FUNDAMENTAL RESTRUCTURING

The key insight is that we're computing:
  EV(cardsA, cardsB, pointDiff, prizes) = (1/n) * sum over k of EV_k

where EV_k is the EV when prize[k] is revealed first.

Current approach: cache EV_k for each (cardsA, cardsB, pointDiff, prizes, k)
Better approach: cache the AVERAGE EV directly, with no prizeIndex

The challenge: the matrix computation uses a SPECIFIC prize value (prizes[prizeIndex])
to compute pointDiff changes.

NEW IDEA: Reformulate to remove prizeIndex entirely

Instead of:
  matrix[i][j] = avg over k of: EV(newA, newB, newDiff_k, newPrizes_k)
  where newDiff_k = pointDiff + cmp(A[i], B[j]) * prizes[k]
  
We can factor this differently! The EV after one round depends on:
1. The new card sets (newA, newB)
2. The new pointDiff (depends on which prize was revealed)
3. The new prize set (same regardless of which prize was revealed - just one fewer!)

Since prizes are revealed UNIFORMLY at random, we can compute:
  matrix[i][j] = (1/n) * sum_k EV(newA, newB, pointDiff + cmp*prizes[k], prizes_without_k)

But here's the key: prizes_without_k is a DIFFERENT set for each k!

Wait, no. Let me think again...

Actually the prizes at the NEXT level depend on which prize was revealed THIS level.
So we can't simply drop prizeIndex.

HOWEVER: We can use a different state representation.

Instead of tracking which SPECIFIC prizes remain, track:
- The SUM of remaining prizes
- The SET of remaining prizes (as a sorted tuple)

Since the game outcome only depends on the FINAL score difference,
and we average over all orderings, we might be able to collapse states.

Let me verify: two states with same (cardsA, cardsB, pointDiff, sorted(prizes))
should have the same EV!
"""

from solver import calculateEV
from utils import full
from collections import defaultdict

calculateEV.cache_clear()
ev = calculateEV(full(4), full(4), 0, full(4), 3, "v")

# Group by (cardsA, cardsB, pointDiff, sorted_prizes, returnType)
collapsed = defaultdict(list)
for key, value in calculateEV.cache.items():
    cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
    canonical_key = (cardsA, cardsB, pointDiff, tuple(sorted(prizes)), returnType)
    collapsed[canonical_key].append((prizes, prizeIndex, value))

print("Checking if states with same canonical key have same EV...")
inconsistent = 0
consistent = 0
for canonical_key, entries in collapsed.items():
    evs = [e[2] for e in entries]
    if max(evs) - min(evs) > 1e-10:
        inconsistent += 1
        print(f"INCONSISTENT: {canonical_key}")
        for e in entries[:5]:
            print(f"  prizes={e[0]}, idx={e[1]}, ev={e[2]}")
    else:
        consistent += 1

print(f"\nConsistent: {consistent}, Inconsistent: {inconsistent}")
if inconsistent == 0:
    print("SUCCESS! Can eliminate prizeIndex and sort prizes!")
