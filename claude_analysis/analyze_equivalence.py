"""
Key insight: The prize values can be NORMALIZED too!

If cardsA and cardsB use values 1,3,5 and prizes are (1,3,5),
we can map everything to 1,2,3.

But more importantly: the ABSOLUTE prize values don't matter for the EV!
Only their RELATIVE values matter for determining win/lose/tie.

Furthermore, for a symmetric game where cardsA == cardsB, the EV is always 0 when pointDiff=0.
This is exploited by the symmetry check in the matrix construction.

Let's look at what truly distinguishes states:
"""
from itertools import permutations
import numpy as np

def analyze_state_equivalence():
    """
    Two GOPS states are equivalent if they produce the same game tree structure.
    
    Key question: can we normalize (cardsA, cardsB, prizes) together?
    
    The answer is yes! If we relabel all values consistently, the game is the same.
    
    Example: (cardsA=(1,3), cardsB=(2,4), prizes=(1,3))
    is equivalent to (cardsA=(1,2), cardsB=(1.5,3), prizes=(1,2)) after normalization
    
    But wait - cards and prizes might have DIFFERENT value sets!
    """
    
    # Analyze: what matters is the COMPARISON between cards and prizes
    # Card-card comparisons determine round outcomes
    # Prize values determine point changes
    
    # For EV calculation, what matters:
    # 1. Which card beats which (relative ordering within each hand, and between hands)
    # 2. The sum/distribution of remaining prizes
    
    # CRITICAL INSIGHT:
    # The prizes tuple determines the points at stake
    # But since we average over all prize orderings (via the k loop),
    # only the SET of remaining prizes matters, not their order!
    
    print("Testing prize order independence...")
    
    # This means (1,2,3) and (3,1,2) and (2,3,1) should give same EV
    # for the same cards and pointDiff!
    
    from solver import calculateEV
    from utils import full
    
    # Clear cache for accurate test
    calculateEV.cache_clear()
    
    cardsA = (1, 2, 3)
    cardsB = (1, 2, 3)
    pointDiff = 5
    
    results = {}
    for perm in permutations([1, 2, 3]):
        for prizeIdx in range(3):
            ev = calculateEV(cardsA, cardsB, pointDiff, perm, prizeIdx, "v")
            key = (perm, prizeIdx)
            results[key] = ev
    
    # Check if all permutations give same average EV
    avg_by_perm = {}
    for perm in permutations([1, 2, 3]):
        evs = [results[(perm, i)] for i in range(3)]
        avg_by_perm[perm] = sum(evs) / 3
    
    print("Average EV by prize permutation:")
    for perm, avg in sorted(avg_by_perm.items()):
        print(f"  {perm}: {avg:.6f}")
    
    unique_avgs = set(round(v, 10) for v in avg_by_perm.values())
    print(f"\nUnique average values: {len(unique_avgs)}")
    
    if len(unique_avgs) == 1:
        print("CONFIRMED: Prize ordering doesn't matter! Only the set matters.")
        print("\nOptimization: sort prizes tuple before caching!")

analyze_state_equivalence()
