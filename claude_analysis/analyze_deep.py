"""
Deep analysis of state space reductions
"""
from solver import calculateEV
from utils import full, compress_cards
from collections import defaultdict
from itertools import permutations
import math

def count_theoretical_states(n):
    """Count the theoretical number of unique states"""
    # After k rounds:
    # - Each player has (n-k) cards left (from an initial set of n)
    # - Remaining cards could be any subset of {1..n}
    # - pointDiff can range from -sum(1..n) to +sum(1..n)
    # - prizes is any subset of size (n-k)
    
    # Number of ways to choose which k cards each player has played
    # and which k prizes have been awarded
    total = 0
    max_prize = n * (n + 1) // 2
    
    for k in range(n + 1):  # rounds played
        cards_left = n - k
        # Each player's remaining cards: C(n, cards_left) choices each
        card_combinations = math.comb(n, cards_left) ** 2
        # Prize combinations: C(n, cards_left)
        prize_combinations = math.comb(n, cards_left)
        # Point diff range: roughly -max_prize to +max_prize
        # But with symmetry, only need 0 to max_prize
        
        print(f"Round {k}: {card_combinations} card pairs × {prize_combinations} prize sets")
        total += card_combinations * prize_combinations
    
    return total

print("Theoretical state counts (before pointDiff):")
for n in range(1, 8):
    total = count_theoretical_states(n)
    print(f"  n={n}: {total:,} base states")

print("\n" + "="*60)
print("Key insight: The prizeIndex parameter is REDUNDANT!")
print("="*60)
print("""
When we call: calculateEV(cards, cards, diff, prizes, prizeIndex, "v")

We then loop over ALL remaining prizes and average them:
    for k in range(cardsLeft - 1):
        ev += calculateEV(..., newPrizes, k, "v")
    ev /= cardsLeft - 1

This means we're computing the EV for a UNIFORMLY RANDOM prize ordering.

The prizeIndex parameter just tells us which prize to use THIS round,
but we average over all possibilities for subsequent rounds.

OPTIMIZATION: Since we average over all k, the prizeIndex doesn't actually
add any information - we can precompute the average once!
""")

print("\nCurrent implementation issue:")
print("  - calculateEV(A, B, diff, (1,2,3), 0, 'v') and")
print("  - calculateEV(A, B, diff, (1,2,3), 1, 'v') and")  
print("  - calculateEV(A, B, diff, (1,2,3), 2, 'v')")
print("  are cached SEPARATELY but give the SAME result (the average)!")

print("\nVerifying this claim...")
calculateEV.cache_clear()
cardsA = (1, 2, 3)
cardsB = (1, 2, 3)
prizes = (1, 2, 3)
pointDiff = 2

for idx in range(3):
    ev = calculateEV(cardsA, cardsB, pointDiff, prizes, idx, "v")
    print(f"  prizeIndex={idx}: EV = {ev}")
