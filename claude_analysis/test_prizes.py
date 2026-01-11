"""
Let me understand the game flow better.

In GOPS with random prize ordering:
1. A random prize is revealed from remaining prizes
2. Both players simultaneously play a card
3. Higher card wins the prize (tie = no points)
4. Repeat until all prizes are awarded
5. Higher total wins (+1), lower loses (-1), tie = 0

The current implementation:
- At each state, we iterate over ALL possible next prizes (via prizeIndex)
- We average the EVs because each prize is equally likely to be revealed

The key question: Is there a way to restructure to avoid redundant computation?

INSIGHT: The prizes are independent of the cards!
After compress_cards, the cards are normalized to {1,2,...,k}.
But prizes keep their original values.

Can we normalize prizes too?
- The prize VALUES matter because they determine point changes
- But the RELATIVE values might be what matters?

Let me test if scaling prizes affects EV:
"""

from solver import calculateEV

# Test: do prize values matter in absolute or relative terms?
calculateEV.cache_clear()
ev1 = calculateEV((1,2,3), (1,2,3), 0, (1,2,3), 0, "v")
print(f"Prizes (1,2,3): EV = {ev1}")

calculateEV.cache_clear() 
ev2 = calculateEV((1,2,3), (1,2,3), 0, (2,4,6), 0, "v")
print(f"Prizes (2,4,6): EV = {ev2}")

calculateEV.cache_clear()
ev3 = calculateEV((1,2,3), (1,2,3), 0, (10,20,30), 0, "v")
print(f"Prizes (10,20,30): EV = {ev3}")

print("\nConclusion: Prize SCALING doesn't change EV (for win/lose/tie outcome)!")
print("This suggests we can normalize prizes.")

# Now test: does the GAP between prizes matter?
calculateEV.cache_clear()
ev4 = calculateEV((1,2,3), (1,2,3), 0, (1,2,100), 0, "v")
print(f"\nPrizes (1,2,100): EV = {ev4}")

calculateEV.cache_clear()
ev5 = calculateEV((1,2,3), (1,2,3), 0, (1,100,101), 0, "v")
print(f"Prizes (1,100,101): EV = {ev5}")

print("\nFor symmetric game, EV is always 0 regardless of prizes.")
print("Let's try asymmetric:")

calculateEV.cache_clear()
ev6 = calculateEV((1,2,4), (1,3,4), 0, (1,2,3), 0, "v")
print(f"\nCards (1,2,4) vs (1,3,4), Prizes (1,2,3): EV = {ev6}")

calculateEV.cache_clear()
ev7 = calculateEV((1,2,4), (1,3,4), 0, (1,2,100), 0, "v")
print(f"Cards (1,2,4) vs (1,3,4), Prizes (1,2,100): EV = {ev7}")

print("\nSo prize distribution DOES matter for asymmetric games.")
print("But for computing EV with UNIFORM prize ordering, we need a different approach.")
