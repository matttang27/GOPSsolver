"""Main GOPS solver"""

from __future__ import annotations
import time
import numpy as np
from typing import Union
from functools import lru_cache

import globals
from linprog import findBestStrategy_valueonly_cached, findBestStrategy_cached
from utils import cmp, compress_cards, guaranteed, full

@lru_cache(maxsize=None)
def calculateEV(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, 
               prizes: tuple[int, ...], prizeIndex: int, returnType: str) -> Union[int, float]:
    """Calculate the expected value for the current game state (cached version)."""
    
    if len(cardsA) == 1:
        if (returnType == "v"): 
            return cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)
        if (returnType == "p"):
            return np.array([1])
        if (returnType == "m"):
            return np.matrix([[cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)]])
    
    # If pointDiff is negative, swap players to reduce number of unique states
    if pointDiff < 0 and returnType == "v":
        return -calculateEV(cardsB, cardsA, -pointDiff, prizes, prizeIndex, "v")
    
    # Update global counters
    globals.totalCalculated += 1

    cardsLeft = len(cardsA)
    matrix = np.zeros((cardsLeft, cardsLeft))

    # Check for guaranteed win
    alreadyWon = guaranteed(cardsA, cardsB, pointDiff, prizes)
    if (alreadyWon != 0 and returnType == "v"):
        globals.guarantee += 1
        globals.caught += 1
        return alreadyWon

    # Calculate matrix
    for i in range(cardsLeft):
        for j in range(cardsLeft):
            if cardsA == cardsB and pointDiff == 0:
                if i == j:
                    matrix[i][j] = 0
                    continue
                elif i > j:
                    matrix[i][j] = -matrix[j][i]
                    continue
            
            newA = cardsA[:i] + cardsA[i+1:]
            newB = cardsB[:j] + cardsB[j+1:]
            newA, newB = compress_cards(newA, newB)
            newDiff = pointDiff + cmp(cardsA[i], cardsB[j]) * prizes[prizeIndex]
            newPrizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
            
            ev = 0.0
            for k in range(cardsLeft - 1):
                ev += calculateEV(newA, newB, newDiff, newPrizes, k, "v")
            ev /= cardsLeft - 1
            matrix[i][j] = ev

    if returnType == "m":
        return matrix

    matrix_tuple = tuple(tuple(row) for row in matrix)
    if returnType == "v":
        v = findBestStrategy_valueonly_cached(matrix_tuple)
        if (abs(v - 1) < 1e-10 or abs(v + 1) < 1e-10):
            globals.guarantee += 1
        return v
    else:
        # Only do full LP when you need probabilities
        p, v = findBestStrategy_cached(matrix_tuple)
        return p if returnType == "p" else v

if __name__ == "__main__":
    
    # Find threshold point using binary search
    if False:
        for i in range(1, 7):
            # Binary search to find threshold where EV transitions from < 1 to exactly 1
            left = 0
            right = 1000
            threshold = -1
            
            while left <= right:
                mid = (left + right) // 2
                ev_current = calculateEV(full(i), full(i), mid, full(i), i - 1, "v")
                
                if ev_current >= 1:
                    threshold = mid
                    right = mid - 1  # Search for a smaller threshold
                else:
                    left = mid + 1   # Search for a larger threshold
                    
            if threshold != -1:
                ev_at_threshold = calculateEV(full(i), full(i), threshold, full(i), i - 1, "v")
                ev_before = calculateEV(full(i), full(i), threshold - 1, full(i), i - 1, "v") if threshold > 0 else 0
                print(f"For full({i}), threshold at pointDiff = {threshold} (EV: {ev_at_threshold}), previous {threshold-1} (EV: {ev_before})")
            else:
                print(f"For full({i}), threshold is higher than 1000")
    
    if True:
        for i in range(1, 10):
            start_time = time.time()
            print(f"Calculating EV for full({i})...")
            
            # Get cache info before calculation
            cache_before = calculateEV.cache_info()
            
            ev = calculateEV(full(i), full(i), 0, full(i), i - 1, "p")
            
            # Get cache info after calculation
            cache_after = calculateEV.cache_info()
            
            end_time = time.time()
            
            # Calculate cache statistics for this iteration
            new_hits = cache_after.hits - cache_before.hits
            new_misses = cache_after.misses - cache_before.misses
            total_calls = new_hits + new_misses
            hit_rate = (new_hits / total_calls * 100) if total_calls > 0 else 0
            
            print(f"EV for full({i}) = {ev}")
            print(f"Time taken: {end_time - start_time:.3f} seconds")
            print(f"Cache hits: {new_hits}, misses: {new_misses}, hit rate: {hit_rate:.1f}%")
            print(f"Total cache size: {cache_after.currsize} entries")
            print(f"Cumulative hits: {cache_after.hits}, misses: {cache_after.misses}")
            globals.print_stats()


    if False:
        pr = cProfile.Profile()
        pr.enable()
        
        # Run the calculation you want to profile
        ev = calculateEV(full(7), full(7), 0, full(7), 6, "p")
        
        pr.disable()
        
        # Print results
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Show top 20 functions

