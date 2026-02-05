"""
MAJOR OPTIMIZATION STRATEGIES
=============================

Given the profiling results, here are the most impactful optimizations:

1. REDUCE CACHE KEY OVERHEAD (saves ~0.4s / 26% of time)
   - Current: _make_key creates a _HashedSeq for every lookup
   - Better: Use tuples directly since all args are already hashable
   - Even better: Pre-compute a single integer hash

2. REDUCE NUMBER OF STATES (saves proportional time)
   - Normalize prizes (sort them) - but we proved this matters!
   - Actually, we can RESTRUCTURE to avoid prizeIndex in cache key

3. USE NUMBA JIT (potentially 10-100x speedup on hot loops)
   - The matrix computation loop is prime for JIT

4. MEMOIZE AT MATRIX LEVEL (new approach)
   - The game VALUE only depends on the payoff MATRIX
   - If two states produce identical matrices, they have same value
   - Matrix is determined by (cardsA, cardsB, pointDiff) + prizes

Let me implement a fast prototype with inlined caching:
"""

import numpy as np
from typing import Union
from functools import lru_cache
import time

# Global cache using tuple keys directly
_ev_cache = {}
_cache_hits = 0
_cache_misses = 0

def cmp(a, b):
    return (a > b) - (a < b)

def guaranteed_fast(cardsA, cardsB, pointDiff, prizes):
    """Optimized guaranteed win check"""
    cardsLeft = len(prizes)
    if cardsLeft == 0:
        return cmp(pointDiff, 0)
    
    # Quick bounds check
    max_remaining = sum(prizes)
    if pointDiff > max_remaining:
        return 1
    if pointDiff < -max_remaining:
        return -1
    
    sorted_prizes = sorted(prizes, reverse=True)
    
    # Count guaranteed wins for A
    max_b = max(cardsB)
    guaranteeA = sum(1 for card in cardsA if card > max_b)
    
    # Count guaranteed wins for B  
    max_a = max(cardsA)
    guaranteeB = sum(1 for card in cardsB if card > max_a)
    
    # Calculate threshold
    guarantee_pts = [sum(sorted_prizes[:i]) - sum(sorted_prizes[i:]) for i in range(cardsLeft + 1)]
    
    if (guarantee_pts[guaranteeA] + pointDiff) > 0:
        return 1
    if (pointDiff - guarantee_pts[guaranteeB]) < 0:
        return -1
    return 0

def compress_cards_fast(cardsA, cardsB):
    """Compress card values to remove gaps"""
    all_values = sorted(set(cardsA) | set(cardsB))
    value_map = {old_val: new_val for new_val, old_val in enumerate(all_values, 1)}
    return tuple(value_map[c] for c in cardsA), tuple(value_map[c] for c in cardsB)

# Simple LP value calculation for small matrices
def solve_game_value_fast(matrix):
    """Fast game value for small matrices"""
    n, m = matrix.shape
    
    # Saddle point check
    row_mins = matrix.min(axis=1)
    col_maxs = matrix.max(axis=0)
    maximin = row_mins.max()
    minimax = col_maxs.min()
    
    if abs(maximin - minimax) < 1e-10:
        return maximin
    
    # For 2x2, use analytical formula
    if n == 2 and m == 2:
        a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        det = a - b - c + d
        if abs(det) > 1e-10:
            return (a * d - b * c) / det
    
    # Fallback to scipy
    from scipy.optimize import linprog
    c_vec = np.zeros(n + 1)
    c_vec[-1] = -1
    
    A_ub = np.column_stack([-matrix.T, np.ones(m)])
    b_ub = np.zeros(m)
    A_eq = np.array([[1]*n + [0]])
    b_eq = np.array([1])
    bounds = [(0, None)]*n + [(None, None)]
    
    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return -res.fun if res.success else 0.0

def calculateEV_fast(cardsA, cardsB, pointDiff, prizes, prizeIndex):
    """Optimized EV calculation with minimal overhead"""
    global _cache_hits, _cache_misses
    
    # Base case
    if len(cardsA) == 1:
        return cmp(pointDiff + cmp(cardsA[0], cardsB[0]) * prizes[0], 0)
    
    # Symmetry: swap if pointDiff < 0
    if pointDiff < 0:
        return -calculateEV_fast(cardsB, cardsA, -pointDiff, prizes, prizeIndex)
    
    # Cache lookup with tuple key directly
    key = (cardsA, cardsB, pointDiff, prizes, prizeIndex)
    if key in _ev_cache:
        _cache_hits += 1
        return _ev_cache[key]
    _cache_misses += 1
    
    # Check guaranteed win
    result = guaranteed_fast(cardsA, cardsB, pointDiff, prizes)
    if result != 0:
        _ev_cache[key] = result
        return result
    
    cardsLeft = len(cardsA)
    matrix = np.zeros((cardsLeft, cardsLeft))
    
    # Build payoff matrix
    for i in range(cardsLeft):
        for j in range(cardsLeft):
            # Symmetry optimization for symmetric games
            if cardsA == cardsB and pointDiff == 0:
                if i == j:
                    matrix[i, j] = 0
                    continue
                elif i > j:
                    matrix[i, j] = -matrix[j, i]
                    continue
            
            newA = cardsA[:i] + cardsA[i+1:]
            newB = cardsB[:j] + cardsB[j+1:]
            newA, newB = compress_cards_fast(newA, newB)
            newDiff = pointDiff + cmp(cardsA[i], cardsB[j]) * prizes[prizeIndex]
            newPrizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
            
            ev = 0.0
            for k in range(cardsLeft - 1):
                ev += calculateEV_fast(newA, newB, newDiff, newPrizes, k)
            ev /= (cardsLeft - 1)
            matrix[i, j] = ev
    
    v = solve_game_value_fast(matrix)
    _ev_cache[key] = v
    return v

def full(n):
    return tuple(range(1, n + 1))

# Test
if __name__ == "__main__":
    for n in range(1, 7):
        _ev_cache.clear()
        _cache_hits = _cache_misses = 0
        
        start = time.time()
        ev = calculateEV_fast(full(n), full(n), 0, full(n), n - 1)
        elapsed = time.time() - start
        
        print(f"full({n}): EV={ev:.6f}, time={elapsed:.3f}s, cache_size={len(_ev_cache)}, hits={_cache_hits}, misses={_cache_misses}")
