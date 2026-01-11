"""
Optimized solver with minimal overhead.
Key changes:
1. Use dictionary with tuple keys directly (no _HashedSeq)
2. Inline saddle point check
3. Avoid numpy overhead where possible
"""

import numpy as np
from scipy.optimize import linprog
import time

# Simple dict cache
_cache = {}

def cmp(a, b):
    return (a > b) - (a < b)

def guaranteed(cardsA, cardsB, pointDiff, prizes):
    """Check for guaranteed win/loss"""
    if not prizes:
        return cmp(pointDiff, 0)
    
    cardsLeft = len(prizes)
    sorted_prizes = sorted(prizes, reverse=True)
    
    # Precompute guarantee thresholds
    guarantee = []
    s = 0
    for i in range(cardsLeft + 1):
        guarantee.append(s - (sum(sorted_prizes) - s))
        if i < cardsLeft:
            s += sorted_prizes[i]
    
    # Count guaranteed wins
    max_b = cardsB[-1] if cardsB else 0
    max_a = cardsA[-1] if cardsA else 0
    
    guaranteeA = sum(1 for c in cardsA if c > max_b)
    guaranteeB = sum(1 for c in cardsB if c > max_a)
    
    if guarantee[guaranteeA] + pointDiff > 0:
        return 1
    if pointDiff - guarantee[guaranteeB] < 0:
        return -1
    return 0

def compress_cards(cardsA, cardsB):
    """Compress to 1..k"""
    all_vals = sorted(set(cardsA) | set(cardsB))
    mapping = {v: i+1 for i, v in enumerate(all_vals)}
    return tuple(mapping[c] for c in cardsA), tuple(mapping[c] for c in cardsB)

def solve_matrix_value(matrix):
    """Solve zero-sum game for value only"""
    n, m = matrix.shape
    
    # Saddle point check
    row_mins = np.min(matrix, axis=1)
    col_maxs = np.max(matrix, axis=0)
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)
    
    if maximin >= minimax - 1e-10:
        return maximin
    
    # LP solve
    c = np.zeros(n + 1)
    c[-1] = -1
    
    A_ub = np.column_stack([-matrix.T, np.ones(m)])
    b_ub = np.zeros(m)
    A_eq = np.array([[1]*n + [0]])
    b_eq = [1]
    bounds = [(0, None)]*n + [(None, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return -res.fun if res.success else 0.0

def calculateEV_v2(cardsA, cardsB, pointDiff, prizes, prizeIndex):
    """Optimized calculation"""
    # Base case
    n = len(cardsA)
    if n == 1:
        return cmp(pointDiff + cmp(cardsA[0], cardsB[0]) * prizes[0], 0)
    
    # Symmetry exploit
    if pointDiff < 0:
        return -calculateEV_v2(cardsB, cardsA, -pointDiff, prizes, prizeIndex)
    
    # Cache check
    key = (cardsA, cardsB, pointDiff, prizes, prizeIndex)
    if key in _cache:
        return _cache[key]
    
    # Guaranteed win check
    g = guaranteed(cardsA, cardsB, pointDiff, prizes)
    if g != 0:
        _cache[key] = g
        return g
    
    # Build matrix
    matrix = np.empty((n, n))
    prize_val = prizes[prizeIndex]
    new_prizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
    
    for i in range(n):
        newA_base = cardsA[:i] + cardsA[i+1:]
        card_i = cardsA[i]
        
        for j in range(n):
            # Symmetry for symmetric positions
            if cardsA == cardsB and pointDiff == 0:
                if i == j:
                    matrix[i, j] = 0
                    continue
                elif i > j:
                    matrix[i, j] = -matrix[j, i]
                    continue
            
            newB = cardsB[:j] + cardsB[j+1:]
            newA, newB = compress_cards(newA_base, newB)
            newDiff = pointDiff + cmp(card_i, cardsB[j]) * prize_val
            
            # Average over remaining prizes
            ev = 0.0
            for k in range(n - 1):
                ev += calculateEV_v2(newA, newB, newDiff, new_prizes, k)
            matrix[i, j] = ev / (n - 1)
    
    v = solve_matrix_value(matrix)
    _cache[key] = v
    return v

def full(n):
    return tuple(range(1, n + 1))

if __name__ == "__main__":
    # Warm up scipy
    solve_matrix_value(np.array([[0, 1], [-1, 0]]))
    
    for n in range(1, 8):
        _cache.clear()
        start = time.time()
        ev = calculateEV_v2(full(n), full(n), 0, full(n), n - 1)
        elapsed = time.time() - start
        print(f"full({n}): EV={ev:.10f}, time={elapsed:.3f}s, cache={len(_cache)}")
