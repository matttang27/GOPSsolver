"""
CRITICAL OPTIMIZATION: Eliminate redundant prize averaging!

Current code does:
  for i in range(n):
    for j in range(n):
      ev = 0
      for k in range(n-1):  # Average over prizes for NEXT round
        ev += calculateEV(newA, newB, newDiff, newPrizes, k)
      ev /= (n-1)
      matrix[i,j] = ev

This means for each matrix entry, we average over (n-1) prize positions.
Total recursive calls per matrix: n^2 * (n-1)

But here's the key: the average is over the SAME set of (newPrizes, newA, newB, newDiff)!
We're computing the same average multiple times when the same (newA, newB, newDiff, newPrizes) appears!

SOLUTION: Create a helper function that computes the AVERAGE EV for a position:
  avgEV(cardsA, cardsB, pointDiff, prizes) = (1/n) * sum_k calculateEV(..., k)

And cache THAT instead!
"""

import numpy as np
from functools import lru_cache
import time

def cmp(a, b):
    return (a > b) - (a < b)

@lru_cache(maxsize=None)
def compress_cards(cardsA, cardsB):
    all_vals = sorted(set(cardsA) | set(cardsB))
    mapping = {v: i+1 for i, v in enumerate(all_vals)}
    return tuple(mapping[c] for c in cardsA), tuple(mapping[c] for c in cardsB)

@lru_cache(maxsize=None)
def guaranteed(cardsA, cardsB, pointDiff, prizes):
    if not prizes:
        return cmp(pointDiff, 0)
    
    cardsLeft = len(prizes)
    sorted_prizes = sorted(prizes, reverse=True)
    
    guarantee = []
    s = 0
    for i in range(cardsLeft + 1):
        guarantee.append(s - (sum(sorted_prizes) - s))
        if i < cardsLeft:
            s += sorted_prizes[i]
    
    max_b = cardsB[-1] if cardsB else 0
    max_a = cardsA[-1] if cardsA else 0
    
    guaranteeA = sum(1 for c in cardsA if c > max_b)
    guaranteeB = sum(1 for c in cardsB if c > max_a)
    
    if guarantee[guaranteeA] + pointDiff > 0:
        return 1
    if pointDiff - guarantee[guaranteeB] < 0:
        return -1
    return 0

@lru_cache(maxsize=None)
def solve_matrix_value_cached(matrix_tuple):
    """Solve zero-sum game, cached by matrix content"""
    n = len(matrix_tuple)
    matrix = np.array(matrix_tuple)
    
    row_mins = np.min(matrix, axis=1)
    col_maxs = np.max(matrix, axis=0)
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)
    
    if maximin >= minimax - 1e-10:
        return float(maximin)
    
    # Use OR-Tools
    from ortools.linear_solver import pywraplp
    solver = pywraplp.Solver.CreateSolver('GLOP')
    p = [solver.NumVar(0, solver.infinity(), f'p_{i}') for i in range(n)]
    v = solver.NumVar(-solver.infinity(), solver.infinity(), 'v')
    
    for j in range(n):
        constraint = solver.Constraint(0, solver.infinity())
        for i in range(n):
            constraint.SetCoefficient(p[i], matrix[i, j])
        constraint.SetCoefficient(v, -1)
    
    prob_constraint = solver.Constraint(1, 1)
    for i in range(n):
        prob_constraint.SetCoefficient(p[i], 1)
    
    objective = solver.Objective()
    objective.SetCoefficient(v, 1)
    objective.SetMaximization()
    
    solver.Solve()
    return v.solution_value()

@lru_cache(maxsize=None)
def avgEV(cardsA, cardsB, pointDiff, prizes):
    """
    Compute average EV over all prize orderings.
    This is the key innovation - cache at this level!
    """
    n = len(cardsA)
    
    # Base case: 1 card each
    if n == 1:
        return cmp(pointDiff + cmp(cardsA[0], cardsB[0]) * prizes[0], 0)
    
    # Symmetry: always have pointDiff >= 0
    if pointDiff < 0:
        return -avgEV(cardsB, cardsA, -pointDiff, prizes)
    
    # Check guaranteed
    g = guaranteed(cardsA, cardsB, pointDiff, prizes)
    if g != 0:
        return g
    
    # Average over which prize is revealed THIS round
    total_ev = 0.0
    for prizeIndex in range(n):
        prize_val = prizes[prizeIndex]
        new_prizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
        
        # Build matrix for this prize
        matrix = np.zeros((n, n))
        for i in range(n):
            newA_base = cardsA[:i] + cardsA[i+1:]
            for j in range(n):
                if cardsA == cardsB and pointDiff == 0:
                    if i == j:
                        matrix[i, j] = 0
                        continue
                    elif i > j:
                        matrix[i, j] = -matrix[j, i]
                        continue
                
                newB = cardsB[:j] + cardsB[j+1:]
                newA, newB = compress_cards(newA_base, newB)
                newDiff = pointDiff + cmp(cardsA[i], cardsB[j]) * prize_val
                
                # Recursive call to avgEV (not individual prizeIndex)
                matrix[i, j] = avgEV(newA, newB, newDiff, new_prizes)
        
        matrix_tuple = tuple(tuple(row) for row in matrix)
        v = solve_matrix_value_cached(matrix_tuple)
        total_ev += v
    
    return total_ev / n

def full(n):
    return tuple(range(1, n + 1))

if __name__ == "__main__":
    for n in range(1, 8):
        avgEV.cache_clear()
        compress_cards.cache_clear()
        guaranteed.cache_clear()
        solve_matrix_value_cached.cache_clear()
        
        start = time.time()
        ev = avgEV(full(n), full(n), 0, full(n))
        elapsed = time.time() - start
        
        cache_size = avgEV.cache_info().currsize
        print(f"full({n}): EV={ev:.10f}, time={elapsed:.3f}s, avgEV_cache={cache_size}")
