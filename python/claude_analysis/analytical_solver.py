"""
The prizeIndex matters because different prize values lead to different outcomes.
But we're averaging anyway, so let's think about this differently.

INSIGHT: The function currently computes matrix[i][j] by:
1. For each (i,j) move pair
2. Loop over k (remaining prizes for next round)
3. Average the EVs

But the k loop is INSIDE the matrix computation!

We could restructure to:
1. For each prize k (THIS round's prize)
2. Compute the matrix for this prize value
3. Solve the matrix game
4. Average the game values

This doesn't reduce states but might allow better caching.

Actually, let's try a completely different approach: ANALYTICAL BOUNDS

For small subgames (1-3 cards), we can compute exact solutions analytically.
This avoids the LP overhead entirely.
"""

import numpy as np

def solve_2x2_game(matrix):
    """Analytical solution for 2x2 zero-sum game"""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    
    # Check for saddle point
    row_mins = np.min(matrix, axis=1)
    col_maxs = np.max(matrix, axis=0)
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)
    
    if abs(maximin - minimax) < 1e-10:
        return maximin  # Pure strategy equilibrium
    
    # Mixed strategy
    det = (a - b - c + d)
    if abs(det) < 1e-10:
        # Degenerate case
        return (a + d) / 2
    
    # Optimal strategy for row player: p1 = (d - c) / det, p2 = (a - b) / det
    # Game value: v = (ad - bc) / det
    v = (a * d - b * c) / det
    return v

def solve_game_value(matrix):
    """Solve for game value using minimax theorem"""
    n = matrix.shape[0]
    
    # Quick check for pure strategy equilibrium (saddle point)
    row_mins = np.min(matrix, axis=1)
    col_maxs = np.max(matrix, axis=0)
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)
    
    if abs(maximin - minimax) < 1e-10:
        return maximin
    
    if n == 2 and matrix.shape[1] == 2:
        return solve_2x2_game(matrix)
    
    # For larger games, need LP (return None to signal fallback)
    return None

# Test the analytical solver
test_matrices = [
    np.array([[0, 1], [-1, 0]]),  # Rock-paper-scissors style
    np.array([[1, -1], [-1, 1]]),  # Matching pennies
    np.array([[3, 0], [5, 1]]),   # Has saddle point
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # 3x3
]

print("Testing analytical solver:")
for i, m in enumerate(test_matrices):
    v = solve_game_value(m)
    print(f"Matrix {i+1} ({m.shape}): value = {v}")

# Now let's see what % of matrices in a real solve are 2x2 or have saddle points
from solver.solver import calculateEV
from solver.linprog import findBestStrategy_valueonly_cached
from solver.utils import full

calculateEV.cache_clear()
ev = calculateEV(full(5), full(5), 0, full(5), 4, "v")

# Count matrix sizes in LP cache
from solver.linprog import findBestStrategy_cached
sizes = {}
for key in findBestStrategy_cached.cache.keys():
    n = len(key)
    sizes[n] = sizes.get(n, 0) + 1

print(f"\nMatrix sizes in LP cache:")
for n, count in sorted(sizes.items()):
    print(f"  {n}x{n}: {count} matrices")
