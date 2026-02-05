"""
Benchmark comparison between original and scipy-only solvers.
"""

import time
import numpy as np
from scipy.optimize import linprog
from ortools.linear_solver import pywraplp

def solve_scipy(matrix):
    n, m = matrix.shape
    c = np.zeros(n + 1)
    c[-1] = -1
    A_ub = np.column_stack([-matrix.T, np.ones(m)])
    b_ub = np.zeros(m)
    A_eq = np.array([[1]*n + [0]])
    b_eq = [1]
    bounds = [(0, None)]*n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return -res.fun if res.success else 0.0

def solve_ortools(matrix):
    n, m = matrix.shape
    solver = pywraplp.Solver.CreateSolver('GLOP')
    p = [solver.NumVar(0, solver.infinity(), f'p_{i}') for i in range(n)]
    v = solver.NumVar(-solver.infinity(), solver.infinity(), 'v')
    
    for j in range(m):
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

# Test matrices
np.random.seed(42)
matrices = [np.random.randn(n, n) for n in [2, 3, 4, 5, 6] for _ in range(100)]

print("Comparing LP solvers:")

# Scipy
start = time.time()
for m in matrices:
    solve_scipy(m)
scipy_time = time.time() - start
print(f"Scipy: {scipy_time:.3f}s")

# OR-Tools
start = time.time()
for m in matrices:
    solve_ortools(m)
ortools_time = time.time() - start
print(f"OR-Tools: {ortools_time:.3f}s")

print(f"\nOR-Tools is {scipy_time/ortools_time:.1f}x faster")
