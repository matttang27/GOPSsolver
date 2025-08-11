"""Linear programming functions for GOPS solver"""

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog
from typing import Optional
from cache import lru_cache
import globals
import time

def findBestStrategy_scipy_fallback(payoffMatrix: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Fallback to SciPy when OR-Tools fails."""
    numRows, numCols = payoffMatrix.shape
    c = np.zeros(numRows + 1)
    c[-1] = -1  # maximize v
    
    A_ub = []
    b_ub = []
    for j in range(numCols):
        row = [-payoffMatrix[i][j] for i in range(numRows)] + [1]
        A_ub.append(row)
        b_ub.append(0)
    
    A_eq = [[1]*numRows + [0]]
    b_eq = [1]
    bounds = [(0, None)]*numRows + [(None, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                 bounds=bounds, method='highs')
    
    if res.success:
        probabilities = res.x[:-1]
        expected_value = -res.fun
        return probabilities, expected_value
    else:
        return None, None

def findBestStrategy(payoffMatrix: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float]]:
    """OR-Tools linear programming solver with SciPy fallback"""

    numRows, numCols = payoffMatrix.shape

    t0 = time.perf_counter()
    solver = pywraplp.Solver.CreateSolver('GLOP')
    t1 = time.perf_counter()

    if not solver:
        return None, None

    # Variables: p_0, ..., p_{numRows-1}, v
    p = [solver.NumVar(0, solver.infinity(), f'p_{i}') for i in range(numRows)]
    v = solver.NumVar(-solver.infinity(), solver.infinity(), 'v')
    t2 = time.perf_counter()

    # Constraints: sum_i p_i * M[i,j] >= v for all j
    for j in range(numCols):
        constraint = solver.Constraint(0, solver.infinity())
        for i in range(numRows):
            constraint.SetCoefficient(p[i], payoffMatrix[i, j])
        constraint.SetCoefficient(v, -1)
    t3 = time.perf_counter()

    # Probability constraint: sum_i p_i = 1
    prob_constraint = solver.Constraint(1, 1)
    for i in range(numRows):
        prob_constraint.SetCoefficient(p[i], 1)
    t4 = time.perf_counter()

    # Objective: maximize v
    objective = solver.Objective()
    objective.SetCoefficient(v, 1)
    objective.SetMaximization()
    t5 = time.perf_counter()

    # Solve
    solve_start = t5
    status = solver.Solve()
    solve_end = time.perf_counter()

    total_time = solve_end - t0
    create_solver_time = t1 - t0
    var_time = t2 - t1
    constraint_time = t3 - t2
    prob_constraint_time = t4 - t3
    objective_time = t5 - t4
    solve_time = solve_end - solve_start

    # Track timing by matrix size
    matrix_size = numRows * numCols
    
    if not hasattr(globals, 'timing_stats'):
        globals.timing_stats = {}
    
    if matrix_size not in globals.timing_stats:
        globals.timing_stats[matrix_size] = {
            'count': 0,
            'total_time': 0,
            'create_solver_time': 0,
            'var_time': 0,
            'constraint_time': 0,
            'prob_constraint_time': 0,
            'objective_time': 0,
            'solve_time': 0
        }
    
    stats = globals.timing_stats[matrix_size]
    stats['count'] += 1
    stats['total_time'] += total_time
    stats['create_solver_time'] += create_solver_time
    stats['var_time'] += var_time
    stats['constraint_time'] += constraint_time
    stats['prob_constraint_time'] += prob_constraint_time
    stats['objective_time'] += objective_time
    stats['solve_time'] += solve_time

    if status == pywraplp.Solver.OPTIMAL:
        probabilities = np.array([p[i].solution_value() for i in range(numRows)])
        game_value = v.solution_value()
        return probabilities, game_value
    else:
        globals.ortools_fails += 1  # Update global counter
        return findBestStrategy_scipy_fallback(payoffMatrix)

@lru_cache(maxsize=None)
def findBestStrategy_cached(matrix_tuple):
    """Cache linear programming results"""
    matrix = np.array(matrix_tuple)
    return findBestStrategy(matrix)

def findBestStrategy_valueonly(matrix_tuple, payoffMatrix: np.ndarray) -> float:
    """Ultra-fast value calculation for when you don't need probabilities"""
    
    # Quick pure strategy check
    row_mins = np.min(payoffMatrix, axis=1)
    max_row_min = np.max(row_mins)
    
    col_maxs = np.max(payoffMatrix, axis=0)
    min_col_max = np.min(col_maxs)
    
    if abs(max_row_min - min_col_max) < 1e-10:
        return max_row_min
    
    # Otherwise solve simplified LP
    p, v = findBestStrategy_cached(matrix_tuple)
    return v

def findBestStrategy_valueonly_cached(matrix_tuple):
    """Cache value-only results"""
    matrix = np.array(matrix_tuple)
    return findBestStrategy_valueonly(matrix_tuple, matrix)

def findBestCounterplay(payoffMatrix: np.ndarray, p: np.ndarray) -> None:
    """
    Find and print the opponent's best counterplay strategy.
    
    Args:
        payoffMatrix: The game's payoff matrix
        p: Probability distribution for row player's strategy
    """
    ev_cols = [sum(p[i] * payoffMatrix[i][j] for i in range(payoffMatrix.shape[0])) for j in range(payoffMatrix.shape[1])]
    worst = min(ev_cols)
    worst_col = ev_cols.index(worst)
    
    print(f"Counterplay → min EV = {worst:.3f} (vs col {worst_col}), p = {p}")

def findBestStrategyKnownRange(payoffMatrix: np.ndarray, lower, higher) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Fallback to SciPy when OR-Tools fails."""
    numRows, numCols = payoffMatrix.shape
    c = np.zeros(numRows + 1)
    c[-1] = -1  # maximize v
    
    A_ub = []
    b_ub = []
    for j in range(numCols):
        row = [-payoffMatrix[i][j] for i in range(numRows)] + [1]
        A_ub.append(row)
        b_ub.append(0)
    
    A_eq = [[1]*numRows + [0]]
    b_eq = [1]
    bounds = [(0, None)]*numRows + [(lower, higher)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                 bounds=bounds, method='highs')
    
    if res.success:
        probabilities = res.x[:-1]
        expected_value = -res.fun
        return probabilities, expected_value
    else:
        return None, None
