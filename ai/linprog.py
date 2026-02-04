"""Linear programming functions for GOPS solver"""

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog
from typing import Optional
import time

solver_cache = {}

def get_solver_for_size(numRows, numCols):
    key = (numRows, numCols)
    if key in solver_cache:
        return solver_cache[key]
    solver = pywraplp.Solver.CreateSolver('GLOP')
    # Attach variables as attributes
    solver.p = [solver.NumVar(0, solver.infinity(), f'p_{i}') for i in range(numRows)]
    solver.v = solver.NumVar(-solver.infinity(), solver.infinity(), 'v')
    solver.constraints = []
    for j in range(numCols):
        constraint = solver.Constraint(0, solver.infinity())
        solver.constraints.append(constraint)
    prob_constraint = solver.Constraint(1, 1)
    for i in range(numRows):
        prob_constraint.SetCoefficient(solver.p[i], 1)
    objective = solver.Objective()
    objective.SetCoefficient(solver.v, 1)
    objective.SetMaximization()
    solver_cache[key] = solver
    return solver

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
    solver = get_solver_for_size(numRows, numCols)
    t1 = time.perf_counter()
    p = solver.p
    v = solver.v

    # Update constraint coefficients here if needed
    # Update constraint coefficients for the current payoffMatrix
    for j in range(numCols):
        constraint = solver.constraints[j]
        constraint.Clear()
        for i in range(numRows):
            constraint.SetCoefficient(p[i], payoffMatrix[i, j])
        constraint.SetCoefficient(v, -1)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        probabilities = np.array([p[i].solution_value() for i in range(numRows)])
        game_value = v.solution_value()
        return probabilities, game_value
    else:
        return findBestStrategy_scipy_fallback(payoffMatrix)

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
