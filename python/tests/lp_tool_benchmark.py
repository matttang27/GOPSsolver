# First, install OR-Tools
# pip install ortools

import time
import numpy as np
from ortools.linear_solver import pywraplp

def benchmark_ortools():
    """Benchmark using Google OR-Tools (C++ solver with Python interface)"""
    
    sizes = [3, 4, 5, 6, 7, 8]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices with OR-Tools...")
        
        times = []
        num_tests = 100 if size <= 5 else 50
        
        for test in range(num_tests):
            # Generate random payoff matrix (same as your GOPS matrices)
            payoffMatrix = np.random.uniform(-1, 1, (size, size))
            
            start_time = time.perf_counter()
            
            # Create solver (GLOP is Google's fast LP solver)
            solver = pywraplp.Solver.CreateSolver('GLOP')
            if not solver:
                print("Could not create solver")
                continue
            
            # Variables: p_0, ..., p_{size-1}, v
            p = [solver.NumVar(0, solver.infinity(), f'p_{i}') for i in range(size)]
            v = solver.NumVar(-solver.infinity(), solver.infinity(), 'v')
            
            # Constraints: sum_i p_i * M[i,j] >= v for all j
            for j in range(size):
                constraint = solver.Constraint(0, solver.infinity())
                for i in range(size):
                    constraint.SetCoefficient(p[i], payoffMatrix[i, j])
                constraint.SetCoefficient(v, -1)
            
            # Probability constraint: sum_i p_i = 1
            prob_constraint = solver.Constraint(1, 1)
            for i in range(size):
                prob_constraint.SetCoefficient(p[i], 1)
            
            # Objective: maximize v
            objective = solver.Objective()
            objective.SetCoefficient(v, 1)
            objective.SetMaximization()
            
            # Solve
            status = solver.Solve()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            if status != pywraplp.Solver.OPTIMAL:
                print(f"  Warning: Non-optimal solution for test {test}")
        
        if times:
            avg_time = np.mean(times)
            print(f"  Average time: {avg_time*1000:.3f}ms")
            print(f"  Per second: {1/avg_time:.1f} solves/sec")

def benchmark_scipy_comparison():
    """Compare with your current scipy approach"""
    from scipy.optimize import linprog
    
    sizes = [3, 4, 5, 6, 7, 8]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices with SciPy...")
        
        times = []
        num_tests = 100 if size <= 5 else 50
        
        for test in range(num_tests):
            payoffMatrix = np.random.uniform(-1, 1, (size, size))
            
            start_time = time.perf_counter()
            
            # Your exact formulation
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
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        if times:
            avg_time = np.mean(times)
            print(f"  Average time: {avg_time*1000:.3f}ms")
            print(f"  Per second: {1/avg_time:.1f} solves/sec")

if __name__ == "__main__":
    print("=== LP Solver Benchmark ===\n")
    
    print("1. Google OR-Tools (C++ backend):")
    benchmark_ortools()
    
    print("\n" + "="*50 + "\n")
    
    print("2. SciPy linprog (your current method):")
    benchmark_scipy_comparison()