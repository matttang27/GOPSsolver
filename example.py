from scipy.optimize import linprog
import time
import numpy as np

# Matrix A: Player A's payoff matrix
A = np.array([
    [-1, 1],
    [0, -1]
])

# Objective: minimize -v
c = [0, 0, -1]

# Constraints: -A^T @ p + v <= 0 (i.e., worst-case columns)
A_ub = [
    [-A[0][0], -A[1][0], 1],  # Opponent plays col 0
    [-A[0][1], -A[1][1], 1],  # Opponent plays col 1
]
b_ub = [0, 0]

# Probabilities sum to 1
A_eq = [[1, 1, 0]]
b_eq = [1]

# Bounds for p1, p2, and v
bounds = [(0, 1), (0, 1), (None, None)]

# Solve
start_time = time.time()
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
elapsed = time.time() - start_time
print(f"⏱ linprog took {elapsed:.6f} seconds\n")

if res.success:
    p1, p2, v = res.x
    strategy = np.array([p1, p2])
    print(f"✅ Optimal strategy (Player A): card 1 = {p1:.3f}, card 3 = {p2:.3f}")
    print(f"🎯 Guaranteed expected value: {v:.3f}")
    

    # Check alternative strategies
    test_strategies = {
        "Always card 1": [1, 0],
        "Always card 3": [0, 1],
        "50/50": [0.5, 0.5],
        "70/30": [0.7, 0.3],
        "30/70": [0.3, 0.7],
        "Optimal (should match)": [p1, p2]
    }

    for name, s in test_strategies.items():
        ev_col0 = s[0]*A[0][0] + s[1]*A[1][0]
        ev_col1 = s[0]*A[0][1] + s[1]*A[1][1]
        worst = min(ev_col0, ev_col1)
        print(f"{name.ljust(20)} → min EV = {worst:.3f} (vs col {0 if ev_col0 < ev_col1 else 1})")
    
else:
    print("❌ Linear program failed to solve.")
