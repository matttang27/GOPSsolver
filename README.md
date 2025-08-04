I've been playing Game of Pure Strategy (or Goofspiel) with my friends, and I keep losing :)

Need to figure out the optimal strategy to clap everyone

There's two methods I can think of right now:

**1. Solving the nash equilibrium for every state, then working backwards to find the best EV.**

- Will find the mathematically proven best strategy
- Might be impossible for higher amounts of cards because of how many combinations there

**2. Using self-play reinforcement learning to trial and error the best strategy**

- Simpler to execute I think, program the game + rewards for the AI, and let it go wild
- Need to decide rewards
- Possibly inaccurate
- Lots of time



Let's look at Option 1 first.

The idea is to build a tree of all possible states and their payoffs.

Let's play a 3 card GOPS, where each player has 1-3, and the prizes are 1-3. Assume the first prize is 1. Let's try to calculate the expected value of Player A: 2, Player B: 1

Then, Player A has 1, with (1,3), Player B has 0, with (2,3), and the cards are 2,3. We then branch off further, assuming the prize is 2.

The immediate payoff matrix would be a 2x2 matrix.

|       | 2   | 3   |
|-------|-----|-----|
| **1** | -2  | -2  |
| **3** |  2  |  0  |

We then add 1 to each cell, since A is up 1.

|       | 2   | 3   |
|-------|-----|-----|
| **1** | -1  | -1  |
| **3** |  3  |  1  |

This is the difference in points after the second round, but we need to consider the future states as well. Thankfully for 3-card GOPS, the last round is set.

For example, in column (1,3), where player A plays 1 and player B players 3, the remaining cards are 3 and 2, with the prize being 3. Therefore player A will win 3 points.

|       | 2         | 3         |
|-------|-----------|-----------|
| **1** | -1 + 0    | -1 + 2    |
| **3** |  3 + -3   |  1 + -3   |


|       | 2   | 3   |
|-------|-----|-----|
| **1** | -1  |  1  |
| **3** |  0  | -2  |

The goal of GOPS is not to maximize your score difference, but to simply win. Therefore, in (3,3), even though Player B will win by 2, it is still only a win, so you consider it -1.

|       | 2   | 3   |
|-------|-----|-----|
| **1** | -1  |  1  |
| **3** |  0  | -1  |

Looking at the matrix, it looks like Player A should play 1 because it gives an average of 0, but if Player B knows this, they will play 2 all the time and win. Therefore, we need to find the mixed strategy that gives the best expected value, assuming your opponent knows your strategy and plays the best counter-strategy. This is the Nash equilibrium.

Now enter linear programming (which I learned in CO250 omg it's useful). We have two variables, p1 and p2, which are the probabilities of playing card 1 and card 3 respectively. The constraints are that p1 + p2 = 1, and both p1 and p2 must be between 0 and 1.

What is the optimal counter-strategy for Player B, given p1 and p2 for Player A? Note that Player B simply wants to minimize Player A's EV, as this is a zero-sum game.

If Player B chooses **2**, the expected value (EV) for Player A is:

```
EV = -1 × p₁ + 0 × p₂ = -p₁
```

If Player B chooses **3**, the EV for Player A is:

```
EV = 1 × p₁ + (-1) × p₂ = p₁ - p₂
```

Player B will always pick the column (strategy) that gives Player A the lowest EV. Thus, Player B's counter-strategy is always a pure strategy: either 2 or 3.

Player B wants to minimize Player A's EV, so:

- If `-p₁ < p₁ - p₂`, then Player B chooses **2**.
- Otherwise, Player B chooses **3**.

Simplifying the inequality:

```
-p₁ < p₁ - p₂
-2p₁ < -p₂
p₂ < 2p₁
```

So in this case, if `p₁ > 1/2p₂`, Player B will choose **2**; otherwise, Player B will choose **3**. That is Player B chooses 2 if Player A picks 1 at least 2/3rds of the time.

Since we're trying to find p1, p2 that **maximizes** Player A's **minimum** EV, where:

minEV = min(-p1, p1 - 2p2)

We turn this into a linear program, with variables

p1, p2, minEV are real numbers

max minEV

and objectives (which are also constraints): 
-1 * p1 + 0 * p2 >= minEV
1 * p1 + -1 * p2 >= min EV

and constraints
0 <= p1, p2 <= 1
p1 + p2 = 1

In LPs we need to format it as a minimzation problem with inequalities <= 0.

```
minimize -minEV

subject to:

-(1 * p1 + 0 * p2) + minEV <= 0
-(1 * p1 + -1 * p2) + minEV <= 0
0 <= p1, p2 <= 1
p1 + p2 = 1
```

Here is the code in python
```python

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
```