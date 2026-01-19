[Game of Pure Strategy](https://en.wikipedia.org/wiki/Goofspiel)  (or Goofspiel), is a simple two-player card game where players bid cards to win prize cards. To put it simply, 
each player has cards from 1 to 13, and every turn a random prize card valued from 1 to 13 is revealed. Both players simultaneously play a card, and the player who played the higher card wins the prize card's value in points. If the players tie, either both split the points (effectively 0 points), or the prize card stays for the next round, increasing the value. The game continues until all prize cards are claimed, and the player with the most points wins. For sake of simplicity, we will be using the former tiebreaking rule.

Once you play this game a few times, the general strategy becomes apparent: you want to win prize cards by being slightly above your opponent's bid, while losing prize cards by as much as possible, while also prioritizing high-value prize cards. This leads to a complex and circular mind game of bluffing and predicting your opponent's moves.

For example, for a big prize, the optimal move would be to play a card that is just above what you expect your opponent to play, unless they play the largest card, in which case you should play your smallest card to minimize loss. However, the opponent is also thinking the same way, leading to a complex web of strategies and counter-strategies.

I tried researching online but didn't find any solutions until I actually implemented the entire thing, and then I found this paper: https://www.econstor.eu/bitstream/10419/98554/1/749748680.pdf

It's nice to know that they also formulated it as a dynamic programming + LP problem. However, they calculated raw point gain instead of win probability.

To calculate the EV of a given state, we consider all possible actions of both players and the resulting states. We create a matrix of payoffs, where each cell is the expected value of the resulting state. We then use linear programming to find the optimal mixed strategy for Player A, assuming Player B plays the best counter-strategy. The value of this optimal strategy is the EV of the current state.

Let's start with an example.

We play a 2 card GOPS. For asymmetry (as a symmetric state obviously has an EV of 0), let's say Player A has cards (1,3), Player B has cards (2,3), but player A is up by 1 point. Let's say the current prize card is 2, and the last prize card is 3. This is equivalent to a 3 card GOPS where player A played 2 and B played 1, and A won the prize of 1 point.

The payoff matrix is now a 2x2 matrix, where the rows are Player A's possible plays (1,3) and the columns are Player B's possible plays (2,3).

|       | 2   | 3   |
| ----- | --- | --- |
| **1** | ?   | ?   |
| **3** | ?   | ?   |

we can now manually calculate the payoff for each action. For example, if A plays 1 and B plays 2, then B wins the prize of 2 and has 2 points. They then tie for the last round and B ends up winning by 1 point. When calculating EV, we consider A's perspective, so this cell is -1. It's important to note that -1 is not the point difference, but rather the outcome: even if A loses by 10 points, it's still just -1.

Filling in the rest of the cells, we get:

|       | 2   | 3   |
| ----- | --- | --- |
| **1** | -1  | 1   |
| **3** | 0   | -1  |

Looking at the matrix, it looks like Player A should play 1 because it gives an average of 0, but if Player B knows this, they will play 2 all the time and win. Therefore, we need to find the mixed strategy that gives the best expected value, assuming your opponent knows your strategy and plays the best counter-strategy. This is the Nash equilibrium.

Now enter linear programming (which I learned in CO250 omg it's useful). We have two variables, p1 and p2, which are the probabilities of playing card 1 and card 3 respectively. The constraints are that p1 + p2 = 1, and both p1 and p2 must be between 0 and 1.

What is the optimal counter-strategy for Player B, given p1 and p2 for Player A? Note that for any strategy of Player A, Player B always has a **pure** strategy counter that minimizes Player A's expected value.

If Player B chooses **2**, the expected value (EV) for Player A is:

```
EV = -1 × p₁ + 0 × p₂ = -p₁
```

If Player B chooses **3**, the EV for Player A is:

```
EV = 1 × p₁ + (-1) × p₂ = p₁ - p₂
```

Player B will always pick the column (strategy) that gives Player A the lowest EV. Therefore,

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
-1 _ p1 + 0 _ p2 >= minEV
1 _ p1 + -1 _ p2 >= min EV

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

The solution is p1 = 0.333, p2 = 0.667, with an expected value of -0.333.

Let's now play 3-card GOPS, where Player A has (1,2,3), B has (1,2,3) and the prizes are (1,2,3). Let's say the first prize is 1.

We want to build a matrix like this:

|       | 1   | 2   | 3   |
| ----- | --- | --- | --- |
| **1** |     |     |     |
| **2** |     |     |     |
| **3** |     |     |     |

Let's go to Cell (2,1), where Player A plays 2, and Player B plays 1. Player A wins the prize of 1, and now we have Player A with (1,3), Player B with (2,3), and the prizes are (2,3), and the point difference is +1.

Now, there are two possible next prizes: 2 or 3. For prize 2, we have already calculated the EV of this state, which is -0.333. For prize 3, we can do the same process again, leading to -0.333 as well. We average these two EVs together, filling Cell (2,1) with -0.333.

We can already fill in 4 other cells immediately as well. If both players are in the same state and play the same card, the resulting state is also identical, so the EV is 0. Therefore, (1,1), (2,2), (3,3) are all 0. Additionally, Player A playing 1 and B playing 2 is the opposite of our previous example, so the EV is now flipped, or 0.333.

We now have:

|       | 1     | 2     | 3     |
| ----- | ----- | ----- | ----- |
| **1** | 0     | 0.333 | ?     |
| **2** | -0.333 | 0     | ?     |
| **3** | ?     | ?     | 0     |

Filling in the rest of the cells using 2-card EVs, we get:

|       | 1     | 2     | 3     |
| ----- | ----- | ----- | ----- |
| **1** | 0     | 0.333 | 1     |
| **2** | -0.333 | 0     | 0.333     |
| **3** | -1     | -0.333     | 0     |

We can see that A playing 1 is a **strictly dominant strategy**, as it gives a better EV than their other options regardless of what B plays. Therefore, both A and B should play 1 all the time.

That is the basis of the algorithm. To formalize:

A,B = cards of Player A / B, ordered tuples. A_i is the card of Player A at index i
P = remaining prizes, ordered tuple, not including the current prize.
D = point difference (A - B)
C = current prize
|A| = |B| = |P - 1| = x

To calculate EV(A,B,P,D,C), we use a recursive method.

If x = 0:
- Return 1 if D > 0, 0 if D = 0, -1 if D < 0

Otherwise, create a matrix, where each cell (I,J) represents A playing their ith card, and B playing their jth card. The value of each cell is

\[
M_{i,j}
\;=\;
\operatorname{EV}\!\left(A,B,P,D,C \mid a=A_i,\; b=B_j\right)
\;=\;
\frac{1}{|P|}
\sum_{c \in P}
\operatorname{EV}\!\left(A\setminus\{a\},\, B\setminus\{b\},\, P\setminus\{c\},\, D+\Delta(a,b,C),\, c\right)
\]

\[
\Delta(a,b,C)=
\begin{cases}
+C & a>b\\
0  & a=b\\
-C & a<b
\end{cases}
\]

We then use linear programming to find the optimal mixed strategy for Player A, assuming Player B plays the best counter-strategy. The value of this optimal strategy is the EV of the current state.

## Optimizations

Calculating the EV naively is very slow, as the number of states grows factorially with the number of cards. For a N card game, there are N! possible arrangements of cards for each player, and N! possible arrangements of prizes, leading to (N!)^3 states. For 13 card GOPS, this is approximately 2.4e29 states.

Even 5 card GOPS has (5!)^3 = 1,728,000 states, and so my initial script took 2 minutes to calculate the EV of 5 cards.

My current solution calculates 5 card EV in 0.1 seconds, and as mentioned earlier, with a 10x factor per extra card. Let's explore the various methods I used.

### Memoization

The simplest and most effective optimization was to save results. Surprisingly, the number of unique states that are seen from a given state is only 10% of all the states seen. I was not expecting this, as I thought it would be rare (for example, playing a 4 then a 5 vs reversed order, but you won both anyways). 
