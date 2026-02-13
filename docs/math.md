# Solver Derivation Notes

I started this project in order to try and 'solve' Goofspiel - I hoped that creating a Nash Equilibrium solution would allow me to find patterns and strategies that I could employ. However, it would be much harder than I thought.

This readme will assume you know how Goofspiel is played. I am using the identical cards = 0 points, instead of adding the prize card to the next round.
 
<details>
<summary>Why?</summary>

- It complicates the state space, as now the effective current prize can be much larger than N.
- If high cards are tied, the optimal move is usually to continue tying, leading to a boring game. This also heavily punishes you if you would lose the highest card battle (ex. having 10-9-7 vs 10-9-8).
</details>



## Creating the Solver

Our objective is to maximize winrate, and not points gained. Therefore, a final state where we win is worth 1, a draw is 0, and a loss is -1. Final value = $sgn(PointDiff)$.

A state can be defined as $S = (A,B,P,d,c)$, where A = player A cards, B = player B cards, P = remaining cards (not including the current card c), d = point difference, c = current card.
Let $N = |A| = |B| = |P|+1$

Therefore, if player A plays $A_i$ in A and player B plays $B_j$ in B, and a new prize card $c_n$ is shown, the new state is:
$$
S(A_i,B_j,c_n) = (A - A_i, B - B_j, P - c_n, d + c * sgn(A_i - B_j), c_n)
$$

(We remove $A_i$ from A, $B_j$ from B, and $c_n$ from P, then change point difference based on who won the previous prize card).

Since the prize card can be any card in $P$, the EV of an action set $(A_i,B_j)$ is
$$
EV(S(A_i,B_j)) = \frac{1}{|P|}\sum_{c_n \in P} EV(S(A_i,B_j,c_n))
$$

To find the best strategy, we must account for all possible combinations of A and B actions, and the possible new prize cards shown after each action.

Therefore, the EV of state $S$ can be calculated by an $N$x$N$ payoff matrix, where each cell $i,j = EV(S(A_i,B_j))$

This algorithm is recursive and grows at an exponential pace. For an $N$ card state, we have $N * N * (N-1)$ child states, each with child states of their own. Overall, traversing the full tree of an $N$ card state is $N^3 * (N-1) ^ 3 ... = N!^3$.

However, many states will be seen multiple times. Without considering pointDiff for now, for n cards, there are $C(N,n)$ unique states for player A cards, $C(N,n)$ for player B cards, $C(N,n-1)$ for $P$, and $(N-n+1)$ choices for current card $c$, giving us $C(N,n)^2 \cdot C(N,n-1) \cdot (N-n+1)$ total unique states to consider.

Considering pointDiffs is tricky, as states have different possible pointDiffs. For example, the starting state can only be 0, while end states can have large magnitude differences (up to roughly $\pm(\frac{N(N+1)}{2}-2)$), so the number of possible pointDiff values grows on the order of $N^2$ in the worst case.
