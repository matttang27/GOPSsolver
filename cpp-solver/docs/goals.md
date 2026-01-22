# Goals

Build a fast C++ solver for Goofspiel (Game of Pure Strategy) based on my Python reference implementation, with optional optimizations that can be toggled to compare performance and memory use.

## Goofspiel (simplified)
- Two players each hold cards 1..N.
- A prize card is revealed each round from a separate 1..N deck.
- Both players simultaneously bid one of their remaining cards.
- Higher bid wins the prize value; ties cause no point swing.
- The game value is 1 for player A win, -1 for player B win, and 0 for tie at the end of N rounds.

## Python solver (simplified)
- State = remaining cards for both players, score difference, and remaining prize cards.
- For the current revealed prize, build an NxN payoff matrix over all bid pairs.
- Each bid pair recurses on the reduced state and averages over all possible next prize cards (uniform).
- Solve the zero-sum matrix game with linear programming to get the optimal value and mixed strategy.
- Heavy memoization, card-value compression to canonical form, optional player-swap symmetry, and a guaranteed-win check to prune states.
