from __future__ import annotations
from scipy.optimize import linprog
import time
import numpy as np
from typing import Optional, Union

# Omg it's the spaceship operator
def cmp(a: Union[int, float], b: Union[int, float]) -> int:
    """Compare two numbers and return -1, 0, or 1 (spaceship operator)."""
    return (a > b) - (a < b)



def findBestStrategy(payoffMatrix: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Given a n x n payoff matrix, returns p, the probabilities for the best strategy, and v, the expected value.
    
    Args:
        payoffMatrix: n x m numpy array representing the payoff matrix
        
    Returns:
        Tuple of (probability distribution, expected value) or (None, None) if failed
    """
    numRows, numCols = payoffMatrix.shape
    # Variables: p_0, ..., p_{numRows-1}, v
    c = np.zeros(numRows + 1)
    c[-1] = -1  # maximize v (minimize -v)

    # For each column, constraint: sum_i p_i * payoffMatrix[i][j] >= v
    # → -sum_i p_i * payoffMatrix[i][j] + v <= 0
    A_ub = []
    b_ub = []
    for j in range(numCols):
        row = [-payoffMatrix[i][j] for i in range(numRows)] + [1]
        A_ub.append(row)
        b_ub.append(0)

    # Probabilities sum to 1
    A_eq = [ [1]*numRows + [0] ]
    b_eq = [1]

    # Probabilities in [0,1], v unbounded
    bounds = [(0, 1)] * numRows + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        p = res.x[:-1]
        v = res.x[-1]
        return p, v
    else:
        return None, None

# Given a payoff matrix and probability distribution p, finds the opponent's best pure strategy, which is also it's best counterplay.
# Prints the minimum expected value and the column index of the best counterplay.
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

def calculateEV(cardsA: list[int], cardsB: list[int], pointDiff: int, prizes: list[int], prizeIndex: int, returnType: str) -> Union[int, float]:
    """
    Calculate the expected value for the current game state.
    
    Args:
        cardsA: Player A's remaining cards
        cardsB: Player B's remaining cards
        pointDiff: Current point difference
        prizes: List of remaining prizes
        prizeIndex: Current prize index
        returnType: Whether to return the matrix, the ev, or the optimal probability distribution.

    Returns:
        Expected value (int for base case, float for recursive case)
    """
    #print(f"Calculating EV for A: {cardsA}, B: {cardsB}, pointDiff: {pointDiff}, prizes: {prizes}, prizeIndex: {prizeIndex}")
    if (len(cardsA) == 1):
        return cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)

    cardsLeft = len(cardsA)

    #print(f"Cards left: {cardsLeft}")
    # Create n x n matrix

    matrix = np.zeros((cardsLeft, cardsLeft))
    for i in range(cardsLeft):
        for j in range(cardsLeft):
            newA = cardsA[:i] + cardsA[i+1:]
            newB = cardsB[:j] + cardsB[j+1:]
            newDiff = pointDiff + cmp(cardsA[i], cardsB[j]) * prizes[prizeIndex]
            newPrizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
            ev = 0.0
            #print("Calculating EV for A:", newA, "B:", newB, "Diff:", newDiff, "Prizes:", newPrizes)
            for k in range(cardsLeft - 1):
                ev += calculateEV(newA, newB, newDiff, newPrizes, k, "v")
                #print(ev)

            ev /= cardsLeft - 1
            #print(f"EV for A[{i}] vs B[{j}]: {ev}")
            matrix[i][j] = ev

    if (returnType == "m"):
        return matrix

    #print(f"Payoff matrix:\n{matrix}")
    p, v = findBestStrategy(matrix)

    #print(f"Best strategy found: p = {p}, v = {v}")
    if returnType == "p":
        return p
    else:
        return v

def full(n):
    """Returns an array from 1 to n"""
    return [i for i in range(1, n + 1)]

#check how long it takes


start_time = time.time()
print([calculateEV(full(5), full(5), 0, full(5), 1, "p")])
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

