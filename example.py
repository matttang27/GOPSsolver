from __future__ import annotations
from scipy.optimize import linprog
import time
import numpy as np
from typing import Optional, Union

# Omg it's the spaceship operator
def cmp(a: Union[int, float], b: Union[int, float]) -> int:
    """Compare two numbers and return -1, 0, or 1 (spaceship operator)."""
    return (a > b) - (a < b)

# Given a n x n payoff matrix, returns p, the probabilities for the best strategy, and v, the expected value.
def findBestStrategy(payoffMatrix: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Find the best mixed strategy for a zero-sum game.
    
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

def createMatrix(cardsA: list[int], cardsB: list[int], pointDiff: int, prizes: list[int], prize: int) -> np.ndarray:
    """
    Create a payoff matrix for the given game state.
    
    Args:
        cardsA: Player A's remaining cards
        cardsB: Player B's remaining cards  
        pointDiff: Current point difference
        prizes: List of remaining prizes
        prize: Current prize value
        
    Returns:
        Numpy array representing the payoff matrix
    """
    if (len(cardsA) == 1):
        return np.array([[cmp(pointDiff + cmp(cardsA[0], cardsB[0]) * prize, 0)]])
    elif (len(cardsA) == 2):
        # Create a 2x2 matrix
        matrix = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                newCardsA = cardsA.copy()
                newCardsB = cardsB.copy()
                newCardsA.remove(cardsA[i])
                newCardsB.remove(cardsB[j])
                matrix[i][j] = pointDiff + cmp(cardsA[i], cardsB[j]) * prize
                matrix[i][j] = calculateEV(newCardsA, newCardsB, int(matrix[i][j]), [], prizes[0])
        return matrix
    else:
        raise ValueError(f"Unsupported card count: {len(cardsA)}")

def calculateEV(cardsA: list[int], cardsB: list[int], pointDiff: int, prizes: list[int], prizeIndex: int) -> Union[int, float]:
    """
    Calculate the expected value for the current game state.
    
    Args:
        cardsA: Player A's remaining cards
        cardsB: Player B's remaining cards
        pointDiff: Current point difference
        prizes: List of remaining prizes
        prizeIndex: Current prize index

    Returns:
        Expected value (int for base case, float for recursive case)
    """
    print(f"Calculating EV for A: {cardsA}, B: {cardsB}, pointDiff: {pointDiff}, prizes: {prizes}, prizeIndex: {prizeIndex}")
    if (len(cardsA) == 1):
        return cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)

    cardsLeft = len(cardsA)

    print(f"Cards left: {cardsLeft}")
    # Create n x n matrix

    matrix = np.zeros((cardsLeft, cardsLeft))
    for i in range(cardsLeft):
        for j in range(cardsLeft):
            newA = cardsA[:i] + cardsA[i+1:]
            newB = cardsB[:j] + cardsB[j+1:]
            newDiff = pointDiff + cmp(cardsA[i], cardsB[j]) * prizes[prizeIndex]
            newPrizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
            ev = 0.0
            print("Calculating EV for A:", newA, "B:", newB, "Diff:", newDiff, "Prizes:", newPrizes)
            for k in range(cardsLeft - 1):
                ev += calculateEV(newA, newB, newDiff, newPrizes, k)
                print(ev)

            ev /= cardsLeft - 1
            print(f"EV for A[{i}] vs B[{j}]: {ev}")
            matrix[i][j] = ev

    print(f"Payoff matrix:\n{matrix}")
    p, v = findBestStrategy(matrix)

    print(f"Best strategy found: p = {p}, v = {v}")
    if v is not None:
        return v
    else:
        raise RuntimeError("Failed to find best strategy")

#print(calculateEV([1],[3],1,[2],0))
#print(calculateEV([3],[1],1,[2],0))
#print(calculateEV([2,3],[2,3],1,[2,3],0))
#print(calculateEV([2,3],[1,3],0,[2,3],0))
#print(calculateEV([1,3],[2,3],2,[1,3],0))
print(calculateEV([1,3],[2,3],2,[1,3],1))
#print(calculateEV([1,2,3],[1,2,3],0,[1,2,3],0))
#print(calculateEV([1,2,3],[1,2,3],0,[1,2,3],1))
#print(calculateEV([1,2,3],[1,2,3],0,[1,2,3],2))