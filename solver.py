from __future__ import annotations
from scipy.optimize import linprog
import time
import numpy as np
from typing import Optional, Union
from functools import lru_cache

# Omg it's the spaceship operator
def cmp(a: Union[int, float], b: Union[int, float]) -> int:
    """Compare two numbers and return -1, 0, or 1 (spaceship operator)."""
    return (a > b) - (a < b)

def compress_cards(cardsA: tuple[int, ...], cardsB: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compress card values to remove gaps while preserving relative order.
    
    Args:
        cardsA: First player's cards
        cardsB: Second player's cards
    
    Returns:
        Tuple of (compressed_cardsA, compressed_cardsB)
    """
    # Get all unique values and sort them
    all_values = sorted(set(cardsA) | set(cardsB))
    
    # Create mapping from old values to new compressed values
    value_map = {old_val: new_val for new_val, old_val in enumerate(all_values, 1)}
    
    # Apply mapping to both card sets
    compressed_A = tuple(value_map[card] for card in cardsA)
    compressed_B = tuple(value_map[card] for card in cardsB)
    
    return compressed_A, compressed_B

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

def guaranteed(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, prizes: tuple[int, ...]) -> int:
    """
    Check if one side has enough cards higher than the other to guarantee a win.
    Checks also if one side has enough pointDiff even if they don't have higher cards.
    
    Args:
        cardsA: First player's cards
        cardsB: Second player's cards
        pointDiff: Current point difference
        prizes: Remaining prize cards
        
    Returns:
        1 if A is guaranteed, -1 if B is guaranteed, 0 otherwise.
    """
    # Guarantee array: guarantee[i] = sum of i largest prizes - sum of remaining prizes.
    # If you have X cards higher than the entire other side, you get a minimum of guarantee[X] points.
    cardsLeft = len(prizes)
    sorted_prizes = sorted(prizes, reverse=True)  # Sort descending to get largest first
    guarantee = [sum(sorted_prizes[:i]) - sum(sorted_prizes[i:]) for i in range(cardsLeft + 1)]
    [-6, 0, 4, 6]
    guaranteeA = sum(1 for card in cardsA if card > cardsB[-1])
    guaranteeB = sum(1 for card in cardsB if card > cardsA[-1])
    
    if (guarantee[guaranteeA] + pointDiff) > 0:
        return 1
    elif (pointDiff - guarantee[guaranteeB]) < 0:
        return -1
    else:
        return 0


@lru_cache(maxsize=None)
def calculateEV(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, prizes: tuple[int, ...], prizeIndex: int, returnType: str) -> Union[int, float]:
    """
    Calculate the expected value for the current game state (cached version).
    """
    if len(cardsA) == 1:
        if (returnType == "v"): 
            return cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)
        if (returnType == "p"):
            return np.array([1])
        if (returnType == "m"):
            return np.matrix([[cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)]])
    
    #If pointDiff is negative, swap players to reduce number of unique states
    if pointDiff < 0 and returnType == "v":
        return -calculateEV(cardsB, cardsA, -pointDiff, prizes, prizeIndex, "v")

    cardsLeft = len(cardsA)
    matrix = np.zeros((cardsLeft, cardsLeft))

    

    alreadyWon = guaranteed(cardsA, cardsB, pointDiff, prizes)
    if (alreadyWon != 0 and returnType == "v"):
        return alreadyWon


    for i in range(cardsLeft):
        for j in range(cardsLeft):
            if cardsA == cardsB and pointDiff == 0:
                if i == j:
                    matrix[i][j] = 0
                    continue
                elif i > j:
                    matrix[i][j] = -matrix[j][i]
                    continue
            
            newA = cardsA[:i] + cardsA[i+1:]
            newB = cardsB[:j] + cardsB[j+1:]

            newA, newB = compress_cards(newA, newB)
            newDiff = pointDiff + cmp(cardsA[i], cardsB[j]) * prizes[prizeIndex]
            newPrizes = prizes[:prizeIndex] + prizes[prizeIndex+1:]
            ev = 0.0
            
            for k in range(cardsLeft - 1):
                ev += calculateEV(newA, newB, newDiff, newPrizes, k, "v")

            ev /= cardsLeft - 1
            matrix[i][j] = ev

    if returnType == "m":
        return matrix

    p, v = findBestStrategy(matrix)

    if returnType == "p":
        return p
    else:
        return v

def full(n):
    """Returns a tuple from 1 to n"""
    return tuple(i for i in range(1, n + 1))

#check how long it takes
if __name__ == "__main__":
    for i in range(1, 9):
        start_time = time.time()
        print(f"Calculating EV for full({i})...")
        
        # Get cache info before calculation
        cache_before = calculateEV.cache_info()
        
        ev = calculateEV(full(i), full(i), 0, full(i), i - 1, "p")
        
        # Get cache info after calculation
        cache_after = calculateEV.cache_info()
        
        end_time = time.time()
        
        # Calculate cache statistics for this iteration
        new_hits = cache_after.hits - cache_before.hits
        new_misses = cache_after.misses - cache_before.misses
        total_calls = new_hits + new_misses
        hit_rate = (new_hits / total_calls * 100) if total_calls > 0 else 0
        
        print(f"EV for full({i}) = {ev}")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        print(f"Cache hits: {new_hits}, misses: {new_misses}, hit rate: {hit_rate:.1f}%")
        print(f"Total cache size: {cache_after.currsize} entries")
        print(f"Cumulative hits: {cache_after.hits}, misses: {cache_after.misses}")
        print()

    # Final cache summary
    print("Final cache statistics:")
    print(calculateEV.cache_info())

