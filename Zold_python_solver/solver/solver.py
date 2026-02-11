"""Main GOPS solver"""

from __future__ import annotations
import numpy as np
from typing import Union
from .cache import lru_cache
from multiprocessing import Manager, Pool, active_children


import tests.globals as globals
from .linprog import findBestStrategy_valueonly_cached, findBestStrategy_cached
from .utils import cmp, compress_cards, guaranteed

@lru_cache(maxsize=None)
def calculateEV(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, 
               prizes: tuple[int, ...], prizeIndex: int, returnType: str) -> Union[int, float]:
    """Calculate the expected value for the current game state (cached version)."""
    
    if len(cardsA) == 1:
        if (returnType == "v"): 
            return cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)
        if (returnType == "p"):
            return np.array([1])
        if (returnType == "m"):
            return np.matrix([[cmp(pointDiff + (cmp(cardsA[0], cardsB[0]) * prizes[0]), 0)]])
    
    # If pointDiff is negative, swap players to reduce number of unique states
    if pointDiff < 0 and returnType == "v":
        return -calculateEV(cardsB, cardsA, -pointDiff, prizes, prizeIndex, "v")
    
    # Update global counters
    globals.totalCalculated += 1

    cardsLeft = len(cardsA)
    matrix = np.zeros((cardsLeft, cardsLeft))

    # Check for guaranteed win
    #dominance = check_dominance_guaranteed(cardsA, cardsB, pointDiff, prizes)
    alreadyWon = guaranteed(cardsA, cardsB, pointDiff, prizes)
    if (alreadyWon != 0 and returnType == "v"):
        globals.guarantee += 1
        globals.caught += 1
        return alreadyWon

    # Calculate matrix
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

    matrix_tuple = tuple(tuple(row) for row in matrix)
    if returnType == "v":
        v = findBestStrategy_valueonly_cached(matrix_tuple)
        if (abs(v - 1) < 1e-10 or abs(v + 1) < 1e-10):
            globals.guarantee += 1
        return v
    else:
        # Only do full LP when you need probabilities
        p, v = findBestStrategy_cached(matrix_tuple)
        return p if returnType == "p" else v


if __name__ == "__main__":
    print(calculateEV((1, 2, 3), (1,2,3), 0, (1, 2, 3), 0, "m"))