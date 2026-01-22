import numpy as np
from scipy.optimize import linprog


def sign(a):
    return 1 if a > 0 else -1 if a < 0 else 0
def newPop(list: tuple, index: int):
    return list[:index] + list[index+1:]
def calculateEV(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, 
               remaining: tuple[int, ...], current: int , returnType: str) -> float:
    x = len(cardsA)
    if x == 1:
        return sign(pointDiff + sign(cardsA[0] - cardsB[0]) * remaining[0])
    matrix = np.zeros((x,x))
    for i in range(x):
        for j in range(x):
            newA = newPop(cardsA,i)
            newB = newPop(cardsB,j)
            newDiff = pointDiff + sign(cardsA[i] - cardsB[j]) * current
            sumEV = 0
            for k in range(x - 1):
                newRemaining = newPop(remaining)
                sumEV += calculateEV(newA, newB, newDiff, newRemaining, remaining[k])
            matrix[i][j] = sumEV / (x - 1)
    
def solve_matrix(matrix: np.ndarray):
    shape = matrix.shape[0]
    c = [0] * shape + [-1]
    A_ub = np.hstack([-matrix.T, np.ones((shape, 1))])
    b_ub = [0] * shape
    A_eq = [[1] * shape + [0]]
    b_eq = [1]
    bounds = [(0, 1)] * shape + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                 bounds=bounds, method='highs')
"""
This is a payoff matrix, the EVs after A and B play their cards. (Note that actual EVs would be between 0 and 1)
        B
  | 0 | 1 | 2 |
A | -1| 2 | 0 |
  | 3 | -2| 1 |

We want to find the optimal mixed strategy for A.

Given an x by x matrix of EVs, we want to maximize the minimum expected value of every column.

want max min( [0,0]p_0 + [1,0]p_1, [0,1]p_0 + [1,1]p_1 )
create v, where v <= [0,0]p_0 + [1,0]p_1,  v <= [0,1]p_0 + [1,1]p_1

min 0p_0 + 0p_1 + 1v

p_i is the probability you will choose the ith action.
sum of p_i = 1

therefore, for each column i in the matrix:
v <= sum( matrix[j][i] * p_j) for j in range(x)
-sum(matrix[j][i] * p_j) + v <= 0

Therefore, to do multiplication with the vector [p_0, p_1, ..., v]
A_ub = transposed(matrix) with an extra column of 1s at the end

"""
