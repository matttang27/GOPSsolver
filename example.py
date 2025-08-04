from scipy.optimize import linprog
import time
import numpy as np

# Omg it's the spaceship operator
def cmp(a, b):
    return (a > b) - (a < b)

def findBestStrategy(payoffMatrix):
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

def findBestCounterplay(payoffMatrix,p):
    ev_cols = [sum(p[i] * payoffMatrix[i][j] for i in range(payoffMatrix.shape[0])) for j in range(payoffMatrix.shape[1])]
    worst = min(ev_cols)
    worst_col = ev_cols.index(worst)
    
    print(f"Counterplay → min EV = {worst:.3f} (vs col {worst_col}), p = {p}")

def createMatrix(cardsA: list[int], cardsB: list[int], pointDiff: int, prizes: list[int], prize: int):
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
def calculateEV(cardsA: list[int], cardsB: list[int], pointDiff: int, prizes: list[int], prize: int):
    if (len(cardsA) == 1):
        return cmp(pointDiff + cmp(cardsA[0], cardsB[0]) * prize, 0)
        
    elif (len(cardsA) == 2):
        #create a 2x2 matrix
        matrix = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                newCardsA = cardsA.copy()
                newCardsB = cardsB.copy()
                newCardsA.remove(cardsA[i])
                newCardsB.remove(cardsB[j])
                matrix[i][j] = pointDiff + cmp(cardsA[i], cardsB[j]) * prize
                matrix[i][j] = calculateEV(newCardsA, newCardsB, int(matrix[i][j]), [], prizes[0])

        p,v = findBestStrategy(matrix)
        return v


print(calculateEV([1],[3],1,[],2))
print(calculateEV([3],[1],1,[],2))
print(calculateEV([2,3],[2,3],1,[3],2))