from solver import calculateEV
calculateEV.cache_clear()

# Test with asymmetric hands and pointDiff=0
cardsA = (1, 2, 4)
cardsB = (1, 3, 4)
prizes = (2, 3, 5)

for i in range(3):
    calculateEV.cache_clear()
    ev = calculateEV(cardsA, cardsB, 0, prizes, i, "v")
    print(f'idx={i} (prize={prizes[i]}): EV = {ev}')
