import cProfile
import pstats
from solver import calculateEV
from utils import full

pr = cProfile.Profile()
pr.enable()

ev = calculateEV(full(6), full(6), 0, full(6), 5, "v")

pr.disable()

print(f"EV: {ev}")
stats = pstats.Stats(pr)
stats.sort_stats("tottime")  # Sort by total time in function
stats.print_stats(30)
