"""Check what's hitting the LP solver"""
from solver.solver import calculateEV
from solver.linprog import findBestStrategy_cached, findBestStrategy_valueonly_cached
from solver.utils import full

calculateEV.cache_clear()
findBestStrategy_cached.cache_clear()

ev = calculateEV(full(6), full(6), 0, full(6), 5, "v")

print("LP cache (findBestStrategy_cached):")
sizes = {}
for key in findBestStrategy_cached.cache.keys():
    n = len(key)
    sizes[n] = sizes.get(n, 0) + 1
for n, count in sorted(sizes.items()):
    print(f"  {n}x{n}: {count}")

# Check the valueonly cache
print("\nTotal calculateEV cache entries:", calculateEV.cache_info().currsize)
print("Total findBestStrategy_cached entries:", findBestStrategy_cached.cache_info().currsize)

# What about saddle point hits?
import tests.globals as globals
print(f"\nGlobals stats:")
print(f"  Total calculated: {globals.totalCalculated}")
print(f"  Guaranteed wins: {globals.guarantee}")
print(f"  OR-Tools failures: {globals.ortools_fails}")
