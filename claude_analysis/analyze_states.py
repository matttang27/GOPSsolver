"""Analyze state space and caching behavior"""
from solver import calculateEV
from utils import full
import time

for n in range(1, 8):
    calculateEV.cache_clear()
    start = time.time()
    ev = calculateEV(full(n), full(n), 0, full(n), n - 1, "v")
    elapsed = time.time() - start
    cache_info = calculateEV.cache_info()
    print(f"full({n}): cache_size={cache_info.currsize:,}, misses={cache_info.misses:,}, hits={cache_info.hits:,}, time={elapsed:.3f}s")
