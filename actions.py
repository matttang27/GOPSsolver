import time
import cProfile
import pstats
import globals
from solver import calculateEV
from utils import full



def runFull():
    for i in range(1, 7):
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
        print(
            f"Cache hits: {new_hits}, misses: {new_misses}, hit rate: {hit_rate:.1f}%"
        )
        print(f"Total cache size: {cache_after.currsize} entries")
        print(f"Cumulative hits: {cache_after.hits}, misses: {cache_after.misses}")
        print(calculateEV.cache)
        globals.print_stats()


def profile():
    pr = cProfile.Profile()
    pr.enable()

    # Run the calculation you want to profile
    ev = calculateEV(full(7), full(7), 0, full(7), 6, "p")

    pr.disable()

    # Print results
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Show top 20 functions


def findGuaranteeThreshold():
    for i in range(1, 7):
        # Binary search to find threshold where EV transitions from < 1 to exactly 1
        left = 0
        right = 1000
        threshold = -1

        while left <= right:
            mid = (left + right) // 2
            ev_current = calculateEV(full(i), full(i), mid, full(i), i - 1, "v")

            if ev_current >= 1:
                threshold = mid
                right = mid - 1  # Search for a smaller threshold
            else:
                left = mid + 1  # Search for a larger threshold

        if threshold != -1:
            ev_at_threshold = calculateEV(
                full(i), full(i), threshold, full(i), i - 1, "v"
            )
            ev_before = (
                calculateEV(full(i), full(i), threshold - 1, full(i), i - 1, "v")
                if threshold > 0
                else 0
            )
            print(
                f"For full({i}), threshold at pointDiff = {threshold} (EV: {ev_at_threshold}), previous {threshold-1} (EV: {ev_before})"
            )
        else:
            print(f"For full({i}), threshold is higher than 1000")
