import time
import cProfile
import pstats
import globals
from solver import calculateEV
from utils import full
from linprog import findBestStrategy_scipy_fallback, findBestStrategyKnownRange
import numpy as np



def runFull():
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
        print(
            f"Cache hits: {new_hits}, misses: {new_misses}, hit rate: {hit_rate:.1f}%"
        )
        print(f"Total cache size: {cache_after.currsize} entries")
        print(f"Cumulative hits: {cache_after.hits}, misses: {cache_after.misses}")
        #print(calculateEV.cache)
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


def test_scipy_vs_adominant_speed(ev_lower=0.95, ev_upper=1.01):
    """
    Compare speed of findBestStrategy_scipy_fallback vs findBestStrategyKnownRange
    on matrices with EV in [ev_lower, ev_upper) from full(6).
    Also print the percentage of states in that range.
    """
    print("Simulating full(6) to populate cache...")
    calculateEV.cache_clear()
    ev = calculateEV(full(6), full(6), 0, full(6), 5, "v")
    print(f"Simulation complete. EV for full(6): {ev}")

    # Gather all matrices in cache with EV in the specified range
    print(f"Collecting matrices with {ev_lower} <= EV < {ev_upper} from cache...")
    high_ev_matrices = []
    total_states = 0
    for key, value in calculateEV.cache.items():
        ev_val = value if not isinstance(value, tuple) else value[0]
        total_states += 1
        if ev_lower <= ev_val <= ev_upper:
            row_tuple = key[0]
            col_tuple = key[1]
            high_ev_matrices.append((row_tuple, col_tuple))
    print(f"Found {len(high_ev_matrices)} matrices with {ev_lower} <= EV < {ev_upper}.")
    if total_states > 0:
        percent = 100 * len(high_ev_matrices) / total_states
        print(f"That is {percent:.2f}% of all {total_states} states.")
    else:
        print("No states found in cache.")

    # Clear the cache to avoid interference
    calculateEV.cache_clear()

    # Prepare matrices as numpy arrays
    matrices = []
    for row_tuple, col_tuple in high_ev_matrices:
        payoff_matrix = np.subtract.outer(row_tuple, col_tuple)
        matrices.append(payoff_matrix)

    # Time findBestStrategy_scipy_fallback
    print("Timing findBestStrategy_scipy_fallback...")
    start = time.time()
    for mt in matrices:
        findBestStrategy_scipy_fallback(mt)
    scipy_time = time.time() - start
    print(f"findBestStrategy_scipy_fallback: {scipy_time:.3f} seconds for {len(matrices)} matrices.")

    # Time findBestStrategyKnownRange
    print("Timing findBestStrategyKnownRange...")
    start = time.time()
    for mt in matrices:
        findBestStrategyKnownRange(mt, ev_lower, ev_upper)
    adom_time = time.time() - start
    print(f"findBestStrategyKnownRange: {adom_time:.3f} seconds for {len(matrices)} matrices.")

    if adom_time > 0:
        print(f"Speedup: {scipy_time/adom_time:.2f}x (scipy/knownrange)")
    else:
        print("findBestStrategyKnownRange took 0 seconds (no matrices or error).")

if __name__ == "__main__":
    runFull()