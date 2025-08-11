totalCalculated = 0
guarantee = 0
caught = 0
ortools_fails = 0
timing_stats = {}

def reset_counters():
    """Reset all global counters"""
    global totalCalculated, guarantee, caught, ortools_fails, timing_stats
    totalCalculated = 0
    guarantee = 0
    caught = 0
    ortools_fails = 0
    timing_stats = {}

def print_stats():
    """Print current statistics"""
    print(f"Total calculated: {totalCalculated}")
    if totalCalculated > 0 and guarantee > 0:
        print(f"Guaranteed cases: {guarantee} ({guarantee / totalCalculated * 100:.1f}%)")
        print(f"Caught by detection: {caught} ({caught / guarantee * 100:.1f}%)")
    print(f"OR-Tools failures: {ortools_fails}")
    for matrix_size, stats in sorted(timing_stats.items()):
        count = stats['count']
        print(f"\nMatrix size: {matrix_size} ({count} solves)")
        for key in ['create_solver_time', 'var_time', 'constraint_time', 'prob_constraint_time', 'objective_time', 'solve_time']:
            pct = 100 * stats[key] / stats['total_time'] if stats['total_time'] > 0 else 0
            print(f"  {key.replace('_', ' ').capitalize()}: {pct:.1f}%")
        print(f"  Total time: {stats['total_time']:.3f} seconds")
