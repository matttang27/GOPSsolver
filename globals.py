totalCalculated = 0
guarantee = 0
caught = 0
ortools_fails = 0

def reset_counters():
    """Reset all global counters"""
    global totalCalculated, guarantee, caught, ortools_fails
    totalCalculated = 0
    guarantee = 0
    caught = 0
    ortools_fails = 0

def print_stats():
    """Print current statistics"""
    print(f"Total calculated: {totalCalculated}")
    if totalCalculated > 0 and guarantee > 0:
        print(f"Guaranteed cases: {guarantee} ({guarantee / totalCalculated * 100:.1f}%)")
        print(f"Caught by detection: {caught} ({caught / guarantee * 100:.1f}%)")
    print(f"OR-Tools failures: {ortools_fails}")