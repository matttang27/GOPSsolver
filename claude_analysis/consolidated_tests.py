"""
CONSOLIDATED COMPRESSION TEST SUITE
====================================
This file combines all compression strategy tests into one efficient run.

SUMMARY OF FINDINGS:
-------------------
1. GCD Normalization: VALID, ~1.02x reduction (minimal benefit)
2. Sorted Prizes (no prizeIndex): INVALID - prizeIndex matters for EV
3. Outcome Mapping: VALID, ~1.4x reduction overall
   - 3-4x at shallow depths (1-2 prizes)
   - ~1.1x at deep depths (many prizes)
4. Scale Invariance: VALID - scaling (pointDiff, prizes) together preserves EV
5. Compressed Mapping (win/tie/loss counts): INVALID - loses information

REDUNDANT FILES REMOVED:
- test_prize_scaling.py, test_prize_scaling2.py -> merged into scale_invariance_test
- test_boundary_signature.py, test_minimal_signature.py -> merged into signature_tests
- test_outcome_mapping.py -> merged into outcome_mapping_test  
- test_gcd_compression.py, test_compression_v2.py -> merged into gcd_test
- test_prize_structure.py, test_collapse.py -> merged into prize_index_test
- test_prizes.py, test_idx.py -> basic tests covered by others
- analyze_equivalence.py -> insights merged
"""

import time
from collections import defaultdict
from math import gcd
from functools import reduce
from itertools import product

# Import solver
import sys
sys.path.insert(0, '..')
from solver.solver import calculateEV
from solver.utils import full


def compute_gcd(pointDiff, prizes):
    """Compute GCD of pointDiff and all prizes."""
    if not prizes:
        return 1
    values = list(prizes) + ([abs(pointDiff)] if pointDiff else [])
    return reduce(gcd, values)


def outcome_mapping(pointDiff, prizes):
    """Map all 3^n round outcomes to final game results."""
    if not prizes:
        return (1 if pointDiff > 0 else (-1 if pointDiff < 0 else 0),)
    results = []
    for mask in range(3 ** len(prizes)):
        swing = 0
        temp = mask
        for p in prizes:
            swing += (temp % 3 - 1) * p
            temp //= 3
        final = pointDiff + swing
        results.append(1 if final > 0 else (-1 if final < 0 else 0))
    return tuple(results)


def compressed_mapping(pointDiff, prizes):
    """Count wins/ties/losses instead of full mapping."""
    if not prizes:
        r = 1 if pointDiff > 0 else (-1 if pointDiff < 0 else 0)
        return (1 if r == 1 else 0, 1 if r == 0 else 0, 1 if r == -1 else 0)
    wins = ties = losses = 0
    for mask in range(3 ** len(prizes)):
        swing = 0
        temp = mask
        for p in prizes:
            swing += (temp % 3 - 1) * p
            temp //= 3
        final = pointDiff + swing
        if final > 0: wins += 1
        elif final < 0: losses += 1
        else: ties += 1
    return (wins, ties, losses)


def run_solver(n):
    """Run solver for n cards and return cache."""
    calculateEV.cache_clear()
    ev = calculateEV(full(n), full(n), 0, full(n), n - 1, "v")
    return ev, dict(calculateEV.cache)


def check_consistency(grouped):
    """Check if all entries in each group have same EV."""
    bad = 0
    for entries in grouped.values():
        evs = [e[-1] for e in entries] if isinstance(entries[0], tuple) else entries
        if max(evs) - min(evs) > 1e-9:
            bad += 1
    return bad


def print_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


# ============================================================================
# TEST 1: GCD NORMALIZATION
# ============================================================================
def test_gcd_compression(cache):
    """Test GCD normalization of (pointDiff, prizes)."""
    by_gcd = defaultdict(list)
    for key, value in cache.items():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        g = compute_gcd(pointDiff, prizes)
        norm_prizes = tuple(p // g for p in prizes)
        norm_diff = pointDiff // g
        canonical = (cardsA, cardsB, norm_diff, norm_prizes, prizeIndex, returnType)
        by_gcd[canonical].append(value)
    
    bad = check_consistency(by_gcd)
    reduction = len(cache) / len(by_gcd)
    return len(by_gcd), bad, reduction


# ============================================================================
# TEST 2: SORTED PRIZES (IGNORING prizeIndex)
# ============================================================================
def test_sorted_prizes(cache):
    """Test if sorting prizes and ignoring prizeIndex works."""
    by_sorted = defaultdict(list)
    for key, value in cache.items():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        sorted_prizes = tuple(sorted(prizes))
        canonical = (cardsA, cardsB, pointDiff, sorted_prizes, returnType)
        by_sorted[canonical].append(value)
    
    bad = check_consistency(by_sorted)
    reduction = len(cache) / len(by_sorted)
    return len(by_sorted), bad, reduction


# ============================================================================
# TEST 3: OUTCOME MAPPING
# ============================================================================
def test_outcome_mapping(cache):
    """Test outcome mapping compression."""
    by_mapping = defaultdict(list)
    for key, value in cache.items():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        current = prizes[prizeIndex]
        remaining = tuple(p for i, p in enumerate(prizes) if i != prizeIndex)
        mappings = tuple(outcome_mapping(pointDiff + o*current, remaining) for o in [-1,0,1])
        canonical = (cardsA, cardsB, mappings, current, returnType)
        by_mapping[canonical].append(value)
    
    bad = check_consistency(by_mapping)
    reduction = len(cache) / len(by_mapping)
    return len(by_mapping), bad, reduction


# ============================================================================
# TEST 4: COMPRESSED MAPPING (WIN/TIE/LOSS COUNTS)
# ============================================================================
def test_compressed_mapping(cache):
    """Test if just counting wins/ties/losses is sufficient."""
    by_compressed = defaultdict(list)
    for key, value in cache.items():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        current = prizes[prizeIndex]
        remaining = tuple(p for i, p in enumerate(prizes) if i != prizeIndex)
        mappings = tuple(compressed_mapping(pointDiff + o*current, remaining) for o in [-1,0,1])
        canonical = (cardsA, cardsB, mappings, current, returnType)
        by_compressed[canonical].append(value)
    
    bad = check_consistency(by_compressed)
    reduction = len(cache) / len(by_compressed)
    return len(by_compressed), bad, reduction


# ============================================================================
# TEST 5: GCD + OUTCOME MAPPING COMBINED
# ============================================================================
def test_gcd_and_mapping(cache):
    """Test GCD normalization combined with outcome mapping."""
    by_both = defaultdict(list)
    for key, value in cache.items():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        # GCD normalize first
        g = compute_gcd(pointDiff, prizes)
        norm_prizes = tuple(p // g for p in prizes)
        norm_diff = pointDiff // g
        # Then compute outcome mapping
        current = norm_prizes[prizeIndex]
        remaining = tuple(p for i, p in enumerate(norm_prizes) if i != prizeIndex)
        mappings = tuple(outcome_mapping(norm_diff + o*current, remaining) for o in [-1,0,1])
        canonical = (cardsA, cardsB, mappings, current, returnType)
        by_both[canonical].append(value)
    
    bad = check_consistency(by_both)
    reduction = len(cache) / len(by_both)
    return len(by_both), bad, reduction


# ============================================================================
# TEST 6: OUTCOME MAPPING BY DEPTH
# ============================================================================
def test_mapping_by_depth(cache):
    """Analyze outcome mapping reduction at each depth."""
    by_depth = defaultdict(lambda: {'original': 0, 'mapped': set()})
    
    for key, value in cache.items():
        cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
        depth = len(prizes)
        by_depth[depth]['original'] += 1
        
        current = prizes[prizeIndex]
        remaining = tuple(p for i, p in enumerate(prizes) if i != prizeIndex)
        mappings = tuple(outcome_mapping(pointDiff + o*current, remaining) for o in [-1,0,1])
        canonical = (cardsA, cardsB, mappings, current, returnType)
        by_depth[depth]['mapped'].add(canonical)
    
    return {d: (v['original'], len(v['mapped'])) for d, v in by_depth.items()}


# ============================================================================
# TEST 7: SCALE INVARIANCE
# ============================================================================
def test_scale_invariance():
    """Verify that scaling (pointDiff, prizes) together preserves EV."""
    test_cases = [
        ((1,2,3), (1,2,3), 1, (1,2,3), 2, (2,4,6)),  # 2x scale
        ((1,2,3,4), (1,2,3,4), 1, (1,2,3,4), 3, (3,6,9,12)),  # 3x scale
        ((1,2,4), (1,3,4), 1, (1,2,3), 2, (2,4,6)),  # asymmetric cards
    ]
    
    results = []
    for cardsA, cardsB, diff1, prizes1, diff2, prizes2 in test_cases:
        calculateEV.cache_clear()
        ev1 = calculateEV(cardsA, cardsB, diff1, prizes1, 0, "v")
        calculateEV.cache_clear()
        ev2 = calculateEV(cardsA, cardsB, diff2, prizes2, 0, "v")
        results.append((abs(ev1 - ev2) < 1e-9, ev1, ev2))
    
    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" GOPS SOLVER - COMPRESSION STRATEGY ANALYSIS")
    print(" Consolidated Test Suite")
    print("="*70)
    
    # Run tests for n=6 and n=7
    for n in [6, 7]:
        print_header(f"TESTING WITH n={n} CARDS")
        
        start = time.time()
        ev, cache = run_solver(n)
        elapsed = time.time() - start
        
        print(f"Solver: EV={ev:.6f}, time={elapsed:.2f}s, cache={len(cache):,} states")
        
        # Run all compression tests
        results = {}
        
        states, bad, reduction = test_gcd_compression(cache)
        results['GCD'] = (states, bad, reduction)
        
        states, bad, reduction = test_sorted_prizes(cache)
        results['Sorted (no idx)'] = (states, bad, reduction)
        
        states, bad, reduction = test_outcome_mapping(cache)
        results['Outcome Map'] = (states, bad, reduction)
        
        states, bad, reduction = test_compressed_mapping(cache)
        results['Compressed Map'] = (states, bad, reduction)
        
        states, bad, reduction = test_gcd_and_mapping(cache)
        results['GCD + Mapping'] = (states, bad, reduction)
        
        # Print results table
        print(f"\n{'Strategy':<20} {'States':>10} {'Invalid':>10} {'Reduction':>10} {'Valid':>8}")
        print("-" * 60)
        for name, (states, bad, reduction) in results.items():
            valid = "✓" if bad == 0 else "✗"
            print(f"{name:<20} {states:>10,} {bad:>10} {reduction:>10.2f}x {valid:>8}")
        
        # Depth analysis for outcome mapping
        if n == 7:
            print_header("OUTCOME MAPPING BY DEPTH (n=7)")
            depth_results = test_mapping_by_depth(cache)
            print(f"{'Depth':<8} {'Original':>12} {'Mapped':>12} {'Reduction':>12}")
            print("-" * 46)
            for depth in sorted(depth_results.keys()):
                orig, mapped = depth_results[depth]
                red = orig / mapped if mapped > 0 else 0
                print(f"{depth:<8} {orig:>12,} {mapped:>12,} {red:>12.2f}x")
    
    # Scale invariance test
    print_header("SCALE INVARIANCE TEST")
    scale_results = test_scale_invariance()
    all_pass = all(r[0] for r in scale_results)
    print(f"All tests passed: {'✓' if all_pass else '✗'}")
    for i, (passed, ev1, ev2) in enumerate(scale_results):
        status = "✓" if passed else "✗"
        print(f"  Case {i+1}: {status} EV1={ev1:.6f}, EV2={ev2:.6f}")
    
    # Final report
    print_header("FINAL REPORT")
    print("""
VALIDATED COMPRESSION STRATEGIES:
---------------------------------
1. GCD Normalization
   - Status: VALID but minimal benefit (~1.02x)
   - Scales (pointDiff, prizes) by their GCD
   - Preserves game-theoretic equivalence

2. Scale Invariance  
   - Status: VALID
   - Multiplying (pointDiff, prizes) by constant k preserves EV
   - Already exploited via GCD normalization

3. Outcome Mapping
   - Status: VALID with ~1.4x overall reduction
   - Maps (pointDiff, prizes) to win/lose/tie outcomes for each 3^n scenario
   - Best at shallow depths (3-4x), worst at deep depths (~1x)
   - Overhead of computing 3^n mappings may exceed benefit for large n

INVALID COMPRESSION STRATEGIES:
-------------------------------
1. Sorted Prizes (ignoring prizeIndex)
   - Status: INVALID
   - prizeIndex matters - different indices give different EVs
   - The prize revealed NOW affects strategy

2. Compressed Mapping (just counting wins/ties/losses)
   - Status: INVALID  
   - Loses strategic information
   - Same counts can have different optimal strategies

KEY INSIGHTS:
-------------
- The exponential growth is fundamental to the game structure
- State compression gives modest constant-factor improvements
- For n=13, even 2x improvement doesn't change feasibility
- May need approximate methods (MCCFR, neural networks) for full game
""")


if __name__ == "__main__":
    main()
