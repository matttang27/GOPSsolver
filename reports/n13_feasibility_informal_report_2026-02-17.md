# N=13 State-Space Feasibility Investigation (Informal)

Date: 2026-02-17

## Purpose
Estimate whether a full `N=13` solve is tractable if we approximate mid/deep leaves, and quantify how much work remains above a cutoff (for example, instant EV oracle at `k=9` or `k=7`).

## Scope
- Objective: `win` caches (`reports/full6.evc` to `reports/full9.evc`).
- Existing solver optimizations are included implicitly because counts are read from produced `.evc` files.
- Archived code under `Zold_python_solver/` was not used.

## What We Did
1. Parsed each `.evc` and bucketed records by hand size `k = popcount(A)`.
2. Computed empirical growth from `N=6 -> 9` totals.
3. Computed a raw combinatorial reference count (no symmetry/compression/diff canonicalization):

   `raw(N,k) = C(N,k)^2 * C(N,k-1) * (N-k+1)`

4. Compared observed-to-raw ratio by diagonal `d = N-k`.
5. Added extra anchors by forward reachability with solver-identical canonicalization/compression (`scripts/tmp_count_k_states.cpp`):
   - `N=10` down to `k=7`
   - `N=11` down to `k=9`
6. Built low/mid/high `N=13` forecasts by extrapolating diagonal reduction ratios.

## Observed Cache Counts

### `full6.evc` (total 29,068)
- `k=1`: 224
- `k=2`: 3,998
- `k=3`: 16,056
- `k=4`: 8,340
- `k=5`: 450

### `full7.evc` (total 236,237)
- `k=1`: 328
- `k=2`: 8,524
- `k=3`: 71,034
- `k=4`: 125,664
- `k=5`: 29,805
- `k=6`: 882

### `full8.evc` (total 1,815,709)
- `k=1`: 431
- `k=2`: 14,736
- `k=3`: 183,108
- `k=4`: 831,592
- `k=5`: 696,410
- `k=6`: 87,864
- `k=7`: 1,568

### `full9.evc` (total 13,881,471)
- `k=1`: 544
- `k=2`: 22,410
- `k=3`: 374,073
- `k=4`: 2,911,032
- `k=5`: 7,256,250
- `k=6`: 3,089,310
- `k=7`: 225,260
- `k=8`: 2,592

## Empirical Growth
From observed totals:
- `6 -> 7`: `8.127x`
- `7 -> 8`: `7.686x`
- `8 -> 9`: `7.645x`

Interpretation: state growth is still exponential-like, but far smaller than naive tree growth.

## Additional Anchors from Reachability Counter
(`scripts/tmp_count_k_states.exe`)

### `N=10`
- `k=9`: 4,050
- `k=8`: 517,320
- `k=7`: 11,303,019

### `N=11`
- `k=10`: 6,050
- `k=9`: 1,089,945

These are useful because they anchor shallow diagonals (`d=1..3`) beyond `N=9`.

## Reduction Ratios by Diagonal (Observed / Raw)
For `d = N-k`:
- `d=1`: 0.4167, 0.4286, 0.4375, 0.4444, 0.4500 (upward trend)
- `d=2`: 0.6178, 0.6437, 0.6671, 0.6897, 0.7096 (upward trend)
- `d=3`: 0.6690, 0.7327, 0.7931, 0.8687, 0.9344 (strong upward trend)

Raw `N=13` per-layer counts (reference only):
- `k=12`: 26,364
- `k=11`: 5,220,072
- `k=10`: 233,936,560
- `k=9`: 3,289,732,875
- `k=8`: 17,053,975,224
- `k=7`: 35,371,207,872
- (all layers total raw: 98,464,247,427)

## `N=13` Forecast Scenarios
Constructed by assigning diagonal reduction ratios (`d=1..12`) consistent with observed/anchored trends.

### Low scenario
- Total: **27,901,650,750**
- `k>=10`: 226,114,713
- `k>=8`: 9,186,031,171

### Mid scenario (baseline)
- Total: **50,887,525,557**
- `k>=10`: 233,446,541
- `k>=8`: 15,131,988,785

### High scenario
- Total: **60,548,094,052**
- `k>=10`: 235,995,237
- `k>=8`: 17,004,421,647

## Key Findings
1. The existing optimizations (symmetry/compression/memoization/shortcuts) are very strong; reduction is much larger than a flat 40% claim.
2. Despite that, `N=13` full-cache size is still expected in the **tens of billions** of states.
3. Even with an instant `k=9` oracle, exact work at `k>=10` is still about **233M states** (mid), which is:
   - `233,446,541 / 13,881,471 = 16.82x` the full `N=9` cache size.
4. Even with an instant `k=7` oracle, exact `k>=8` work remains around **9.2B to 17.0B** states.

## Practical Conclusion
A leaf oracle alone is unlikely to make full `N=13` exact-equilibrium cache generation practical. It helps, but substantial additional reductions are still needed (for example: aggressive state/action pruning, approximate LP/policy solving, or solving only selected frontiers instead of full cache materialization).

## Notes / Limitations
- Forecasts are extrapolations from `N<=11` partial anchors, not exact counts.
- `full9.evc.json` metadata toggle list differs slightly from earlier files, but counts here come from `.evc` record keys directly.
- This report focuses on state-count feasibility, not final exploitability quality under approximation.
