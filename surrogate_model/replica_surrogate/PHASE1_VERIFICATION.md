# Phase-1 Structural Verification

**Dataset:** 18-row burner dataset (`surrogate_model/data/burner_data.csv`)
**Commit:** ae273fbab4fb2a73efb35dc3df5322954817bad6
**Date:** 2026-05-26

## Checks

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | BSA phi-dependent structure (N=5 and N=50) | PASS | std(BSA_mean) = 0.181774 (N=5), 0.183939 (N=50) |
| 2 | N=5 vs N=50 BSA band shapes differ | PASS | mean abs diff of std bands = 0.00867679 |
| 3 | Reproducibility: same seeds → bit-identical predictions | PASS | bit-identical: True |
| 4 | Spot-check de-standardization round-trip | PASS | pred=[3.6582, 0.2355], actual=[3.6846, 0.2665] |

## Summary

All 4 checks PASS.

## Notes

- Phase-2 physical verification (band convergence at N=50 vs N=100 on full 1000-row dataset, BSA relative width) is explicitly out of scope and deferred.
