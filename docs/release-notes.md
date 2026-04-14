---
description: Release-style summary for the multivariate batching refresh and version 0.3.5 on main.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.5`
- switched multi-series explicit-model training to internal joint multivariate windows through `BaseMultivariate`
- kept the public model API unchanged
- refreshed the curated daily benchmark pages from strict reruns of their documented settings
- preserved the old pre-multivariate library release on `deprecated/library-v0.3.4`

## What changed

| Area | Summary |
|---|---|
| Explicit models | moved multi-series batching to `BaseMultivariate` while keeping the public constructors unchanged |
| Validation | synced the shared library tests needed for the multivariate batching refresh |
| Curated benchmark pages | refreshed the mixed, aggregated, and detailed daily benchmark pages from the benchmark branch reruns |
| Branch policy | kept benchmark scripts on `benchmark` and preserved the pre-multivariate snapshot on `deprecated/library-v0.3.4` |
| Release metadata | bumped the package version to `0.3.5` |

## Branch roles

| Branch | Role | Status |
|---|---|---|
| `main` | curated library branch and published docs | active |
| `benchmark` | canonical source branch for benchmark workflows and release preparation | active |
| `deprecated/library-v0.3.4` | historical pre-multivariate library snapshot | deprecated |

## Public API status

The exported package API remains:
- `TimeBase`
- `TimeBaseTrend`
- `AutoTimeBase`
- `AutoTimeBaseTrend`
