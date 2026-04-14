---
description: Release-style summary for the multivariate batching refresh and version 0.3.5.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.5`
- switched multi-series explicit-model training to internal joint multivariate windows through `BaseMultivariate`
- kept the public model API unchanged
- refreshed the published daily benchmark pages from strict reruns of their documented settings
- preserved the old pre-multivariate library release on `deprecated/library-v0.3.4`

## What changed

| Area | Summary |
|---|---|
| Explicit models | moved multi-series batching to `BaseMultivariate` while keeping the public constructors unchanged |
| Test coverage | added unit coverage for multivariate batching behavior and wrapper expectations |
| Benchmark evidence | reran the published mixed, aggregated, and detailed daily benchmark settings and stored the comparison evidence |
| Documentation | refreshed README, branch docs, scripts docs, release notes, and benchmark pages |
| Release metadata | bumped the package version to `0.3.5` |

## Benchmark headline

| Workflow | Headline |
|---|---|
| AirPassengers | `TimeBase` improves, while `TimeBaseTrend` regresses |
| Daily panel, mixed scope | `TimeBaseTrend` becomes the best overall model |
| Daily panel, aggregated only | `AutoTheta` remains best overall |
| Daily panel, detailed only | `TimeBaseTrend` remains best overall |

## Branch roles

| Branch | Role | Status |
|---|---|---|
| `benchmark` | canonical source branch for benchmark workflows and release preparation | active |
| `main` | curated library branch and published docs | active |
| `deprecated/library-v0.3.4` | historical pre-multivariate library snapshot | deprecated |

## Public API status

The exported package API remains:
- `TimeBase`
- `TimeBaseTrend`
- `AutoTimeBase`
- `AutoTimeBaseTrend`
