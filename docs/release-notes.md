---
description: Release-style summary for the property-based testing refresh and version 0.3.6 on main.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.6`
- added Hypothesis-backed property tests for the main library invariants
- centralized reusable property strategies in `tests/property_strategies.py`
- kept example-based tests for public API facts and heavy integration behavior
- preserved the public model API unchanged

## What changed

| Area | Summary |
|---|---|
| Test coverage | added Hypothesis-backed invariant coverage for decomposition, core contracts, defaults, factories, and explicit model forward behavior |
| Strategy plumbing | centralized reusable property strategies in `tests/property_strategies.py` |
| Documentation | refreshed README, contributor guidance, agent instructions, release notes, and property-testing documentation |
| Public API | kept the exported package surface unchanged |
| Release metadata | bumped the package version to `0.3.6` |

## Testing headline

| Area | Headline |
|---|---|
| Pure model helpers | now validated over bounded input domains instead of one-off examples only |
| Explicit model wrappers | retain example-based API tests and add property-based contract checks |
| Integration layer | remains example-based to keep runtime bounded and failure modes readable |

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
