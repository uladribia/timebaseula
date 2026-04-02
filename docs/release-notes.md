---
description: Release-style summary for benchmark-branch distribution-loss support and version 0.3.4.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.4`
- added explicit probabilistic-loss support for `TimeBase` and `TimeBaseTrend`
- kept `benchmark` as the canonical source branch for benchmark workflows and release curation
- added benchmark CLI options to run Gaussian and Poisson smoke benchmarks reproducibly
- validated the benchmark branch on AirPassengers and the prepared daily panel dataset

## What changed

| Area | Summary |
|---|---|
| Explicit models | added loss-aware output adapters for multi-output NeuralForecast losses |
| Test coverage | added unit and integration coverage for distribution and quantile loss paths |
| Benchmark scripts | added `--neural-loss` support for AirPassengers and daily-panel workflows |
| Documentation | refreshed README, model docs, usage docs, and script docs for probabilistic support |
| Release metadata | bumped the package version to `0.3.4` |

## Branch roles

| Branch | Role |
|---|---|
| `main` | release-oriented library branch with curated benchmark reports |
| `benchmark` | full benchmarking and tuning branch with reproducibility tooling |

## Public API status

On the `benchmark` branch, the exported package API is:
- `TimeBase`
- `TimeBaseTrend`
- `AutoTimeBase`
- `AutoTimeBaseTrend`
