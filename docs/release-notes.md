---
description: Release-style summary for main-branch probabilistic-loss support and version 0.3.4.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.4`
- added explicit probabilistic-loss support for `TimeBase` and `TimeBaseTrend` on the release branch
- kept `main` focused on package code, tests, and curated docs without benchmark scripts
- kept reproducible benchmark and tuning workflows on the `benchmark` branch
- refreshed the user docs to describe Gaussian, Poisson, and multi-output loss support

## What changed

| Area | Summary |
|---|---|
| Explicit models | added loss-aware output adapters for multi-output NeuralForecast losses |
| Validation | added shared unit and integration coverage for distribution and quantile loss paths |
| Documentation | refreshed README, models, and usage docs for probabilistic support |
| Branch policy | kept benchmark scripts and tuning utilities off `main` while curating from `benchmark` |
| Release metadata | bumped the package version to `0.3.4` |

## Branch roles

| Branch | Role |
|---|---|
| `main` | release-oriented library branch with curated benchmark reports |
| `benchmark` | full benchmarking and tuning branch with reproducibility tooling |

## Public API status

The exported package API is:
- `TimeBase`
- `TimeBaseTrend`
- `AutoTimeBase`
- `AutoTimeBaseTrend`
