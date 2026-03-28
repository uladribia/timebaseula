---
description: Release-style summary for the main-branch auto-wrapper release and version 0.3.1.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.1`
- added `AutoTimeBase` and `AutoTimeBaseTrend` to the main-branch library surface
- kept `main` focused on package code, tests, and user docs without benchmark scripts
- kept reproducible benchmark and tuning workflows on the `benchmark` branch
- refreshed the docs to describe the new auto-wrapper support

## What changed

| Area | Summary |
|---|---|
| Auto wrappers | added `AutoTimeBase` and `AutoTimeBaseTrend` for NeuralForecast auto tuning |
| Package layout | kept explicit models and auto wrappers in separate modules |
| Documentation | refreshed installation, overview, models, and usage docs for the expanded API |
| Branch policy | kept benchmark scripts and tuning utilities off `main` |
| Release metadata | bumped the package version to `0.3.1` |

## Branch roles

| Branch | Role |
|---|---|
| `main` | release-oriented library branch with curated benchmark reports |
| `benchmark` | full benchmarking and tuning branch with reproducibility tooling |

## Public API status

The exported package API is now:
- `TimeBase`
- `TimeBaseTrend`
- `AutoTimeBase`
- `AutoTimeBaseTrend`
