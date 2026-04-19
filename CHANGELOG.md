---
description: Project changelog for TimeBaseUla releases.
---

# Changelog

## Unreleased

## 0.3.6 - 2026-04-19

### Added
- added Hypothesis as a first-class test dependency for invariant-driven library tests
- added `tests/property_strategies.py` to centralize reusable property-based test strategies

### Changed
- converted core library invariants for decomposition, core forward contracts, defaults, factories, and explicit model shape behavior to property-based tests
- updated contributor and agent guidance to prefer Hypothesis for main-library invariants where it adds more signal than example-only tests
- refreshed release-facing documentation to describe the new property-based testing support

## 0.3.5 - 2026-04-13

### Added
- added multivariate batching coverage for the explicit TimeBase family on top of `NeuralForecast` long-format inputs
- added strict published-setting benchmark reruns for the mixed, aggregated, and detailed daily-panel pages
- added the deprecated historical branch `deprecated/library-v0.3.4` to preserve the pre-multivariate library snapshot

### Changed
- switched multi-series explicit-model training to internal joint multivariate windows through `BaseMultivariate`
- refreshed the published daily benchmark pages and benchmark workflow docs to match the exact commands used for the current reports
- updated release-oriented documentation to explain the active `benchmark` and `main` branches plus the deprecated historical branch

## 0.3.4 - 2026-04-02

### Added
- added loss-aware forecast adapters so `TimeBase` and `TimeBaseTrend` support multi-output NeuralForecast losses
- added unit and integration coverage for Gaussian, Poisson, StudentT, NegativeBinomial, Tweedie, and multi-quantile loss paths
- added benchmark CLI support for `--neural-loss` so AirPassengers and daily-panel workflows can run Gaussian and Poisson smoke benchmarks

### Changed
- updated benchmark and library docs to describe explicit probabilistic-loss support
- validated the benchmark branch with Gaussian and Poisson smoke runs on AirPassengers and the prepared daily panel dataset

## 0.3.3 - 2026-03-29

### Added
- added a local pure-Torch decomposition module for `TimeBaseTrend`
- added unit coverage for the local moving-average decomposition helpers

### Changed
- removed the direct dependency of `TimeBaseTrend` on `neuralforecast.models.dlinear.SeriesDecomp`
- documented that the local decomposition is an intentional decoupling choice that can still be reverted to the upstream helper later if needed

## 0.3.2 - 2026-03-29

### Added
- added focused unit coverage for the new internal model modules and shared factory helpers

### Changed
- split the explicit TimeBase implementation into smaller `core`, `base`, `defaults`, `config`, and `factories` modules
- reduced duplication between `TimeBase` and `TimeBaseTrend` by resolving shared explicit-model components through factory helpers
- refreshed README and library documentation to describe the smaller internal module layout

## 0.3.1 - 2026-03-27

### Added
- added `AutoTimeBase` and `AutoTimeBaseTrend` as NeuralForecast-compatible auto-tuning wrappers in `timebaseula.models.auto`
- added unit coverage for the new auto-wrapper API and aggregated tuning integration helpers

### Changed
- split the explicit model implementations and the auto-wrapper implementations into separate modules
- updated the aggregated tuning workflow to tune `AutoTimeBase` and `AutoTimeBaseTrend` through NeuralForecast native auto utilities instead of a repo-local fit/predict loop
- refreshed the aggregated benchmark artifacts and documentation after rerunning the tuning workflow on the aggregated panel

## 0.2.5 - 2026-03-23

### Added
- added `scripts/benchmark_airpassengers.py` to benchmark `AirPassengersPanel` and write a docs-ready markdown report
- added the generated benchmark page and plot to the documentation site
- added unit coverage for the AirPassengers benchmark helpers

### Changed
- simplified the public API to `TimeBase` and `TimeBaseTrend`
- removed the auto-wrapper layer to keep the package smaller and easier to read
- refactored the shared NeuralForecast wrapper logic to reduce overlap between the explicit models
- refreshed the README and docs to match the simplified library surface
- documented parameter effects more explicitly, including `moving_avg_window`
- tuned benchmark neural-model settings and published the exact reproducible configuration in the docs

### Removed
- removed `AutoTimeBase` from the public package
- removed `AutoTimeBaseTrend` from the public package
- removed the direct `ray[tune]` package dependency

## 0.2.0 - 2026-03-22

### Added
- `docs/paper-for-agents.md` with an agent-friendly markdown digest of the TimeBase paper
- explicit benchmark test marker and `make test-benchmark` target

### Changed
- unified shared training logic for `TimeBase` and `TimeBaseTrend` in a common internal wrapper
- refreshed README and MkDocs pages to reflect the supported API and workflows of that release

### Removed
- removed the custom `predict_single_series` helper from the public API
- removed stale root planning files in favor of maintained docs under `docs/plan/`
- stopped tracking generated log artifacts in version control
