---
description: Project changelog for TimeBaseUla releases.
---

# Changelog

## Unreleased

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
