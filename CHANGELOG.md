---
description: Project changelog for TimeBaseUla releases.
---

# Changelog

## Unreleased

### Changed
- simplified the public API to `TimeBase` and `TimeBaseTrend`
- removed the auto-wrapper layer to keep the package smaller and easier to read
- refactored the shared NeuralForecast wrapper logic to reduce overlap between the explicit models
- refreshed the README and docs to match the simplified library surface

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
