---
description: Project changelog for TimeBaseUla releases.
---

# Changelog

## Unreleased

### Changed
- simplified the public API to `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`
- gave the explicit models deterministic defaults so `TimeBase(h=...)` and `TimeBaseTrend(h=...)` work without extra helpers
- rebuilt the auto wrappers around NeuralForecast's native `BaseAuto` pattern
- simplified benchmark tooling to CSV and markdown outputs only

### Removed
- removed package-level synthetic helpers
- removed package-level recommendation helpers and dataset profilers
- removed synthetic benchmark and plotting scripts
- removed compatibility script aliases

## 0.2.0 - 2026-03-22

### Added
- `docs/paper-for-agents.md` with an agent-friendly markdown digest of the TimeBase paper
- `tests/test_generate_datasets.py` to cover the standardized dataset generation CLI
- explicit benchmark test marker and `make test-benchmark` target

### Changed
- unified shared training logic for `TimeBase` and `TimeBaseTrend` in a common internal wrapper
- standardized `scripts/generate_datasets.py` to use Typer, Rich, and rotating logs
- refreshed README and MkDocs pages to reflect the supported API and workflows of that release

### Removed
- removed the custom `predict_single_series` helper from the public API
- removed stale root planning files in favor of maintained docs under `docs/plan/`
- stopped tracking generated log artifacts in version control
