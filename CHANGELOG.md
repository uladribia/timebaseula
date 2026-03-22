---
description: Project changelog for TimeBaseUla releases.
---

# Changelog

## 0.2.0 - 2026-03-22

### Added
- `timebaseula.synthetic.make_synthetic_series` as a package-level synthetic data utility.
- `docs/paper-for-agents.md` with an agent-friendly markdown digest of the TimeBase paper.
- `tests/test_generate_datasets.py` to cover the standardized dataset generation CLI.
- explicit benchmark test marker and `make test-benchmark` target.

### Changed
- unified shared training logic for `TimeBase` and `TimeBaseTrend` in a common internal wrapper.
- standardized `scripts/generate_datasets.py` to use Typer, Rich, and rotating logs.
- updated synthetic plot and MAE scripts to use package utilities and recommendation helpers.
- refreshed README and MkDocs pages to reflect the current supported API and workflows.
- documented that the package is vibecoded and actively reviewed.
- separated the default fast test suite from heavier integration-oriented checks.

### Removed
- removed the custom `predict_single_series` helper from the public API.
- removed stale root planning files in favor of maintained docs under `docs/plan/`.
- stopped tracking generated log artifacts in version control.

### Fixed
- clarified and tested the supported NeuralForecast workflow for predicting a single series after multi-series training.
- resolved documentation drift around exported APIs, scripts, and implementation details.
