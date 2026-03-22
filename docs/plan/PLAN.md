# Maintenance plan

## Purpose
This folder tracks documentation-safe maintenance work. It is not an implementation promise and should not be treated as the source of truth for current features.

## Current goals
1. keep the public API aligned with NeuralForecast
2. keep fast tests separate from heavy training checks
3. keep package utilities under `timebaseula/`, not under `tests/`
4. keep docs aligned with shipped behavior
5. keep agent-readable paper notes available in markdown

## Current repository truths
- Public exports are `TimeBase`, `TimeBaseTrend`, `make_synthetic_series`, and recommendation helpers.
- Single-series prediction after multi-series training is handled by NeuralForecast directly.
- Heavy fit/predict checks belong to the integration suite.
- Benchmark scripts are operational tools, not part of the default unit suite.

## Source of truth
For behavior, prefer:
1. package code under `timebaseula/`
2. tests under `tests/`
3. user docs under `README.md` and `docs/`
