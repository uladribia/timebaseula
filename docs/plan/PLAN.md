# Maintenance plan

## Purpose
This folder tracks documentation-safe maintenance work. It is not an implementation promise and should not be treated as the source of truth for current features.

## Current goals
1. keep the public API aligned with NeuralForecast and Nixtla auto patterns
2. keep fast tests separate from heavy integration checks
3. keep the published library small and readable
4. keep devtools internal and simple
5. keep docs aligned with shipped behavior

## Current repository truths
- Public exports are `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- Explicit models have deterministic defaults.
- Auto wrappers rely on NeuralForecast's native `BaseAuto` infrastructure.
- Benchmark scripts are operational tools, not part of the default unit suite.

## Source of truth
For behavior, prefer:
1. package code under `timebaseula/`
2. tests under `tests/`
3. user docs under `README.md` and `docs/`
