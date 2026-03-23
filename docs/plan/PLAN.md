# Maintenance plan

## Purpose
This folder tracks documentation-safe maintenance work. It is not an implementation promise and should not be treated as the source of truth for current features.

## Current goals
1. keep the public API aligned with NeuralForecast usage
2. keep fast tests separate from heavy integration checks
3. keep the published library small and readable
4. keep docs aligned with shipped behavior
5. keep repository metadata and docs consistent with the tracked code

## Current repository truths
- Public exports are `TimeBase` and `TimeBaseTrend`.
- Explicit models have deterministic defaults.
- The repository is maintained as a library-first package.
- Readability is preferred over extra orchestration layers.

## Source of truth
For behavior, prefer:
1. package code under `timebaseula/`
2. tests under `tests/`
3. user docs under `README.md` and `docs/`
