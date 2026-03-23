---
description: Release notes covering the benchmark and packaging cleanup in the 0.2.2 release.
---

# Release notes

**TL;DR**
- `timebaseula/` is now cleanly separated from internal benchmark and reporting tooling.
- Benchmark CLIs use consistent `benchmark_*` names and keep daily and monthly long-horizon runs fully separate.
- Benchmarks now prefer native Nixtla APIs, joint model execution, and native baselines.
- ARIMA has been removed from scripts, docs, tests, and reports.
- Full-budget benchmark runs show `DLinear` leading on synthetic data, `SeasonalNaive` leading on the custom dataset, and `NLinear` leading most long-horizon runs.

## Highlights

### Library and packaging
- kept the publishable library focused on `timebaseula/`
- moved benchmark and reporting scaffolding into internal `devtools/`
- kept benchmark-only dependencies out of the shipped library package

### Benchmark workflow
- standardized canonical script names to `benchmark_*`
- kept compatibility wrappers for older script names
- split long-horizon daily and monthly execution into separate runs and separate report artifacts
- aligned synthetic, custom, and long-horizon runners around native Nixtla execution paths

### Evaluation and reporting
- used joint `NeuralForecast` and `StatsForecast` execution where possible
- used native `Naive`, `SeasonalNaive`, and `AutoMFLES` baselines
- fixed long-horizon reporting so persisted holdouts and rendered prediction spans match the evaluated forecast horizon
- kept HTML reports aligned with the actual evaluated windows

### Removed
- removed ARIMA from code paths, benchmark scripts, tests, docs, and report styling

## Full-budget benchmark snapshot

| Benchmark | Best observed model | Notes |
|---|---|---|
| synthetic | `DLinear` | won easy, medium, and hard scenarios at `--max-steps 200` |
| custom dataset | `SeasonalNaive` | narrowly led `MFLES` and `AutoTimeBase` on overall MAE |
| long horizon daily | `NLinear` / `DLinear` | `NLinear` won `ECL`, `DLinear` won `TrafficL` |
| long horizon monthly | `NLinear` | won both `ECL` and `TrafficL` |

## Operational notes
- benchmark CLIs default to `--no-refit`
- long-horizon `--n-series` is an exact subset size
- omitting `--n-series` uses the benchmark default slice
- passing `--n-series 0` selects zero series and is not a valid real benchmark run
