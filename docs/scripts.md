---
description: Reference for repository scripts used for datasets, synthetic plots, evaluation, and benchmarks.
---

# Scripts

**TL;DR**
- Operational scripts use **Typer**, **Rich**, rotating log files, and **Matplotlib** for report-friendly visualizations.
- Logs are written under `logs/`.
- Benchmarking and synthetic evaluation are intentionally kept outside the default fast test suite.

## Available scripts

| Script | Purpose |
|---|---|
| `scripts/generate_datasets.py` | prepare cached benchmark datasets |
| `scripts/generate_synthetic_plot.py` | generate standalone HTML visualizations with embedded Matplotlib figures and optional forecast overlays |
| `scripts/eval_dlinear_mae.py` | get DLinear MAE on synthetic scenarios |
| `scripts/check_forecast_mae.py` | compare naive, DLinear, TimeBase, TimeBaseTrend, MFLES and emit a reusable synthetic HTML report |
| `scripts/benchmark_long_horizon.py` | benchmark models on ECL and Traffic and emit markdown or HTML reports |

## Prepare cached benchmark datasets

```bash
uv run --frozen python scripts/generate_datasets.py main
```

Force refresh:

```bash
uv run --frozen python scripts/generate_datasets.py main --force-download
```

## Generate synthetic plots

```bash
uv run --frozen python scripts/generate_synthetic_plot.py --help
```

This script now emits a standalone **HTML** visualization backed by **Matplotlib**.
Useful options include:
- `--output`
- `--title`
- `--forecast-horizon`
- `--include-reference`
- `--include-timebase`
- `--include-timebase-trend`
- `--include-mfles`

## Compare MAE on synthetic scenarios

```bash
uv run --frozen python scripts/check_forecast_mae.py main

uv run --frozen python scripts/check_forecast_mae.py report-html \
  --output-csv logs/synthetic_benchmark_results.csv \
  --output-html logs/synthetic_benchmark_report.html
```

## Benchmark long-horizon datasets

Quick smoke test:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode daily \
  --n-series 5 \
  --horizon 7 \
  --max-steps 10 \
  --skip-arima \
  --output logs/benchmark_results_smoke.csv \
  --html-report
```

Longer run without ARIMA:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode daily \
  --n-series 300 \
  --horizon 28 \
  --max-steps 200 \
  --skip-arima \
  --output logs/benchmark_results_300_daily_h28_no_arima.csv \
  --html-report
```

Generate reports from a benchmark CSV, or emit the HTML report directly from `main` with `--html-report`:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py report \
  --input-csv logs/benchmark_results_smoke.csv \
  --output-md docs/benchmark.md

uv run --frozen python scripts/benchmark_long_horizon.py report-html \
  --input-csv logs/benchmark_results_smoke.csv \
  --output-html logs/benchmark_results_smoke.html
```

## Logging

These scripts use rotating log files with a 5 MB limit:

- `logs/generate_datasets.log`
- `logs/synthetic_plot.log`
- `logs/dlinear_mae.log`
- `logs/forecast_mae_check.log`
- `logs/benchmark.log`
