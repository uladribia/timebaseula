---
description: Script reference for the AirPassengers and daily-panel benchmark workflows.
---

# Scripts

## TL;DR
- Use `scripts/benchmark_airpassengers.py` for the small public reference benchmark.
- Use `scripts/prepare_nixtla_panel.py`, `scripts/benchmark_nixtla_panel.py`, and `scripts/tune_nixtla_panel_aggregated.py` on the `benchmark` branch for the daily-panel workflow.
- The published benchmark pages are now aligned with the exact commands listed below.
- All scripts are CPU-first and use Typer, Rich, Matplotlib, and rotating logs.

## Install script dependencies

```bash
uv sync --group benchmark
```

The benchmark group is intended for Python 3.12+ on non-Windows environments.

## AirPassengers reference benchmark

```bash
uv run --group benchmark python scripts/benchmark_airpassengers.py run \
  --output-markdown docs/benchmark.md \
  --output-plot docs/img/airpassengers-benchmark.png \
  --output-conformal-plot docs/img/airpassengers-timebasetrend-conformal.png \
  --neural-loss normal
```

## Daily panel workflow

### 1. Prepare a Nixtla-ready panel

```bash
uv run python scripts/prepare_nixtla_panel.py \
  --input-path data/input/internal_daily_panel.parquet.gzip \
  --output-dir data/processed/internal_daily_panel \
  --test-ratio 0.2 \
  --verbose
```

The preparation script emits:
- detailed granular series
- location-level aggregates
- item-level aggregates
- one global `total` series

### 2. Reproduce the published benchmark pages

Mixed-scope published page:

```bash
uv run --group benchmark python scripts/benchmark_nixtla_panel.py run \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-markdown docs/daily-panel-benchmark.md \
  --output-dir docs/img/daily-panel-benchmark \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile heavy \
  --max-series 256 \
  --no-include-autotheta \
  --neural-loss normal \
  --verbose
```

Aggregated-only published page:

```bash
uv run --group benchmark python scripts/benchmark_nixtla_panel.py run \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-markdown docs/daily-panel-aggregated-benchmark.md \
  --output-dir docs/img/daily-panel-aggregated-benchmark \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile normal \
  --series-scope aggregated \
  --max-series 64 \
  --tuned-config-path artifacts/tuning/aggregated/best_configs.json \
  --verbose
```

Detailed-only published page:

```bash
uv run --group benchmark python scripts/benchmark_nixtla_panel.py run \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-markdown docs/daily-panel-detailed-benchmark.md \
  --output-dir docs/img/daily-panel-detailed-benchmark \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile heavy \
  --series-scope detailed \
  --no-include-autotheta \
  --max-series 256 \
  --neural-loss normal \
  --verbose
```

### 3. Tune aggregated neural models

```bash
uv run --group benchmark python scripts/tune_nixtla_panel_aggregated.py \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-dir artifacts/tuning/aggregated \
  --tuned-config-path artifacts/tuning/aggregated/best_configs.json \
  --benchmark-markdown docs/daily-panel-aggregated-benchmark.md \
  --benchmark-dir docs/img/daily-panel-aggregated-benchmark \
  --profile heavy \
  --max-series 504 \
  --verbose
```

## Daily benchmark behavior

The daily benchmark:
- selects a manageable dense subset of series for CPU-first runs
- uses a final 28-day holdout for training and inference timing
- aggregates metrics over rolling 28-day cross-validation windows on the tail test span
- defaults to `refit=False` in cross-validation and only falls back when a model requires it
- adapts neural training iterations through `smoke`, `normal`, and `heavy` profiles
- supports mixed-scope, aggregated-only, and detailed-only workflows
- writes the effective model settings into the generated markdown report

## Metrics in the daily reports

| Metric | Meaning |
|---|---|
| `training_time_seconds` | fit time on the final 28-day holdout setup |
| `inference_time_seconds` | predict time for the final 28-day holdout |
| `avg_mae`, `median_mae` | MAE aggregated over rolling 28-day forecast tasks |
| `avg_mean_scaled_mae`, `median_mean_scaled_mae` | MAE divided by the mean target count of each task or series |
| `avg_rmse`, `median_rmse` | RMSE aggregated over rolling 28-day forecast tasks |
| `avg_smape`, `median_smape` | SMAPE aggregated over rolling 28-day forecast tasks |
| `avg_rank`, `median_rank` | within-task model ranking by MAE |
| `wins` | number of rolling forecast tasks won by each model |
