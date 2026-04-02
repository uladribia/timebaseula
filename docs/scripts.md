---
description: Script reference for the AirPassengers and daily-panel benchmark workflows.
---

# Scripts

## TL;DR
- The repository includes `scripts/benchmark_airpassengers.py` for the small reference benchmark.
- It also includes `scripts/prepare_nixtla_panel.py`, `scripts/benchmark_nixtla_panel.py`, and `scripts/tune_nixtla_panel_aggregated.py` for daily panel preparation, benchmarking, and aggregated-model tuning.
- Both workflows use Typer, Rich, Matplotlib, and rotating log files.
- The daily benchmark keeps a fixed 28-day horizon and uses rolling cross-validation on the tail test span.

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

The preparation script emits four kinds of series:
- detailed granular series
- location-level aggregates
- item-level aggregates
- global `total`

```bash
uv run python scripts/prepare_nixtla_panel.py \
  --input-path data/input/internal_daily_panel.parquet.gzip \
  --output-dir data/processed/internal_daily_panel \
  --test-ratio 0.2 \
  --verbose
```

Outputs:

| Path | Purpose |
|---|---|
| `data/processed/internal_daily_panel/panel.parquet` | full long-format panel with detailed granular series plus higher-level and `total` aggregates |
| `data/processed/internal_daily_panel/train.parquet` | global date-based train split |
| `data/processed/internal_daily_panel/test.parquet` | global date-based test split |
| `data/processed/internal_daily_panel/metadata.json` | split summary metadata |
| `logs/prepare_nixtla_panel.log` | rotating execution log |

### 2. Run the daily benchmark

```bash
uv run --group benchmark python scripts/benchmark_nixtla_panel.py run \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-markdown docs/daily-panel-benchmark.md \
  --output-dir docs/img/daily-panel-benchmark \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile normal \
  --max-series 256 \
  --neural-loss normal \
  --verbose
```

Aggregated-only variant:

```bash
uv run --group benchmark python scripts/benchmark_nixtla_panel.py run \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-markdown docs/daily-panel-aggregated-benchmark.md \
  --output-dir docs/img/daily-panel-aggregated-benchmark \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile normal \
  --series-scope aggregated \
  --max-series 256 \
  --neural-loss poisson \
  --verbose
```

Detailed-only internal variant:

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

The daily benchmark:
- selects a manageable dense subset of series for CPU-first runs
- uses a final 28-day holdout for training and inference timing
- aggregates metrics over rolling 28-day cross-validation windows on the tail test span
- defaults to `refit=False` in cross-validation and only falls back when a model requires it
- adjusts neural training iterations using a simple profile system: `smoke`, `normal`, or `heavy`
- supports `--neural-loss mae`, `--neural-loss normal`, and `--neural-loss poisson` for the neural benchmark models
- supports `--series-scope aggregated` to benchmark only aggregated series plus the global total
- supports `--series-scope detailed` to benchmark only the most granular series
- supports `--no-include-autotheta` when an internal or faster benchmark variant should omit that baseline
- records the effective model settings in the generated markdown report

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

The aggregated tuning workflow:
- tunes `AutoDLinear`, `AutoNLinear`, `AutoTimeBase`, and `AutoTimeBaseTrend` with NeuralForecast native auto models
- routes the TimeBase family through the package auto wrappers rather than a custom repo-local fit/predict loop
- writes the best tuned configs to JSON artifacts
- can refresh the aggregated-only benchmark using the tuned configs immediately after tuning

Outputs:

| Path | Purpose |
|---|---|
| `docs/daily-panel-benchmark.md` | markdown report with aggregate metrics and comments |
| `docs/img/daily-panel-benchmark/summary.png` | average rank, wins, inference time, and accuracy-vs-train-time |
| `docs/img/daily-panel-benchmark/distribution.png` | rolling 28-day mean-scaled MAE distribution |
| `docs/img/daily-panel-benchmark/forecast_examples.png` | zoomed holdout forecasts with nearby train context |
| `logs/benchmark_nixtla_panel.log` | rotating execution log |

## Training profiles

| Profile | Intended use | Typical neural step budget |
|---|---|---|
| `smoke` | fast workflow validation | lowest |
| `normal` | standard benchmark runs | medium |
| `heavy` | more generous neural training | highest |

The exact `max_steps` values are adapted to the selected dataset subset size and number of rolling windows, and the final values are written into the benchmark markdown report.

## Metrics in the daily report

| Metric | Meaning |
|---|---|
| `training_time_seconds` | fit time on the final 28-day holdout setup |
| `inference_time_seconds` | predict time for the final 28-day holdout |
| `avg_mae`, `median_mae` | Nixtla MAE aggregated over rolling 28-day forecast tasks |
| `avg_mean_scaled_mae`, `median_mean_scaled_mae` | MAE divided by the mean target count of each task or series |
| `avg_rmse`, `median_rmse` | Nixtla RMSE aggregated over rolling 28-day forecast tasks |
| `avg_smape`, `median_smape` | Nixtla SMAPE aggregated over rolling 28-day forecast tasks |
| `avg_rank`, `median_rank` | within-task model ranking by MAE |
| `wins` | number of rolling forecast tasks won by each model |

The statistical model set includes `AutoMFLES`, `AutoTheta`, and `Naive`.
