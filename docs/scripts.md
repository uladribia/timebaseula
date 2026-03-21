---
description: Reference for the repository scripts used to generate plots, evaluate synthetic scenarios, and benchmark datasets.
---

# Scripts

**TL;DR**
- All repository scripts use **Typer**, **Rich**, and rotating log files.
- Logs are written under `logs/`.
- The scripts are mainly for evaluation and documentation support.

## Available scripts

| Script | Purpose |
|---|---|
| `scripts/generate_synthetic_plot.py` | generate doc images with optional forecast overlays |
| `scripts/eval_dlinear_mae.py` | get DLinear MAE on synthetic scenarios |
| `scripts/check_forecast_mae.py` | compare naive, DLinear, TimeBase, TimeBaseTrend, MFLES |
| `scripts/benchmark_long_horizon.py` | benchmark models on ECL and Traffic |
| `scripts/generate_datasets.py` | download and aggregate long-horizon datasets |

## Generate synthetic plot

```bash
uv run --frozen python scripts/generate_synthetic_plot.py --help
```

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
uv run --frozen python scripts/check_forecast_mae.py
```

This prints a Rich table with MAE values for:

- naive last-value baseline
- `DLinear`
- `TimeBase`
- `TimeBaseTrend`
- `AutoMFLES`

## Benchmark long-horizon datasets

The benchmark script:

- downloads ECL and TrafficL from `datasetsforecast` only when missing
- stores only the daily and monthly parquet aggregates under `datasets/`
- reuses those cached aggregates on later runs
- runs all requested models on CPU
- supports `--mode daily` and `--mode monthly` with tuned defaults

### Recommended modes

| Mode | Frequency | Default horizon | Default max steps |
|---|---|---:|---:|
| `daily` | daily aggregate | 14 | 50 |
| `monthly` | monthly aggregate | 5 | 30 |
| `all` | both daily and monthly | 5 | 30 |

### Quick smoke test

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode daily \
  --n-series 5 \
  --horizon 7 \
  --max-steps 10 \
  --skip-arima \
  --output logs/benchmark_results_smoke.csv
```

### Recommended longer-horizon benchmark runs

Daily longer-horizon run without ARIMA:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode daily \
  --n-series 300 \
  --horizon 28 \
  --max-steps 200 \
  --skip-arima \
  --output logs/benchmark_results_300_daily_h28_no_arima.csv
```

Monthly longer-horizon run without ARIMA:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode monthly \
  --n-series 300 \
  --horizon 8 \
  --max-steps 100 \
  --skip-arima \
  --output logs/benchmark_results_300_monthly_h8_no_arima.csv
```

Full run including both frequencies and ARIMA:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode all \
  --output logs/benchmark_results_full_with_arima.csv
```

Default sampling policy:

| Condition | Series used |
|---|---|
| at least 300 available | 300 |
| between 200 and 299 available | all available |
| fewer than 200 available | all available |
| explicit `--n-series` | exact requested value, capped by availability |

### Models included

Always included:

- `SeasonalNaive`
- `DLinear`
- `NLinear`
- `TimeBase`
- `TimeBaseTrend`
- `AutoMFLES`

Optional:

- `AutoARIMA` unless `--skip-arima` is passed

### Reported metrics

- `MAE`
- `RMSE`
- total trainable parameters
- total training time
- total inference time

### Recent benchmark observations on this machine

For more realistic CPU runs, we used broader slices, longer horizons, and larger neural training budgets:

- daily: `--n-series 300 --horizon 28 --max-steps 200 --skip-arima`
- monthly: `--n-series 300 --horizon 8 --max-steps 100 --skip-arima`

High-level takeaways from those runs:

| Slice | Best MAE |
|---|---|
| ECL daily | `DLinear` |
| TrafficL daily | `TimeBaseTrend` |
| ECL monthly | `AutoMFLES` |
| TrafficL monthly | `AutoMFLES` |

Interpretation:

- On **daily** data, the larger training budget reduces underfitting and keeps the neural models clearly ahead of `SeasonalNaive`.
- On **TrafficL daily**, `TimeBaseTrend` becomes the strongest model in the benchmark.
- On **monthly** data, short train lengths remain the main limitation, so extra training steps do not make `TimeBase` or `TimeBaseTrend` competitive.
- `AutoMFLES` remains much slower than the neural models, but it performs well on the longer-horizon monthly setting.

ARIMA speed check on this machine for `ECL` daily with `5` series:

| Mode | Wall time |
|---|---|
| `--skip-arima` | about `10.7s` |
| full run with ARIMA | about `49.5s` |

That makes `--skip-arima` a good default for exploratory and iterative work, while full ARIMA runs are better left for overnight execution.

The benchmark script can optionally save results to CSV with `--output`.

### Generate a markdown benchmark report

```bash
uv run --frozen python scripts/benchmark_long_horizon.py report \
  --input-csv logs/benchmark_results_300_longer_horizons_no_arima.csv \
  --output-md docs/benchmark.md
```

This generates a scannable markdown page with:

- the full result table
- best-MAE-by-slice summary
- a stable source reference to the CSV used

## Logging

All scripts use rotating log files with a 5 MB limit, matching the repository guidelines.
