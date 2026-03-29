---
description: TimeBaseUla README with installation, public API, defaults, and package usage.
---

# TimeBaseUla

> Compact TimeBase-style forecasting models for NeuralForecast.

## TL;DR
- `timebaseula` is a small Python forecasting library.
- Public API: `TimeBase` and `TimeBaseTrend`.
- The package is CPU-first and integrates with `NeuralForecast`.
- Install and use it from a source checkout.
- The repository also includes benchmark and tuning scripts for `AirPassengersPanel` and daily panel workflows.

## Branch strategy

This repository maintains two long-lived branches:

- `main`: release-oriented library branch with publishable code, curated docs, and published benchmark result pages
- `benchmark`: full benchmarking and tuning branch with scripts, workflow docs, and experiment-oriented scaffolding

If you want to reproduce or extend the internal anonymized benchmarks, use the `benchmark` branch.
If you want the library and the curated benchmark write-up that accompanies a release, use `main`.

## Installation

### From source

```bash
git clone https://github.com/uladribia/timebaseula.git
cd timebaseula
uv sync
```

### Benchmark dependencies

```bash
uv sync --group benchmark
```

The benchmark tooling is intended for Python 3.12+ on non-Windows environments.

## What this package provides

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit segmented-basis forecasting model |
| `TimeBaseTrend` | `TimeBase` plus a trend decomposition branch |

## Quickstart

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 200,
        "ds": pd.date_range("2024-01-01", periods=200, freq="D"),
        "y": range(200),
    }
)

model = TimeBase(h=24, freq="D", max_steps=100)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Conformal prediction intervals

TimeBaseUla uses NeuralForecast's built-in conformal prediction support from
`neuralforecast.utils.PredictionIntervals`. No custom interval code is needed in
`TimeBase` or `TimeBaseTrend`.

```python
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals
from timebaseula import TimeBaseTrend

model = TimeBaseTrend(h=24, freq="D", max_steps=100)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(
    frame,
    val_size=24,
    prediction_intervals=PredictionIntervals(
        n_windows=2,
        method="conformal_distribution",
    ),
)
forecast = nf.predict(level=[80, 95])
```

This adds columns such as `TimeBaseTrend-lo-80` and `TimeBaseTrend-hi-80` to the
forecast output.

The expected data format is the standard `NeuralForecast` long format:

| Column | Meaning |
|---|---|
| `unique_id` | series identifier |
| `ds` | timestamp |
| `y` | target value |

## Default behavior

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| daily `period_len` | `7` |
| monthly `period_len` | `12` |
| other `period_len` | `min(max(2, h), input_size)` |
| `basis_num` | `6` |
| `moving_avg_window` (`TimeBaseTrend`) | `25` |

## What the main parameters do

| Parameter | Effect |
|---|---|
| `h` | Forecast horizon. |
| `input_size` | Amount of history used by the model. |
| `period_len` | Length of each repeated segment used by the TimeBase basis. |
| `basis_num` | Number of basis components used to reconstruct the forecast. |
| `use_period_norm` | Whether each segment is normalized before basis learning. |
| `max_steps` | Maximum number of training steps. |
| `learning_rate` | Optimizer step size. |
| `moving_avg_window` | Only for `TimeBaseTrend`. Controls how smooth the extracted trend is. Larger odd values produce a smoother, slower trend; smaller odd values make the trend react faster. |

## Model overview

### `TimeBase`
- splits the input window into temporal segments
- projects segments to a compact basis
- maps that basis back to future segments

### `TimeBaseTrend`
- decomposes the input into seasonal and trend parts
- sends the seasonal part through the TimeBase branch
- forecasts the trend with a linear head
- sums both forecasts at the end

For full parameter guidance, see `docs/models.md` and `docs/usage.md`.

## Benchmark scripts

This repository includes benchmark workflows plus an aggregated-model tuning workflow.

### AirPassengers reference benchmark

```bash
uv run --group benchmark python scripts/benchmark_airpassengers.py run \
  --output-markdown docs/benchmark.md \
  --output-plot docs/img/airpassengers-benchmark.png
```

### Daily panel workflow for an internal anonymized dataset

1. Prepare a Nixtla-ready panel with detailed and aggregated `unique_id`, `ds`, and `y` series:

```bash
uv run python scripts/prepare_nixtla_panel.py \
  --input-path data/input/internal_daily_panel.parquet.gzip \
  --output-dir data/processed/internal_daily_panel \
  --test-ratio 0.2
```

2. Run the CPU-first daily benchmark with a fixed 28-day forecast horizon:

```bash
uv run --group benchmark python scripts/benchmark_nixtla_panel.py \
  --input-path data/processed/internal_daily_panel/panel.parquet \
  --output-markdown docs/daily-panel-benchmark.md \
  --output-dir docs/img/daily-panel-benchmark \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile normal \
  --max-series 256
```

The preparation step now includes:
- detailed granular series
- location-level aggregates
- item-level aggregates
- one global `total` series

The daily benchmark:
- compares `TimeBase`, `TimeBaseTrend`, `NLinear`, `DLinear`, `AutoMFLES`, `AutoTheta`, and `Naive`
- can also benchmark tuned `AutoTimeBase`, `AutoTimeBaseTrend`, `AutoNLinear`, and `AutoDLinear` variants from a tuning artifact
- supports mixed-scope, aggregated-only, and detailed-only internal benchmark variants
- measures training and inference time on the final 28-day holdout
- aggregates MAE, mean-scaled MAE, RMSE, SMAPE, rank, and wins over rolling 28-day cross-validation windows
- adapts neural training iterations to dataset size and a user-selected profile: `smoke`, `normal`, or `heavy`
- writes a markdown report plus Matplotlib plots, including the effective iteration settings used in the run
- publishes the mixed-scope workflow at `docs/daily-panel-benchmark.md`
- publishes the aggregated-only workflow at `docs/daily-panel-aggregated-benchmark.md`
- publishes the detailed-only internal workflow at `docs/daily-panel-detailed-benchmark.md`

The aggregated tuning workflow:
- lives in `scripts/tune_nixtla_panel_aggregated.py`
- uses NeuralForecast native auto models for `DLinear`, `NLinear`, `AutoTimeBase`, and `AutoTimeBaseTrend`
- tunes the TimeBase family through the package auto wrappers instead of a repo-local fit/predict loop
- writes reusable tuned config JSON artifacts under `artifacts/tuning/`

Note: this benchmark-branch update was prepared in an agent-assisted change.

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `timebaseula/models/core.py` | pure Torch TimeBase core |
| `timebaseula/models/decomposition.py` | pure Torch TimeBaseTrend decomposition |
| `timebaseula/models/base.py` | shared NeuralForecast wrapper logic |
| `timebaseula/models/factories.py` | shared explicit-model factories |
| `timebaseula/models/timebase.py` | public explicit model wrappers |
| `scripts/` | benchmark and reporting scripts |
| `docs/` | MkDocs documentation |
| `tests/` | repository test suite |
| `pyproject.toml` | package metadata and dependencies |

## Documentation

The documentation site covers:
- `docs/index.md`
- `docs/install.md`
- `docs/usage.md`
- `docs/models.md`
- `docs/benchmark.md`
- `docs/release-notes.md`
- `docs/paper-for-agents.md`
- `docs/references.md`

Build the docs locally with:

```bash
uv run --group docs mkdocs build --strict
```

## License

MIT. See [LICENSE](LICENSE).
kdocs build --strict
```

## License

MIT. See [LICENSE](LICENSE).
references.md`

Build the docs locally with:

```bash
uv run --group docs mkdocs build --strict
```

## License

MIT. See [LICENSE](LICENSE).
kdocs build --strict
```

## License

MIT. See [LICENSE](LICENSE).
