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
- This `main` branch keeps the library plus curated benchmark reports.
- Full benchmark and tuning workflows live on the `benchmark` branch.

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

The benchmark and tuning tooling is maintained on the `benchmark` branch.

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

## Benchmark reports

This `main` branch keeps the published benchmark reports that accompany the library documentation:

- AirPassengers reference benchmark: `docs/benchmark.md`
- Mixed-scope internal daily-panel benchmark: `docs/daily-panel-benchmark.md`
- Aggregated-only internal daily-panel benchmark: `docs/daily-panel-aggregated-benchmark.md`
- Detailed-only internal daily-panel benchmark: `docs/daily-panel-detailed-benchmark.md`

If you want to reproduce, extend, or rerun those benchmarks, switch to the `benchmark` branch.

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `docs/` | MkDocs documentation and curated benchmark reports |
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
