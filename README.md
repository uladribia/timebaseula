---
description: TimeBaseUla README with installation, public API, defaults, and main-branch usage.
---

# TimeBaseUla

> Compact TimeBase-style forecasting models for NeuralForecast.

## TL;DR
- `timebaseula` is a small Python forecasting library.
- Public API: `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- The package is CPU-first and integrates with `NeuralForecast`.
- This `main` branch keeps the library, tests, and curated benchmark reports.
- Full benchmark and tuning workflows live on the `benchmark` branch.

## Branch strategy

This repository maintains two long-lived branches:

- `main`: release-oriented library branch with publishable code, curated docs, and published benchmark result pages
- `benchmark`: full benchmarking and tuning branch with scripts, workflow docs, and experiment-oriented scaffolding

If you want to reproduce or extend the benchmark workflows, use the `benchmark` branch.
If you want the library and curated benchmark write-ups, use `main`.

## Installation

```bash
git clone https://github.com/uladribia/timebaseula.git
cd timebaseula
uv sync
```

The main package now includes the dependencies needed for `AutoTimeBase` and `AutoTimeBaseTrend`.
The benchmark and tuning tooling remains on the `benchmark` branch.

## What this package provides

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit segmented-basis forecasting model |
| `TimeBaseTrend` | `TimeBase` plus a trend decomposition branch |
| `AutoTimeBase` | NeuralForecast auto-tuning wrapper for `TimeBase` |
| `AutoTimeBaseTrend` | NeuralForecast auto-tuning wrapper for `TimeBaseTrend` |

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

## Auto wrappers

Use the auto wrappers when you want NeuralForecast to search TimeBase-family hyperparameters.

```python
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

auto_timebase = AutoTimeBase(h=24, num_samples=3, cpus=1, gpus=0, backend="ray")
auto_timebasetrend = AutoTimeBaseTrend(
    h=24,
    num_samples=3,
    cpus=1,
    gpus=0,
    backend="ray",
)
```

For larger benchmark and tuning workflows, switch to the `benchmark` branch.

## Conformal prediction intervals

TimeBaseUla uses NeuralForecast's built-in conformal prediction support from
`neuralforecast.utils.PredictionIntervals`.

```python
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals
from timebaseula import TimeBaseTrend

model = TimeBaseTrend(h=24, freq="D", max_steps=100)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(
    frame,
    val_size=24,
    prediction_intervals=PredictionIntervals(n_windows=2),
)
forecast = nf.predict(level=[80, 95])
```

## Benchmark reports

This `main` branch keeps the published benchmark reports that accompany the library documentation:

- `docs/benchmark.md`
- `docs/daily-panel-benchmark.md`
- `docs/daily-panel-aggregated-benchmark.md`
- `docs/daily-panel-detailed-benchmark.md`

If you want to rerun or extend those workflows, use the `benchmark` branch.

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `docs/` | MkDocs documentation and curated benchmark reports |
| `tests/` | library-focused validation suite |
| `pyproject.toml` | package metadata and dependencies |

## Documentation

Build the docs locally with:

```bash
uv run --group docs mkdocs build --strict
```

## License

MIT. See [LICENSE](LICENSE).
