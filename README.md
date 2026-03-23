---
description: TimeBaseUla README with installation, public API, usage, and internal benchmarking notes.
---

# TimeBaseUla

> Compact TimeBase-style forecasting models for NeuralForecast, with simple defaults and Nixtla-style auto wrappers.

> **Note:** the latest simplification pass in this repository was agent-made.

## TL;DR
- Public API: `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, `AutoTimeBaseTrend`.
- `TimeBase` and `TimeBaseTrend` now have deterministic defaults, so `TimeBase(h=24)` works.
- `AutoTimeBase` and `AutoTimeBaseTrend` are thin wrappers over Nixtla's native `BaseAuto` pattern.
- The publishable package is `timebaseula/`.
- Internal benchmark tooling is kept under `devtools/` and exposed through thin `scripts/` wrappers.
- Benchmarks now produce simple CSV and markdown outputs only.

## Installation

### From PyPI

```bash
pip install timebaseula
```

### From source

```bash
git clone https://github.com/dribia/timebaseula.git
cd timebaseula
uv sync
```

## Public API

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit TimeBase model with simple defaults |
| `TimeBaseTrend` | TimeBase plus trend decomposition |
| `AutoTimeBase` | Nixtla-style auto-tuned wrapper for `TimeBase` |
| `AutoTimeBaseTrend` | Nixtla-style auto-tuned wrapper for `TimeBaseTrend` |

## Quickstart

### Explicit model

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

### Auto model

```python
from neuralforecast import NeuralForecast
from timebaseula import AutoTimeBase

model = AutoTimeBase(h=24, freq="D", num_samples=5, gpus=0)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Default behavior

### `TimeBase` and `TimeBaseTrend`
- no dataset profiler is required
- `input_size` defaults to `max(2 * h, 8)`
- if `freq` is daily, `period_len` defaults to `7`
- if `freq` is monthly, `period_len` defaults to `12`
- otherwise `period_len` defaults to the horizon

### `AutoTimeBase` and `AutoTimeBaseTrend`
- follow the same high-level design as Nixtla's `AutoDLinear`
- use Ray Tune through NeuralForecast's native auto infrastructure
- keep a compact search space for `input_size`, `period_len`, `basis_num`, and training hyperparameters

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `devtools/` | internal benchmark helpers |
| `scripts/` | thin Typer wrappers over `devtools/` |
| `tests/unit/library/` | fast library tests |
| `tests/unit/devtools/` | fast internal-tooling tests |
| `tests/integration/` | heavier NeuralForecast integration checks |

## Development workflow

Fast quality gates:

```bash
make format
make lint
make test
```

Integration checks are worth running for this repository when model construction, auto wrappers, or NeuralForecast interoperability change:

```bash
make test-integration
```

## Internal benchmark scripts

```bash
uv run --frozen python scripts/generate_datasets.py main
uv run --frozen python scripts/benchmark_long_horizon.py run --mode daily --auto-preset smoke --output-pdf logs/benchmark_long_horizon_daily.pdf
uv run --frozen python scripts/benchmark_custom.py --auto-preset smoke --output-pdf logs/custom_dataset_benchmark/report.pdf
```

Both benchmark entrypoints now write:
- a CSV leaderboard with `mae`, `rmse`, `rmae`, `params`, and per-model `execution_time`
- a markdown report with metric notes, a data summary, and representative forecast plots
- a plot directory with train/test/prediction comparisons for selected series
- an optional PDF export of the markdown report via `--output-pdf` rendered with headless Chrome and embedded plot images

Benchmark neural comparisons now use `AutoDLinear`, `AutoNLinear`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
The auto wrappers now run a real Ray Tune search space constrained to CPU-safe benchmark settings across `AutoDLinear`, `AutoNLinear`, `AutoTimeBase`, and `AutoTimeBaseTrend`, and `--auto-num-samples` controls how many auto trials are sampled.
Benchmark cross-validation always runs with `refit=True`; the CLIs no longer expose a refit toggle.

## License

MIT. See [LICENSE](LICENSE).
- `thorough`: `max_steps=20`, `auto_num_samples=4`, tuned to stay around ~5 CPU minutes
You can still override `--max-steps` and `--auto-num-samples` explicitly.
Benchmark cross-validation always runs with `refit=True`; the CLIs no longer expose a refit toggle.

## License

MIT. See [LICENSE](LICENSE).
