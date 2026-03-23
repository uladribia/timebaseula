---
description: TimeBaseUla README with installation, usage, testing, benchmarks, and documentation notes.
---

# TimeBaseUla

> Lightweight TimeBase-style forecasting models for NeuralForecast, built for a CPU-first workflow.

> **Disclosure:** this package has been **vibecoded** and should be treated like an actively reviewed research-to-library port.

**TL;DR**
- Install with `uv sync` for development or `pip install timebaseula` for usage.
- Main exports: `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, `AutoTimeBaseTrend`, `make_synthetic_series`.
- NeuralForecast already supports forecasting a filtered subset of series after multi-series training.
- Visual reports and generated charts in this repo prefer **Matplotlib**, including HTML reports with embedded static figures.
- The publishable package is `timebaseula`; benchmark and reporting scaffolding lives in internal `devtools/` modules and thin `scripts/` wrappers.
- Fast tests are unit-only; heavier training checks live under integration tests.
- Documentation site: <https://dribia.github.io/timebaseula>

<p align="center">
  <img src="docs/img/logo_dribia_blau_cropped.png" alt="TimeBaseUla logo" width="320">
</p>

## What this library is

TimeBaseUla is a compact implementation of the **TimeBase** forecasting idea, adapted to work with [Nixtla NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/).

| Object | Purpose |
|---|---|
| `TimeBase` | Basis-based segment forecaster |
| `TimeBaseTrend` | TimeBase seasonal branch plus linear trend head |
| `make_synthetic_series` | Deterministic synthetic generator used by tests, scripts, and docs |

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

## Quickstart

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

frame = pd.DataFrame(
    {
        "unique_id": "series_1",
        "ds": pd.date_range("2024-01-01", periods=200, freq="D"),
        "y": range(200),
    }
)

model = TimeBase(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
    max_steps=100,
    learning_rate=1e-3,
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Multi-series training, single-series forecast

You do **not** need a package helper for this. Train on multiple series, then ask NeuralForecast to predict only the subset you want.

```python
subset = frame[frame["unique_id"] == "series_1"].copy()
prediction = nf.predict(df=subset)
```

This repository includes an integration test for that flow.

## Automatic defaults

```python
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

model = AutoTimeBase(h=24, freq="D", max_steps=150)
trend_model = AutoTimeBaseTrend(h=24, freq="D", max_steps=150)
```

If you want recommendations without the auto-search wrapper, you can still use:

```python
from timebaseula import recommend_timebase_kwargs, recommend_timebase_trend_kwargs

model = TimeBase(
    h=24,
    **recommend_timebase_kwargs(
        frame,
        freq="D",
        horizon=24,
        max_steps=150,
        include_iteration_recommendation=True,
    ),
)
trend_model = TimeBaseTrend(
    h=24,
    **recommend_timebase_trend_kwargs(
        frame,
        freq="D",
        horizon=24,
        max_steps=150,
        include_iteration_recommendation=True,
    ),
)
```

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `devtools/` | internal benchmark, reporting, and dataset helpers |
| `scripts/` | thin Typer CLI wrappers over `devtools/` |
| `tests/unit/library/` | unit tests for the shipped library |
| `tests/unit/devtools/` | unit tests for internal validation tooling |
| `tests/integration/` | heavier NeuralForecast compatibility checks |

Only `timebaseula/` is included in built distributions.

## Development workflow

Fast quality gates:

```bash
make format
make lint
make test
```

Heavier checks:

```bash
make test-integration
make test-benchmark
```

Notes:
- `make test` runs the fast suite only.
- integration tests cover actual NeuralForecast fitting behavior.
- benchmark-oriented checks are separated from the default suite to keep feedback fast.

## Benchmarking

Prepare cached aggregates:

```bash
uv run --frozen python scripts/generate_datasets.py main
```

Quick smoke benchmark:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py run \
  --mode daily \
  --n-series 5 \
  --horizon 7 \
  --max-steps 10 \
  --skip-arima \
  --output logs/benchmark_results_smoke.csv
```

Generate benchmark reports from persisted outputs. Keep daily and monthly runs in separate CSV and HTML files:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py report \
  --input-csv logs/benchmark_results_smoke.csv \
  --output-md docs/benchmark.md

uv run --frozen python scripts/benchmark_long_horizon.py report-html \
  --input-csv logs/benchmark_results_smoke.csv \
  --output-html logs/benchmark_results_smoke.html

uv run --frozen python scripts/benchmark_synthetic.py run \
  --max-steps 20 \
  --output-csv logs/synthetic_benchmark_results.csv

uv run --frozen python scripts/benchmark_synthetic.py report-html \
  --input-csv logs/synthetic_benchmark_results.csv \
  --output-html logs/synthetic_benchmark_report.html
```

Both benchmark scripts persist report inputs next to the CSV by default, so HTML can be regenerated without rerunning model training. Benchmark entrypoints now use a consistent `benchmark_*` naming scheme, with older script names kept as compatibility aliases. Long-horizon daily and monthly runs are intentionally separated and must be executed independently so splitting, holdouts, and report artifacts are never mixed across frequencies. The benchmark CLIs also keep fixed search budgets and do not use the current iteration auto-suggestion helper.

The HTML reports now share the same tabbed structure used by the custom-dataset benchmark, so long-horizon, synthetic, and custom runs are easier to compare side by side. Representative-series tabs consistently prioritize:

- the longest series
- the highest-variance series
- the strongest-trend series
- two additional random series when the slice contains at least five series

## Documentation highlights

- `docs/usage.md`: NeuralForecast usage patterns, validation tracking, and convergence troubleshooting
- `docs/models.md`: model structure, assumptions, and tuning notes
- `docs/scripts.md`: script reference and logging behavior
- `docs/paper-for-agents.md`: expanded markdown paper digest for LLMs and human readers
- `docs/references.md`: original PDF and bibliographic reference

## Troubleshooting training quality

If a model seems weak, check convergence before blaming the architecture.

Typical signals:
- **underfitting**: both train and validation losses stay high, and validation is still improving at the end
- **overfitting**: train loss keeps dropping while validation gets worse after an earlier best point
- **non-convergence**: losses oscillate, jump, or validation is best very early and then drifts

Recommended first actions:
- use `recommend_timebase_kwargs(...)` or `recommend_timebase_trend_kwargs(...)`
- fit with a non-zero `val_size`
- log validation metrics
- keep the best validation checkpoint instead of only the final weights
- reduce learning rate before making the model larger

See `docs/usage.md` for concrete code examples using:
- `CSVLogger`
- `ModelCheckpoint`
- `EarlyStopping`
- best-checkpoint evaluation with NeuralForecast and TimeBaseUla

## Paper reference

The original paper PDF is bundled as `docs/huang25az.pdf`. A readable markdown digest is available at `docs/paper-for-agents.md`.

## License

MIT. See [LICENSE](LICENSE).
