---
description: TimeBaseUla README with installation, branch strategy, benchmark highlights, and main-branch usage.
---

# TimeBaseUla

> Compact TimeBase-style forecasting models for NeuralForecast.

## TL;DR
- Public API: `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- Multi-series fits keep the standard long-format `NeuralForecast` API.
- Internally, the explicit models now train on joint multivariate windows through `BaseMultivariate`.
- This `main` branch is the curated library branch.
- Full benchmark and tuning workflows live on `benchmark`.
- `deprecated/library-v0.3.4` preserves the pre-multivariate library release for historical reference only.

## Branch strategy

| Branch | Purpose | Status |
|---|---|---|
| `main` | curated release-oriented library branch and published benchmark pages | active |
| `benchmark` | canonical source for benchmark workflows, tuning artifacts, and release preparation | active |
| `deprecated/library-v0.3.4` | pre-multivariate library snapshot kept for historical reference | deprecated |

Use `main` when you want the publishable library surface.
Use `benchmark` when you need benchmark scripts, tuning workflows, or reproducible benchmark regeneration.
Do not start new work on `deprecated/library-v0.3.4`.

## Installation

```bash
git clone https://github.com/uladribia/timebaseula.git
cd timebaseula
uv sync
```

The main package includes the dependencies needed for `AutoTimeBase` and `AutoTimeBaseTrend`.
Benchmark tooling remains on the `benchmark` branch.

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

## What changed in this release

- `TimeBase` and `TimeBaseTrend` now batch multi-series training through joint multivariate windows internally.
- The library test suite now uses Hypothesis for invariant-driven unit tests, with shared strategies under `tests/property_strategies.py`.
- The public constructor API stays unchanged.
- The published daily benchmark pages were refreshed from strict reruns of their documented settings.
- The old pre-multivariate library state was preserved on `deprecated/library-v0.3.4`.

## Curated benchmark highlights

| Workflow | Current headline |
|---|---|
| AirPassengers reference | `TimeBase` improves to `MAE 16.8449`; `TimeBaseTrend` regresses to `MAE 21.7723` |
| Daily panel, mixed scope | `TimeBaseTrend` is now the best overall model |
| Daily panel, aggregated only | `AutoTheta` stays best on `avg_rank`; `AutoTimeBaseTrend` leads `avg_mean_scaled_mae` |
| Daily panel, detailed only | `TimeBaseTrend` remains the best overall model |

Published benchmark pages kept on `main`:
- `docs/benchmark.md`
- `docs/daily-panel-benchmark.md`
- `docs/daily-panel-aggregated-benchmark.md`
- `docs/daily-panel-detailed-benchmark.md`

If you want to rerun or extend those workflows, switch to the `benchmark` branch.

## Data format

The package expects the standard `NeuralForecast` long format:

| Column | Meaning |
|---|---|
| `unique_id` | series identifier |
| `ds` | timestamp |
| `y` | target value |

When multiple `unique_id` values are fit together, the public API remains long-format while the explicit models internally build joint multivariate windows over the active series.

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `timebaseula/models/core.py` | pure Torch TimeBase core |
| `timebaseula/models/decomposition.py` | pure Torch TimeBaseTrend decomposition |
| `timebaseula/models/base.py` | shared NeuralForecast wrapper logic |
| `docs/` | MkDocs documentation and curated benchmark pages |
| `tests/` | library-focused validation suite |
| `tests/` | library-focused validation suite |

## Testing

Run the default fast suite with:

```bash
make test
```

The unit suite now mixes example-based tests with property-based tests powered by
[Hypothesis](https://hypothesis.readthedocs.io/). Shared property strategies live in
`tests/property_strategies.py` so model test files can stay focused on behavioral
contracts rather than generator boilerplate.

## Documentation

Build the docs locally with:

```bash
uv run --group docs mkdocs build --strict
```

This release curation was produced in an agent-assisted workflow.

## License

MIT. See [LICENSE](LICENSE).
