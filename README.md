---
description: TimeBaseUla README with installation, branch strategy, benchmark highlights, and package usage.
---

# TimeBaseUla

> Compact TimeBase-style forecasting models for NeuralForecast.

## TL;DR
- Public API: `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- Multi-series fits keep the standard long-format `NeuralForecast` API.
- Internally, the explicit models now train on joint multivariate windows through `BaseMultivariate`.
- `benchmark` is the canonical branch for benchmark and tuning workflows.
- `main` is the curated library branch.
- `deprecated/library-v0.3.4` preserves the pre-multivariate library release for historical reference only.

## Branch strategy

| Branch | Purpose | Status |
|---|---|---|
| `benchmark` | canonical source for benchmark workflows, tuning artifacts, and release preparation | active |
| `main` | curated release-oriented library branch and published docs | active |
| `deprecated/library-v0.3.4` | pre-multivariate library snapshot kept for historical reference | deprecated |

Use `benchmark` when you need the full daily-panel workflow.
Use `main` when you only need the publishable library surface and curated benchmark pages.
Do not start new work on `deprecated/library-v0.3.4`.

## Installation

```bash
git clone https://github.com/uladribia/timebaseula.git
cd timebaseula
uv sync
```

Benchmark tooling:

```bash
uv sync --group benchmark
```

The benchmark group is intended for Python 3.12+ on non-Windows environments.

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

## What changed in the current release candidate

- `TimeBase` and `TimeBaseTrend` now batch multi-series training through joint multivariate windows internally.
- The public constructor API stays unchanged.
- Daily benchmark docs were refreshed from strict reruns of the published benchmark settings.
- The pre-multivariate library state was preserved on `deprecated/library-v0.3.4`.

## Benchmark highlights

| Workflow | Current headline |
|---|---|
| AirPassengers reference | `TimeBase` improves to `MAE 16.8449`; `TimeBaseTrend` regresses to `MAE 21.7723` |
| Daily panel, mixed scope | `TimeBaseTrend` is now the best overall model |
| Daily panel, aggregated only | `AutoTheta` stays best on `avg_rank`; `AutoTimeBaseTrend` leads `avg_mean_scaled_mae` |
| Daily panel, detailed only | `TimeBaseTrend` remains the best overall model |

Published benchmark pages:
- `docs/benchmark.md`
- `docs/daily-panel-benchmark.md`
- `docs/daily-panel-aggregated-benchmark.md`
- `docs/daily-panel-detailed-benchmark.md`

For exact benchmark commands, see `docs/scripts.md` on the `benchmark` branch.

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
| `scripts/` | benchmark and reporting scripts on `benchmark` |
| `docs/` | MkDocs documentation |
| `tests/` | repository test suite |

## Documentation

Build the docs locally with:

```bash
uv run --group docs mkdocs build --strict
```

This benchmark refresh and release preparation were produced in an agent-assisted workflow.

## License

MIT. See [LICENSE](LICENSE).
