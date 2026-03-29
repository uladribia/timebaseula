---
description: Overview of the TimeBaseUla package and its public API.
---

# Overview

## TL;DR
- `timebaseula` exports `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- The explicit models plug into `NeuralForecast` through a shared wrapper layer.
- Defaults are deterministic and CPU-first.
- `TimeBaseTrend` intentionally uses a local pure-Torch decomposition helper instead of DLinear internals.
- `main` is the release-oriented library branch, while `benchmark` carries the full benchmarking and tuning workflows.

## Package purpose

TimeBaseUla provides compact TimeBase-style forecasting models for `NeuralForecast`.

## Public API

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit TimeBase model |
| `TimeBaseTrend` | Explicit TimeBase model with trend decomposition |
| `AutoTimeBase` | Auto-tuning wrapper for `TimeBase` |
| `AutoTimeBaseTrend` | Auto-tuning wrapper for `TimeBaseTrend` |

## Quick example

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 120,
        "ds": pd.date_range("2024-01-01", periods=120, freq="D"),
        "y": range(120),
    }
)

model = TimeBase(h=12, freq="D")
nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=12)
forecast = nf.predict()
```

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `timebaseula/models/core.py` | pure Torch TimeBase core |
| `timebaseula/models/decomposition.py` | pure Torch TimeBaseTrend decomposition |
| `timebaseula/models/base.py` | shared NeuralForecast wrapper logic |
| `docs/` | package documentation and curated benchmark pages |
| `tests/` | validation suite |
