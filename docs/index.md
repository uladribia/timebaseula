---
description: Overview of the TimeBaseUla package and its public API.
---

# Overview

## TL;DR
- `timebaseula` exports `TimeBase` and `TimeBaseTrend`.
- Both models plug into `NeuralForecast`.
- Defaults are deterministic and CPU-first.
- The repository is focused on a small readable library.

## Package purpose

TimeBaseUla provides compact TimeBase-style forecasting models for `NeuralForecast`.

## Public API

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit TimeBase model |
| `TimeBaseTrend` | Explicit TimeBase model with trend decomposition |

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
| `docs/` | package documentation |
| `tests/` | validation suite |
