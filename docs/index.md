---
description: Overview of the TimeBaseUla package and its public API.
---

# Overview

## TL;DR
- `timebaseula` exports `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- Explicit models have simple deterministic defaults.
- Auto models follow Nixtla's native `BaseAuto` pattern.
- The repository now contains only the library code and its documentation.

## Package purpose

TimeBaseUla provides compact TimeBase-style forecasting models that plug into `NeuralForecast`.

## Public API

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit TimeBase model |
| `TimeBaseTrend` | Explicit TimeBase model with trend decomposition |
| `AutoTimeBase` | Auto-tuned wrapper for `TimeBase` |
| `AutoTimeBaseTrend` | Auto-tuned wrapper for `TimeBaseTrend` |

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
