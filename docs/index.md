---
description: Overview of the TimeBaseUla package, branch roles, and current release highlights.
---

# Overview

## TL;DR
- `timebaseula` exports `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- The explicit models now batch multi-series fits through internal joint multivariate windows.
- `benchmark` is the canonical branch for reproducible benchmark workflows.
- `main` is the curated library branch.
- `deprecated/library-v0.3.4` preserves the pre-multivariate library release and is deprecated.

## Package purpose

TimeBaseUla provides compact TimeBase-style forecasting models for `NeuralForecast`.

## Public API

| Object | Purpose |
|---|---|
| `TimeBase` | explicit segmented-basis model |
| `TimeBaseTrend` | segmented-basis model with a trend decomposition branch |
| `AutoTimeBase` | NeuralForecast auto-tuning wrapper for `TimeBase` |
| `AutoTimeBaseTrend` | NeuralForecast auto-tuning wrapper for `TimeBaseTrend` |

## Branch roles

| Branch | Role | Status |
|---|---|---|
| `benchmark` | benchmarking, tuning, workflow docs, and release preparation | active |
| `main` | curated release branch with publishable library code and benchmark result pages | active |
| `deprecated/library-v0.3.4` | historical pre-multivariate library snapshot | deprecated |

Do benchmark and tuning work on `benchmark`.
Use `main` for the release-oriented library view.
Keep `deprecated/library-v0.3.4` read-only.

## Current benchmark highlights

| Benchmark | Headline |
|---|---|
| AirPassengers | `TimeBase` improves; `TimeBaseTrend` regresses |
| Daily panel, mixed scope | `TimeBaseTrend` is best overall after the strict rerun |
| Daily panel, aggregated only | `AutoTheta` remains best overall |
| Daily panel, detailed only | `TimeBaseTrend` remains best overall |

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
| `docs/` | package documentation |
| `tests/` | validation suite |
