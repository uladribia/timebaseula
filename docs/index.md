---
description: Overview of the TimeBaseUla package, branch roles, and current release highlights on main.
---

# Overview

## TL;DR
- `timebaseula` exports `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- The explicit models now batch multi-series fits through internal joint multivariate windows.
- `main` is the curated library branch.
- `benchmark` is the canonical branch for reproducible benchmark workflows.
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
| `main` | curated release branch with publishable library code and benchmark result pages | active |
| `benchmark` | benchmarking, tuning, workflow docs, and release preparation | active |
| `deprecated/library-v0.3.4` | historical pre-multivariate library snapshot | deprecated |

Use `main` for the release-oriented package view.
Use `benchmark` for reproducible workflows and tuning.
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
| `docs/` | package documentation and curated benchmark pages |
| `docs/` | package documentation and curated benchmark pages |
| `tests/` | validation suite |
