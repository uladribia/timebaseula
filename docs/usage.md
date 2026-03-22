---
description: Usage guide for TimeBaseUla with univariate, multi-series, and recommendation examples.
---

# Usage

**TL;DR**
- Instantiate a `TimeBase` or `TimeBaseTrend` model.
- Pass it to `NeuralForecast(models=[...], freq=...)`.
- Train with `fit(...)` and predict with `predict()`.
- For a single-series forecast after multi-series training, filter the frame and call `nf.predict(df=...)`.

## Univariate example

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 240,
        "ds": pd.date_range("2024-01-01", periods=240, freq="D"),
        "y": range(240),
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

## Multi-series example

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBaseTrend

frames = []
for idx in range(3):
    frames.append(
        pd.DataFrame(
            {
                "unique_id": [f"series_{idx}"] * 240,
                "ds": pd.date_range("2024-01-01", periods=240, freq="D"),
                "y": [idx + step for step in range(240)],
            }
        )
    )

frame = pd.concat(frames, ignore_index=True)
model = TimeBaseTrend(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
    moving_avg_window=25,
    max_steps=100,
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Predict one series after multi-series training

NeuralForecast already supports this use case directly.

```python
single_series = frame[frame["unique_id"] == "series_0"].copy()
result = nf.predict(df=single_series)
```

This repository keeps that flow as the supported approach and tests it in the integration suite.

## Automatic default selection

```python
from timebaseula import recommend_timebase_kwargs, recommend_timebase_trend_kwargs

recommended_timebase = recommend_timebase_kwargs(
    frame=frame,
    freq="D",
    horizon=24,
    max_steps=150,
)
recommended_timebase_trend = recommend_timebase_trend_kwargs(
    frame=frame,
    freq="D",
    horizon=24,
    max_steps=150,
)

model = TimeBase(h=24, **recommended_timebase)
trend_model = TimeBaseTrend(h=24, **recommended_timebase_trend)
```

## Synthetic experiment assets

The repository exposes `timebaseula.make_synthetic_series(...)` as a deterministic generator reused by tests, scripts, and docs.

### Easy scenario

![Easy synthetic series](img/synthetic_series_easy.png)

### Medium scenario

![Medium synthetic series](img/synthetic_series_medium.png)

### Hard scenario

![Hard synthetic series](img/synthetic_series_hard.png)

## Long-horizon benchmark workflow

Prepare cached aggregates:

```bash
uv run --frozen python scripts/generate_datasets.py main
```

Quick verification benchmark:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode daily \
  --n-series 5 \
  --horizon 7 \
  --max-steps 10 \
  --skip-arima \
  --output logs/benchmark_results_smoke.csv
```
