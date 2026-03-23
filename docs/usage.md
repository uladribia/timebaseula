---
description: Usage guide for the explicit TimeBaseUla models.
---

# Usage

## TL;DR
- Use `TimeBase` for the compact segmented-basis model.
- Use `TimeBaseTrend` when you want an added trend branch.
- Fit with a non-zero `val_size`.
- For multi-series training, use `NeuralForecast` directly.

## Explicit models

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase, TimeBaseTrend

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 240,
        "ds": pd.date_range("2024-01-01", periods=240, freq="D"),
        "y": range(240),
    }
)

seasonal_model = TimeBase(h=24, freq="D", max_steps=100)
trend_model = TimeBaseTrend(h=24, freq="D", max_steps=100)

nf = NeuralForecast(models=[seasonal_model, trend_model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Default resolution

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| daily `period_len` | `7` |
| monthly `period_len` | `12` |
| other `period_len` | `min(max(2, h), input_size)` |
| `basis_num` | `6` |
| `TimeBaseTrend.moving_avg_window` | `25` |

## Parameter tuning quick guide

| Parameter | When to increase it | When to decrease it |
|---|---|---|
| `input_size` | when useful patterns extend further into the past | when training is slow or old history is mostly noise |
| `period_len` | when the repeated structure spans longer cycles | when the important repetition is shorter |
| `basis_num` | when the model is too rigid | when the model is overfitting or harder to interpret |
| `max_steps` | when the fit is still improving | when training already converges quickly |
| `moving_avg_window` | when the trend branch is too reactive or noisy | when the trend branch is too flat or lags behind changes |

## Understanding `moving_avg_window`

`TimeBaseTrend` first splits the signal into:
- a smoother trend component
- a residual seasonal component

The `moving_avg_window` controls how smooth that extracted trend becomes.

```python
fast_trend = TimeBaseTrend(h=24, freq="D", moving_avg_window=5)
slow_trend = TimeBaseTrend(h=24, freq="D", moving_avg_window=25)
```

Practical intuition:
- `5`: the trend follows local changes more closely
- `25`: the trend is smoother and slower, so more short-term variation stays in the TimeBase branch

Use only odd values. Even values raise a `ValueError`.

## Multi-series training and subset prediction

```python
subset = frame[frame["unique_id"] == "series_1"].copy()
prediction = nf.predict(df=subset)
```

## Troubleshooting

| Symptom | First thing to try |
|---|---|
| weak fit | increase `max_steps` |
| unstable fit | lower `learning_rate` |
| trend forecast too wiggly | increase `moving_avg_window` |
| trend forecast too flat | decrease `moving_avg_window` |
| too much logging | disable trainer logging |
