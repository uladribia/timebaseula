---
description: Usage guide for the explicit and auto TimeBaseUla models.
---

# Usage

## TL;DR
- Use `TimeBase` for the compact segmented-basis model.
- Use `TimeBaseTrend` when you want an added trend branch.
- Use `AutoTimeBase` or `AutoTimeBaseTrend` when you want NeuralForecast to tune the TimeBase family.
- The explicit models support point, quantile, and distribution losses.
- Fit with a non-zero `val_size`.

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

## Probabilistic example

```python
from neuralforecast.losses.pytorch import DistributionLoss

probabilistic_model = TimeBase(
    h=24,
    freq="D",
    loss=DistributionLoss("Poisson", level=[80, 95]),
)
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

## Auto wrappers

Use the auto wrappers when you want to search over TimeBase-family hyperparameters with NeuralForecast.

```python
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

auto_models = [
    AutoTimeBase(h=24, num_samples=3, cpus=1, gpus=0, backend="ray"),
    AutoTimeBaseTrend(h=24, num_samples=3, cpus=1, gpus=0, backend="ray"),
]
```

The benchmark and tuning scripts that orchestrate larger searches still live on the `benchmark` branch.

## Conformal prediction intervals

NeuralForecast already provides conformal prediction intervals through
`neuralforecast.utils.PredictionIntervals`, and both `TimeBase` and `TimeBaseTrend` work with that API directly.

```python
from neuralforecast.utils import PredictionIntervals

interval_nf = NeuralForecast(models=[trend_model], freq="D")
interval_nf.fit(
    frame,
    val_size=24,
    prediction_intervals=PredictionIntervals(n_windows=2),
)
interval_forecast = interval_nf.predict(level=[80, 95])
```

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
