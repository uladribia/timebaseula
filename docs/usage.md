---
description: Usage guide for explicit and auto TimeBaseUla models.
---

# Usage

## TL;DR
- Use `TimeBase` or `TimeBaseTrend` for explicit models with simple defaults.
- Use `AutoTimeBase` or `AutoTimeBaseTrend` for Nixtla-style search.
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
| other `period_len` | `h` |
| `basis_num` | `6` |
| `TimeBaseTrend.moving_avg_window` | `25` |

## Auto models

```python
from neuralforecast import NeuralForecast
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

models = [
    AutoTimeBase(h=24, freq="D", num_samples=5, gpus=0),
    AutoTimeBaseTrend(h=24, freq="D", num_samples=5, gpus=0),
]

nf = NeuralForecast(models=models, freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

These wrappers:
- subclass NeuralForecast's `BaseAuto`
- use Ray Tune through NeuralForecast's native auto infrastructure
- keep a compact search space for structure and training parameters

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
| auto search is slow | reduce `num_samples` |
| too much logging | keep `gpus=0` |
