---
description: Usage guide for explicit and auto TimeBaseUla models.
---

# Usage

## TL;DR
- Use `TimeBase` or `TimeBaseTrend` when you want explicit models with simple defaults.
- Use `AutoTimeBase` or `AutoTimeBaseTrend` when you want Nixtla-style search.
- Always fit with a non-zero `val_size`.
- For multi-series training, use NeuralForecast directly; no custom helper is needed.

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

### Default resolution

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| `period_len` with daily freq | `7` |
| `period_len` with monthly freq | `12` |
| `period_len` otherwise | `h` |
| `basis_num` | `6` |
| `moving_avg_window` on `TimeBaseTrend` | `25` |

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

These wrappers follow Nixtla's `AutoDLinear` design:
- compact search space
- Ray Tune through NeuralForecast's native auto infrastructure
- no custom dataset profiler or handwritten search loop

## Multi-series training and subset prediction

```python
subset = frame[frame["unique_id"] == "series_1"].copy()
prediction = nf.predict(df=subset)
```

Depending on the installed NeuralForecast version, `predict` may or may not return `unique_id` as an explicit column. The supported contract here is that the subset flow works without a package-specific helper.

## Troubleshooting

| Symptom | First thing to try |
|---|---|
| weak fit | increase `max_steps` |
| unstable fit | lower `learning_rate` |
| auto search is slow | reduce `num_samples` |
| too much logging | keep `gpus=0` and rely on the built-in quiet defaults in the auto wrappers |
