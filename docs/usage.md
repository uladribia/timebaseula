---
description: Usage guide for TimeBaseUla with univariate, multi-series, and inference helper examples.
---

# Usage

**TL;DR**
- Instantiate a `TimeBase` or `TimeBaseTrend` model.
- Pass it to `NeuralForecast(models=[...], freq=...)`.
- Train with `fit(...)` and predict with `predict()`.

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

## Single-series helper

The package exports `predict_single_series`, intended to support focused inference after model training.

```python
from timebaseula import predict_single_series

single_series = frame[frame["unique_id"] == "series_0"].copy()
result = predict_single_series(
    model=model,
    series=single_series,
    h=24,
    input_size=48,
    freq="D",
)
```

## Choosing a model

| Use case | Recommended model |
|---|---|
| mostly seasonal or repeating patterns | `TimeBase` |
| clear trend + repeating structure | `TimeBaseTrend` |
| quick baseline comparison | use the scripts in `scripts/` |

## Practical parameter tips

| Parameter | Guidance |
|---|---|
| `period_len` | set to the natural period when known |
| `basis_num` | start small, such as `4` to `8` |
| `use_period_norm` | keep enabled unless period-wise centering hurts the series |
| `moving_avg_window` | must be odd for `TimeBaseTrend` |
| `scaler_type` | defaults to `identity` to avoid double normalization |

## Long-horizon benchmark workflow

This repository includes a CPU benchmark for aggregated ECL and TrafficL data.

### Prepare cached aggregates

```bash
uv run --frozen python scripts/benchmark_long_horizon.py prepare-data
```

This creates and reuses only these files under `datasets/`:

- `ecl_daily.parquet`
- `ecl_monthly.parquet`
- `trafficl_daily.parquet`
- `trafficl_monthly.parquet`

### Run a quick verification benchmark

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --n-series 5 \
  --horizon 7 \
  --test-size 7 \
  --max-steps 10 \
  --skip-arima \
  --output logs/benchmark_results_smoke.csv
```

### Run the overnight benchmark

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --output logs/benchmark_results_full.csv
```

By default, the script aims for a broad slice of the available series, using up to 300 and at least 200 when available.

## Synthetic experiment assets in this repo

The repository includes a deterministic synthetic generator in `tests/utils/synthetic_series.py` used by the evaluation scripts. The generated scenarios are represented in the docs images below.

### Easy scenario

![Easy synthetic series](img/synthetic_series_easy.png)

### Medium scenario

![Medium synthetic series](img/synthetic_series_medium.png)

### Hard scenario

![Hard synthetic series](img/synthetic_series_hard.png)
