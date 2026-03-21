# Usage Guide

## Synthetic Series Example

The plots below show synthetic series generated with a linear trend, a seasonal component, and consistent noise. The overlays show reference forecasts from DLinear, TimeBase, and TimeBaseTrend models.

![Synthetic series (easy)](img/synthetic_series_easy.png)

![Synthetic series (medium)](img/synthetic_series_medium.png)

![Synthetic series (hard)](img/synthetic_series_hard.png)

## Synthetic Series Generator

The library includes a synthetic series generator for testing and validation:

```python
from tests.utils.synthetic_series import make_synthetic_series

# Easy scenario - basic trend + seasonality + noise
frame_easy = make_synthetic_series(
    length=360,
    noise_std=0.15,
    include_trend=True,
    include_seasonality=True,
    season_period=24,
    amplitude_period=None,
    amplitude_strength=0.0,
)

# Medium scenario - adds amplitude modulation
frame_medium = make_synthetic_series(
    length=360,
    noise_std=0.15,
    include_trend=True,
    include_seasonality=True,
    season_period=24,
    amplitude_period=48,
    amplitude_strength=0.4,
)

# Hard scenario - adds amplitude growth
frame_hard = make_synthetic_series(
    length=360,
    noise_std=0.15,
    include_trend=True,
    include_seasonality=True,
    season_period=24,
    amplitude_period=96,
    amplitude_strength=0.9,
    amplitude_growth_rate=1.2,
)
```

## Target MAE Thresholds

The targets below are derived from running DLinear on the multivariate synthetic cases and adding a 25% margin.

- Easy: **MAE ≤ 0.166** (DLinear 0.1322 × 1.25)
- Medium: **MAE ≤ 0.177** (DLinear 0.1415 × 1.25)
- Hard: **MAE ≤ 0.213** (DLinear 0.1697 × 1.25)

## Complete Examples

### Example 1: Univariate Forecasting

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

# Generate sample data
np.random.seed(42)
n = 200
dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
trend = np.linspace(0, 10, n)
seasonality = 2 * np.sin(2 * np.pi * np.arange(n) / 24)
noise = np.random.randn(n) * 0.1
y = trend + seasonality + noise

df = pd.DataFrame({
    'ds': dates,
    'y': y,
    'unique_id': 'series_1'
})

# Train model
model = TimeBase(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
    max_steps=200,
    learning_rate=1e-2,
)

nf = NeuralForecast(models=[model], freq='D')
nf.fit(df)

# Predict
predictions = nf.predict()
print(predictions.head())
```

### Example 2: Multivariate Forecasting

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from timebaseula import TimeBase, TimeBaseTrend

# Generate multiple series
np.random.seed(42)
n = 200
n_series = 3

frames = []
for i in range(n_series):
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    trend = np.linspace(i, i + 10, n)
    seasonality = 2 * np.sin(2 * np.pi * np.arange(n) / 24 + i)
    noise = np.random.randn(n) * 0.1
    y = trend + seasonality + noise

    frames.append(pd.DataFrame({
        'ds': dates,
        'y': y,
        'unique_id': f'series_{i}'
    }))

df = pd.concat(frames)

# Train models
models = [
    TimeBase(h=24, input_size=48, period_len=24, basis_num=6),
    TimeBaseTrend(h=24, input_size=48, period_len=24, moving_avg_window=25),
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df)

# Predict all series
predictions = nf.predict()
print(predictions['TimeBase'].head(24))
```

### Example 3: Single Series Prediction from Multivariate Model

```python
from timebaseula import predict_single_series

# After training (see Example 2)
# Extract predictions for a specific series
target_series = df[df['unique_id'] == 'series_0'].copy()

pred = predict_single_series(
    model=model,
    series=target_series,
    h=24,
    input_size=48,
    freq='D'
)

print(pred.head())
```

## Model Configuration

### TimeBase Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h` | int | Required | Forecast horizon |
| `input_size` | int | Required | Input window size |
| `period_len` | int | 24 | Segment period length |
| `basis_num` | int | 6 | Number of basis components |
| `use_period_norm` | bool | True | Normalize per period |
| `use_orthogonal` | bool | False | Enable orthogonal regularization |
| `orthogonal_weight` | float | 0.0 | Orthogonal loss weight |

### TimeBaseTrend Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h` | int | Required | Forecast horizon |
| `input_size` | int | Required | Input window size |
| `period_len` | int | 24 | Segment period length |
| `basis_num` | int | 6 | Number of basis components |
| `moving_avg_window` | int | 25 | Moving average window (must be odd) |
| `use_period_norm` | bool | True | Normalize per period in TimeBase |
| `use_orthogonal` | bool | False | Enable orthogonal regularization |
| `orthogonal_weight` | float | 0.0 | Orthogonal loss weight |
