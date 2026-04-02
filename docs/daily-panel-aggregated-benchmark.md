---
description: Benchmark report for a daily panel dataset using TimeBaseUla and baseline models.
---

# Daily panel benchmark

## TL;DR
- Best model in this run: `AutoTheta`
- Benchmarked series: `64`
- Rolling evaluation windows: `6`
- Rolling test size: `168` days
- Forecast horizon: `28` daily steps

## Dataset summary
- Total regularized rows: `60032`
- Total unique dates: `938`
- Cross-validation train window: `2023-09-01 to 2025-10-09`
- Cross-validation test window: `2025-10-10 to 2026-03-26`
- Training profile: `normal`
- Training and inference times are measured on the final single `28`-day holdout.
- Accuracy metrics are aggregated across rolling `28`-day cross-validation windows.

## Aggregate metrics

| metric | AutoTheta | AutoNLinear | AutoTimeBaseTrend | TimeBase | AutoTimeBase | TimeBaseTrend | AutoMFLES | AutoDLinear | NLinear | DLinear | Naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training_time_seconds | 17.0716 | 8.4545 | 5.6182 | 2.5185 | 1.8969 | 3.7961 | 27.2799 | 8.5183 | 2.4087 | 2.5196 | 0.0037 |
| inference_time_seconds | 0.003 | 0.0228 | 0.0255 | 0.0301 | 0.0353 | 0.0297 | 0.0022 | 0.0287 | 0.0287 | 0.0301 | 0.0014 |
| parameters | 0 | 812 | 4054 | 82 | 74 | 2486 | 0 | 4760 | 1596 | 3192 | 0 |
| avg_mae | 149.975 | 167.8676 | 155.3247 | 169.0332 | 169.2781 | 177.8933 | 268.0997 | 211.8093 | 193.391 | 204.2335 | 398.1608 |
| median_mae | 71.7418 | 76.943 | 74.5063 | 73.5407 | 76.288 | 78.9609 | 94.4746 | 89.6441 | 90.0137 | 91.6681 | 159.7679 |
| avg_mean_scaled_mae | 0.2099 | 0.2268 | 0.2143 | 0.2227 | 0.2226 | 0.2274 | 0.28 | 0.2705 | 0.2549 | 0.2641 | 0.4637 |
| median_mean_scaled_mae | 0.1799 | 0.1978 | 0.1984 | 0.1994 | 0.2014 | 0.2125 | 0.2477 | 0.2241 | 0.2447 | 0.2443 | 0.3671 |
| avg_rmse | 206.6159 | 233.6936 | 221.3195 | 235.1417 | 236.4916 | 242.7941 | 345.5384 | 272.4921 | 259.0937 | 268.8465 | 508.4644 |
| median_rmse | 98.5346 | 108.1468 | 105.9561 | 106.3047 | 110.3388 | 109.1131 | 126.2253 | 121.1919 | 121.0245 | 121.9255 | 210.9598 |
| avg_smape | 0.1661 | 0.1733 | 0.1674 | 0.1692 | 0.1696 | 0.1715 | 0.2134 | 0.2017 | 0.1884 | 0.1939 | 0.3319 |
| median_smape | 0.163 | 0.1811 | 0.1713 | 0.1658 | 0.1695 | 0.169 | 0.1846 | 0.1981 | 0.1875 | 0.1886 | 0.2284 |
| avg_rank | 3.5885 | 4.401 | 4.4089 | 4.7266 | 4.9036 | 5.5729 | 5.9141 | 6.5208 | 7.4141 | 8.1406 | 10.4089 |
| median_rank | 3 | 4 | 4 | 5 | 5 | 5 | 6 | 7 | 8 | 9 | 11 |
| wins | 105 | 83 | 51 | 24 | 30 | 19 | 43 | 21 | 4 | 0 | 4 |

## Reproducible model settings

```python
MODEL_SETTINGS = {
  "TimeBase": {
    "input_size": 56,
    "max_steps": 120,
    "learning_rate": 0.001,
    "basis_num": 6,
    "period_len": 7
  },
  "TimeBaseTrend": {
    "input_size": 84,
    "max_steps": 152,
    "learning_rate": 0.001,
    "basis_num": 6,
    "period_len": 7,
    "moving_avg_window": 21
  },
  "NLinear": {
    "input_size": 56,
    "max_steps": 120,
    "learning_rate": 0.002
  },
  "DLinear": {
    "input_size": 56,
    "max_steps": 120,
    "learning_rate": 0.002
  },
  "AutoMFLES": {
    "season_length": 7
  },
  "Naive": {},
  "AutoTheta": {
    "season_length": 7
  },
  "AutoTimeBase": {
    "input_size": 84,
    "learning_rate": 0.001,
    "max_steps": 80,
    "step_size": 1,
    "scaler_type": "identity",
    "basis_num": 8,
    "period_len": 14
  },
  "AutoTimeBaseTrend": {
    "input_size": 140,
    "learning_rate": 0.001,
    "max_steps": 180,
    "step_size": 1,
    "scaler_type": "identity",
    "basis_num": 8,
    "period_len": 14,
    "moving_avg_window": 21
  },
  "AutoDLinear": {
    "input_size": 84,
    "learning_rate": 0.0009754278461230969,
    "max_steps": 1300,
    "step_size": 28,
    "scaler_type": "robust",
    "moving_avg_window": 51
  },
  "AutoNLinear": {
    "input_size": 28,
    "learning_rate": 0.018219620470507993,
    "max_steps": 800,
    "step_size": 1,
    "scaler_type": "standard"
  }
}
```

## Comments
- Best overall trade-off in this run: AutoTheta (average rank 3.5885, wins 105).
- Fastest training model on the final 28-day holdout: Naive (0.0037 s).
- Fastest inference model on the final 28-day holdout: Naive (0.0014 s).
- Best average mean-scaled MAE across rolling 28-day windows: AutoTheta (0.2099).

## Plots

![Benchmark plot](img/daily-panel-aggregated-benchmark/summary.png)

![Benchmark plot](img/daily-panel-aggregated-benchmark/distribution.png)

![Benchmark plot](img/daily-panel-aggregated-benchmark/forecast_examples.png)
