---
description: Published aggregated-only daily-panel benchmark report for TimeBaseUla and baseline models.
---

# Aggregated daily-panel benchmark

## TL;DR
- Best model in this published rerun: `AutoTheta`
- Benchmarked series: `64`
- Rolling evaluation windows: `6`
- Rolling test size: `168` days
- Forecast horizon: `28` daily steps
- Training profile: `normal`
- Tuned auto-model configs are loaded from `artifacts/tuning/aggregated/best_configs.json`

## Dataset summary
- Total regularized rows: `60032`
- Total unique dates: `938`
- Cross-validation train window: `2023-09-01 to 2025-10-09`
- Cross-validation test window: `2025-10-10 to 2026-03-26`
- Training and inference times are measured on the final single `28`-day holdout.
- Accuracy metrics are aggregated across rolling `28`-day cross-validation windows.

## Benchmark machine
- OS: `Ubuntu 24.04` on Linux kernel `6.17.0-19-generic`
- CPU: `Intel(R) Core(TM) Ultra 5 125U`
- Logical CPUs: `14`
- Available memory: about `15 GiB RAM`
- GPU usage: none, CPU-only benchmark runs

## Aggregate metrics

| metric | AutoTheta | AutoTimeBaseTrend | AutoNLinear | TimeBase | TimeBaseTrend | AutoTimeBase | AutoMFLES | AutoDLinear | NLinear | DLinear | Naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training_time_seconds | 17.1573 | 7.0366 | 10.9653 | 1.4354 | 2.7024 | 2.2003 | 32.69 | 11.2634 | 2.2705 | 2.7626 | 0.004 |
| inference_time_seconds | 0.0027 | 0.0582 | 0.0387 | 0.0157 | 0.0208 | 0.0556 | 0.0027 | 0.0335 | 0.0217 | 0.0545 | 0.0016 |
| parameters | 0 | 4054 | 812 | 82 | 2486 | 74 | 0 | 4760 | 1596 | 3192 | 0 |
| avg_mae | 149.975 | 153.6379 | 167.8676 | 166.6526 | 167.0068 | 168.5292 | 268.0997 | 211.8093 | 193.391 | 204.2335 | 398.1608 |
| median_mae | 71.7418 | 71.4471 | 76.943 | 71.8422 | 74.5094 | 75.729 | 94.4746 | 89.6441 | 90.0137 | 91.6681 | 159.7679 |
| avg_mean_scaled_mae | 0.2099 | 0.2084 | 0.2268 | 0.2203 | 0.2176 | 0.2208 | 0.28 | 0.2705 | 0.2549 | 0.2641 | 0.4637 |
| median_mean_scaled_mae | 0.1799 | 0.1929 | 0.1978 | 0.198 | 0.1973 | 0.2012 | 0.2477 | 0.2241 | 0.2447 | 0.2443 | 0.3671 |
| avg_rmse | 206.6159 | 218.9956 | 233.6936 | 232.4799 | 235.0459 | 234.4249 | 345.5384 | 272.4921 | 259.0937 | 268.8465 | 508.4644 |
| median_rmse | 98.5346 | 102.9321 | 108.1468 | 105.1393 | 105.2379 | 108.2826 | 126.2253 | 121.1919 | 121.0245 | 121.9255 | 210.9598 |
| avg_smape | 0.1661 | 0.1634 | 0.1733 | 0.1678 | 0.1665 | 0.1686 | 0.2134 | 0.2017 | 0.1884 | 0.1939 | 0.3319 |
| median_smape | 0.163 | 0.1596 | 0.1811 | 0.166 | 0.1699 | 0.1676 | 0.1846 | 0.1981 | 0.1875 | 0.1886 | 0.2284 |
| avg_rank | 3.724 | 3.8438 | 4.5964 | 4.8099 | 4.8438 | 5.1771 | 6.0729 | 6.7031 | 7.5651 | 8.2292 | 10.4349 |
| median_rank | 3 | 3 | 5 | 5 | 4 | 5 | 6 | 7 | 8 | 9 | 11 |
| wins | 104 | 47 | 85 | 19 | 37 | 24 | 41 | 21 | 3 | 0 | 3 |

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
  "AutoMFLES": {"season_length": 7},
  "Naive": {},
  "AutoTheta": {"season_length": 7},
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
- Best overall trade-off in this run: `AutoTheta` (`avg_rank=3.7240`, `wins=104`).
- Best `avg_mean_scaled_mae`: `AutoTimeBaseTrend` (`0.2084`).
- `TimeBase`, `TimeBaseTrend`, and `AutoTimeBaseTrend` all improve their previously published `avg_mae` values.
- The non-TimeBase baselines reproduce their earlier `avg_mae` values exactly, so the main change again sits inside the TimeBase family.

## Plots

![Benchmark plot](img/daily-panel-aggregated-benchmark/summary.png)

![Benchmark plot](img/daily-panel-aggregated-benchmark/distribution.png)

![Benchmark plot](img/daily-panel-aggregated-benchmark/forecast_examples.png)
