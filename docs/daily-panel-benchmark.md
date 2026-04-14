---
description: Published mixed-scope daily-panel benchmark report for TimeBaseUla and baseline models.
---

# Daily panel benchmark

## TL;DR
- Best model in this published rerun: `TimeBaseTrend`
- Benchmarked series: `256`
- Rolling evaluation windows: `6`
- Rolling test size: `168` days
- Forecast horizon: `28` daily steps
- Training profile: `heavy`
- Published run excludes: `AutoTheta`

## Dataset summary
- Total regularized rows: `240128`
- Total unique dates: `938`
- Cross-validation train window: `2023-09-01 to 2025-10-09`
- Cross-validation test window: `2025-10-10 to 2026-03-26`
- Training and inference times are measured on the final single `28`-day holdout.
- Accuracy metrics are aggregated across rolling `28`-day cross-validation windows.
- Published plots use anonymized series aliases and anonymized units.

## Benchmark machine
- OS: `Ubuntu 24.04` on Linux kernel `6.17.0-19-generic`
- CPU: `Intel(R) Core(TM) Ultra 5 125U`
- Logical CPUs: `14`
- Available memory: about `15 GiB RAM`
- GPU usage: none, CPU-only benchmark runs

## Aggregate metrics

| metric | TimeBaseTrend | TimeBase | AutoMFLES | NLinear | DLinear | Naive |
| --- | --- | --- | --- | --- | --- | --- |
| training_time_seconds | 22.0888 | 12.9693 | 96.0719 | 4.0219 | 3.0257 | 0.0144 |
| inference_time_seconds | 0.0339 | 0.0372 | 0.0039 | 0.0388 | 0.0738 | 0.0031 |
| parameters | 2486 | 82 | 0 | 1596 | 3192 | 0 |
| avg_mae | 66.1189 | 67.7587 | 99.9432 | 74.9211 | 84.7788 | 152.6325 |
| median_mae | 34.904 | 35.3096 | 42.1879 | 39.3079 | 44.9805 | 70.125 |
| avg_mean_scaled_mae | 0.2356 | 0.238 | 0.3015 | 0.2608 | 0.2885 | 0.4726 |
| median_mean_scaled_mae | 0.2077 | 0.2138 | 0.2559 | 0.2307 | 0.2564 | 0.373 |
| avg_rmse | 92.0599 | 94.6376 | 128.3349 | 99.9352 | 109.4186 | 194.0556 |
| median_rmse | 48.4083 | 49.3459 | 56.1837 | 52.3747 | 57.1701 | 93.8925 |
| avg_smape | 0.1842 | 0.1871 | 0.2233 | 0.2007 | 0.2154 | 0.3329 |
| median_smape | 0.1834 | 0.1852 | 0.1968 | 0.1971 | 0.2095 | 0.2272 |
| avg_rank | 2.194 | 2.3203 | 3.2396 | 3.3919 | 4.3353 | 5.5189 |
| median_rank | 2 | 2 | 3 | 4 | 4 | 6 |
| wins | 504 | 398 | 436 | 133 | 22 | 43 |

## Reproducible model settings

```python
MODEL_SETTINGS = {
  "TimeBase": {
    "input_size": 56,
    "max_steps": 256,
    "learning_rate": 0.001,
    "basis_num": 6,
    "period_len": 7
  },
  "TimeBaseTrend": {
    "input_size": 84,
    "max_steps": 304,
    "learning_rate": 0.001,
    "basis_num": 6,
    "period_len": 7,
    "moving_avg_window": 21
  },
  "NLinear": {
    "input_size": 56,
    "max_steps": 240,
    "learning_rate": 0.002
  },
  "DLinear": {
    "input_size": 56,
    "max_steps": 240,
    "learning_rate": 0.002
  },
  "AutoMFLES": {
    "season_length": 7
  },
  "Naive": {}
}
```

## Comments
- Best overall trade-off in this run: `TimeBaseTrend` (`avg_rank=2.1940`, `wins=504`).
- Best `avg_mae` and best `avg_mean_scaled_mae`: `TimeBaseTrend`.
- `TimeBase` remains a close second and stays much smaller at `82` parameters.
- The non-TimeBase baselines reproduce their previously published accuracy values, so the main behavioral change is concentrated in the TimeBase family.

## Plots

![Benchmark plot](img/daily-panel-benchmark/summary.png)

![Benchmark plot](img/daily-panel-benchmark/distribution.png)

![Benchmark plot](img/daily-panel-benchmark/forecast_examples.png)
