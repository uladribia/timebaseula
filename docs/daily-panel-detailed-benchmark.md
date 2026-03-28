---
description: Detailed-only internal daily-panel benchmark report using TimeBaseUla and baseline models.
---

# Detailed daily-panel benchmark

## TL;DR
- Benchmark type: internal anonymized benchmark
- Scope: detailed series only
- Included series: only the most granular series in the prepared panel
- Excluded series: all aggregate series and the global total
- Published run excludes: `AutoTheta`
- Best overall model in this published run: `TimeBaseTrend`
- Best mean-scaled MAE in this published run: `DLinear`
- Benchmarked series: `256`
- Rolling evaluation windows: `6`
- Rolling test size: `168` days
- Forecast horizon: `28` daily steps
- Training profile: `heavy`

## Dataset summary
- Total regularized rows: `240128`
- Total unique dates: `938`
- Eligible detailed candidate series after filtering: `42616`
- Cross-validation train window: `2023-09-01 to 2025-10-09`
- Cross-validation test window: `2025-10-10 to 2026-03-26`
- Training and inference times are measured on the final single `28`-day holdout.
- Accuracy metrics are aggregated across rolling `28`-day cross-validation windows.
- Published plots use anonymized series aliases and anonymized units.
- This published page reports a representative CPU-first subset of a larger internal panel, not an exhaustive run over every detailed series.

## Benchmark machine
- OS: `Ubuntu 24.04` on Linux kernel `6.17.0-19-generic`
- CPU: `Intel(R) Core(TM) Ultra 5 125U`
- Logical CPUs: `14`
- Available memory: about `15 GiB RAM`
- GPU usage: none, CPU-only benchmark runs

## Aggregate metrics

| metric | TimeBaseTrend | DLinear | TimeBase | NLinear | AutoMFLES | Naive |
| --- | --- | --- | --- | --- | --- | --- |
| training_time_seconds | 11.1487 | 7.4642 | 8.3553 | 7.3708 | 159.4172 | 0.0155 |
| inference_time_seconds | 0.0731 | 0.0967 | 0.0619 | 0.0608 | 0.0053 | 0.0031 |
| parameters | 2486 | 3192 | 82 | 1596 | 0 | 0 |
| cv_refit | no | no | no | no | yes | no |
| avg_mae | 4.5542 | 4.5996 | 4.7074 | 4.6677 | 5.1535 | 6.9157 |
| median_mae | 3.681 | 3.7087 | 3.8106 | 3.7838 | 4.0343 | 5.6429 |
| avg_mean_scaled_mae | 0.5991 | 0.5855 | 0.6174 | 0.6021 | 0.604 | 0.7719 |
| median_mean_scaled_mae | 0.4656 | 0.4734 | 0.478 | 0.4784 | 0.5367 | 0.716 |
| avg_rmse | 6.3408 | 6.3834 | 6.517 | 6.4238 | 6.8168 | 8.9014 |
| median_rmse | 4.8191 | 4.8375 | 4.939 | 4.8721 | 5.1775 | 7.0812 |
| avg_smape | 0.3673 | 0.3679 | 0.3681 | 0.3676 | 0.3772 | 0.4785 |
| median_smape | 0.348 | 0.3519 | 0.3495 | 0.3476 | 0.3578 | 0.4178 |
| avg_rank | 2.5944 | 2.9434 | 3.0286 | 3.1556 | 3.8444 | 5.4336 |
| median_rank | 2 | 3 | 3 | 3 | 5 | 6 |
| wins | 450 | 258 | 264 | 167 | 323 | 72 |

## Interpretation
- `TimeBaseTrend` is the strongest overall model in this detailed-only run. It leads on `avg_rank`, `avg_mae`, `median_mae`, `avg_rmse`, and total `wins`, which points to the best overall balance across many granular forecasting tasks.
- `DLinear` has the best `avg_mean_scaled_mae`, so it is the strongest option when normalized error is the main objective and relative scale matters more than raw task wins.
- `TimeBase` remains competitive, but the extra trend branch in `TimeBaseTrend` appears to help more on detailed series than it did on the aggregated benchmark.
- `NLinear` stays close to the top group but does not beat `TimeBaseTrend` or `DLinear` on the headline metrics in this run.
- `AutoMFLES` is still competitive for a statistical baseline, but it is far slower to train and it is the only model in this table that falls back to `refit=True` during cross-validation.

## Recommendation
- Choose `TimeBaseTrend` as the default model for this detailed-only benchmark when you want the strongest overall trade-off across granular series.
- Choose `DLinear` when mean-scaled error is the primary decision metric.
- Keep `TimeBase` and `NLinear` as strong secondary neural baselines.
- Keep `AutoMFLES` as a reference baseline, but interpret it with care because of the cross-validation refit caveat.

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

## Plots

![Benchmark plot](img/daily-panel-detailed-benchmark/summary.png)

![Benchmark plot](img/daily-panel-detailed-benchmark/distribution.png)

![Benchmark plot](img/daily-panel-detailed-benchmark/forecast_examples.png)
