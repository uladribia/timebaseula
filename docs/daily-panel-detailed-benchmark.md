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

## Supplementary Poisson-loss rerun for neural-only models

A follow-up heavy rerun was executed on the same detailed subset using `--neural-loss poisson` and only the NeuralForecast variants (`TimeBaseTrend`, `TimeBase`, `DLinear`, `NLinear`).
This rerun is intended as a probabilistic-loss comparison, not as a replacement for the published headline table above.

### Poisson rerun summary

| metric | TimeBaseTrend | TimeBase | DLinear | NLinear |
| --- | --- | --- | --- | --- |
| training_time_seconds | 22.7206 | 17.5791 | 17.5703 | 18.2361 |
| inference_time_seconds | 1.1394 | 1.1722 | 0.8513 | 1.029 |
| parameters | 2486 | 82 | 3192 | 1596 |
| avg_mae | 4.6205 | 4.7719 | 4.8449 | 4.9541 |
| median_mae | 3.7356 | 3.8686 | 3.8667 | 4.0292 |
| avg_mean_scaled_mae | 0.63 | 0.6427 | 0.645 | 0.6525 |
| median_mean_scaled_mae | 0.4701 | 0.483 | 0.4918 | 0.5055 |
| avg_rmse | 6.3884 | 6.5296 | 6.581 | 6.6686 |
| median_rmse | 4.8184 | 4.963 | 4.9255 | 5.0782 |
| avg_smape | 0.3601 | 0.3619 | 0.3641 | 0.3661 |
| median_smape | 0.342 | 0.3447 | 0.3461 | 0.3479 |
| avg_rank | 1.9056 | 2.3483 | 2.6576 | 3.0885 |
| median_rank | 1 | 2 | 3 | 3 |
| wins | 769 | 351 | 237 | 179 |

### Comparison against the published detailed benchmark

| model | published loss setup avg_mae | Poisson rerun avg_mae | delta | published avg_mean_scaled_mae | Poisson avg_mean_scaled_mae | delta |
| --- | --- | --- | --- | --- | --- | --- |
| TimeBaseTrend | 4.5542 | 4.6205 | +0.0663 | 0.5991 | 0.63 | +0.0309 |
| TimeBase | 4.7074 | 4.7719 | +0.0645 | 0.6174 | 0.6427 | +0.0253 |
| DLinear | 4.5996 | 4.8449 | +0.2453 | 0.5855 | 0.645 | +0.0595 |
| NLinear | 4.6677 | 4.9541 | +0.2864 | 0.6021 | 0.6525 | +0.0504 |

### Analysis of the Poisson rerun
- `TimeBaseTrend` remains the strongest neural model under Poisson loss. It still leads on `avg_mae`, `avg_rmse`, `avg_rank`, and `wins`, so the trend branch remains the most robust detailed-series option even when the loss is changed.
- The Poisson rerun does **not** improve the top detailed models relative to the published benchmark table. All four neural models degrade on both `avg_mae` and `avg_mean_scaled_mae` compared with the existing published results.
- The degradation is modest for `TimeBaseTrend` and `TimeBase`, but noticeably larger for `DLinear` and `NLinear`. That suggests the TimeBase family is more stable than the linear baselines when switching this detailed panel to a count-oriented probabilistic objective.
- Because the rerun does not improve the headline models, the published plots above were kept unchanged. They still reflect the stronger existing benchmark configuration for the top detailed models.
- Operationally, the Poisson setup also increases inference time substantially because NeuralForecast prediction with distribution losses samples from the estimated distribution before returning summary outputs.

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
