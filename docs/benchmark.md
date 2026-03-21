---
description: Benchmark report generated from the long-horizon benchmark CSV.
---

# Benchmark report

Source CSV: `logs/bench_skip.csv`

## TL;DR
- The table below captures the full benchmark result set.
- The summary table lists the best MAE per dataset/frequency slice.
- AutoARIMA can be skipped for faster exploratory runs.

## Observations

- Best overall MAE in this run: `SeasonalNaive` on `ECL D` with `MAE=0.1910`.
- Smallest parameterized model in this run: `SeasonalNaive` with `0` trainable parameters.
- Fastest model with non-zero training time: `TimeBase` with `train_time=0.07s`.

## Full results

| model_name | dataset | frequency | mae | rmse | params | train_time | inference_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SeasonalNaive | ECL | D | 0.1909672054134845 | 0.4105110531596958 | 0 | 0.0 | 0.0 |
| DLinear | ECL | D | 0.3706031404942855 | 0.5065673499755595 | 238 | 0.1036098940003285 | 0.0157916890002525 |
| NLinear | ECL | D | 0.3082192155490222 | 0.5113828251326983 | 119 | 0.074218223000571 | 0.0175445400000171 |
| TimeBase | ECL | D | 0.2251355853554683 | 0.4449820215130232 | 31 | 0.0679969299999356 | 0.016639712000142 |
| TimeBaseTrend | ECL | D | 0.3338519168449704 | 0.5358531030082617 | 150 | 0.0975071200000456 | 0.0178307030000723 |
| AutoMFLES | ECL | D | 0.3666896128553141 | 0.5530935514174181 | 0 | 5.83251188199938 | 0.0078086709982017 |

## Best MAE by slice

| dataset | frequency | best_model | best_mae |
| --- | --- | --- | --- |
| ECL | D | SeasonalNaive | 0.1909672054134845 |
