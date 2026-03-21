---
description: Benchmark report generated from the long-horizon benchmark CSV.
---

# Benchmark report

Source CSV: `logs/benchmark_results_300_longer_horizons_no_arima.csv`

## TL;DR
- This report combines two **300-series, no-ARIMA** benchmark runs with longer forecast horizons.
- Daily used `horizon=28` and `max_steps=200`; monthly used `horizon=8` and `max_steps=100` to reduce neural underfitting.
- Daily winners were strong neural models: `DLinear` on `ECL D` and `TimeBaseTrend` on `TrafficL D`.
- Monthly winners were `AutoMFLES` on both `ECL ME` and `TrafficL ME`, with `SeasonalNaive` and `NLinear` remaining competitive.

## Benchmark setup

Commands used:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode daily \
  --n-series 300 \
  --horizon 28 \
  --max-steps 200 \
  --skip-arima \
  --output logs/benchmark_results_300_daily_h28_no_arima.csv

uv run --frozen python scripts/benchmark_long_horizon.py main \
  --mode monthly \
  --n-series 300 \
  --horizon 8 \
  --max-steps 100 \
  --skip-arima \
  --output logs/benchmark_results_300_monthly_h8_no_arima.csv
```

Interpretation notes:

- The benchmark uses an approximate **20% tail holdout** per series.
- Daily slices have much longer histories, so larger horizons and training budgets are reasonable.
- Monthly slices are still short (`37` points for `ECL`, `25` for `TrafficL`), so they remain a difficult low-context regime for the neural models.

## Observations

- Best overall MAE in this run: `TimeBaseTrend` on `TrafficL D` with `MAE=0.1346`.
- Smallest parameterized neural model in this run: `TimeBase` on `TrafficL ME` with `19` trainable parameters.
- Fastest model with non-zero training time: `NLinear` on `ECL ME` with `train_time=0.42s`.
- On **daily** data, raising the horizon and training budget keeps the neural models competitive and pushes them clearly ahead of `SeasonalNaive`.
- On **monthly** data, the extra training budget does not rescue `TimeBase` or `TimeBaseTrend`; short history remains the dominant limitation.
- `AutoMFLES` improved its relative standing on the longer monthly horizon, but it is still far slower than the neural models.

## Full results

| model_name | dataset | frequency | mae | rmse | params | train_time | inference_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SeasonalNaive | ECL | D | 0.3037887509867498 | 0.4179188967845395 | 0 | 0.0 | 0.0 |
| DLinear | ECL | D | 0.2489622905407869 | 0.3542025679799673 | 3192 | 2.0457964259994696 | 0.1354991389998758 |
| NLinear | ECL | D | 0.2751773667829011 | 0.3810991713121673 | 1596 | 2.52949988599994 | 0.0855807369998729 |
| TimeBase | ECL | D | 0.2574477599297397 | 0.3637016314835751 | 82 | 3.133273438000288 | 0.055222966000656 |
| TimeBaseTrend | ECL | D | 0.250552887985925 | 0.3564660907059449 | 1678 | 7.070884437999666 | 0.1548908940003457 |
| AutoMFLES | ECL | D | 0.2964677782103513 | 0.4070001529474727 | 0 | 208.66864576500663 | 0.706184479001422 |
| SeasonalNaive | TrafficL | D | 0.1537023920520695 | 0.2642096827723658 | 0 | 0.0 | 0.0 |
| DLinear | TrafficL | D | 0.1358528177310827 | 0.2268924593640161 | 3192 | 1.6365842960003647 | 0.057447766000223 |
| NLinear | TrafficL | D | 0.1359612273582318 | 0.2307995184998283 | 1596 | 2.422431391999453 | 0.0948225669999374 |
| TimeBase | TrafficL | D | 0.1363636670891952 | 0.2322926408972248 | 82 | 2.9622092409999823 | 0.0687736820000282 |
| TimeBaseTrend | TrafficL | D | 0.1345640021674743 | 0.2266451024536874 | 1678 | 3.105592056000205 | 0.0670522749996962 |
| AutoMFLES | TrafficL | D | 0.1488523688912477 | 0.2478262002180868 | 0 | 50.48520766399906 | 0.796633012001621 |
| SeasonalNaive | ECL | ME | 0.2466916257576251 | 0.399386804374634 | 0 | 0.0 | 0.0 |
| DLinear | ECL | ME | 0.3667793102786634 | 0.5424554794866039 | 272 | 0.5553441969996129 | 0.0376603690001502 |
| NLinear | ECL | ME | 0.2440902429427524 | 0.39846400334831 | 136 | 0.416549391999979 | 0.0184905319993049 |
| TimeBase | ECL | ME | 0.4436032467607653 | 0.5700919699976115 | 25 | 0.5237924469993231 | 0.0189142979997996 |
| TimeBaseTrend | ECL | ME | 0.5082316029565532 | 0.6939568875493352 | 161 | 0.6079236819996368 | 0.0565584550004132 |
| AutoMFLES | ECL | ME | 0.2384362190045449 | 0.3866161467592039 | 0 | 13.233893621006246 | 0.499461196998709 |
| SeasonalNaive | TrafficL | ME | 0.1705565490963342 | 0.2631933849574819 | 0 | 0.0 | 0.0 |
| DLinear | TrafficL | ME | 0.1716414462555514 | 0.2502644195624174 | 160 | 4.279246238000269 | 0.1010740649999206 |
| NLinear | TrafficL | ME | 0.1718309449422018 | 0.2600209118569055 | 80 | 0.4630851450001501 | 0.0257033620000584 |
| TimeBase | TrafficL | ME | 0.1895988269093635 | 0.2875683310579101 | 19 | 0.5060100769997007 | 0.0299779350007156 |
| TimeBaseTrend | TrafficL | ME | 0.1848082213899297 | 0.2732550945673392 | 99 | 0.8037537369991696 | 0.3138919740003985 |
| AutoMFLES | TrafficL | ME | 0.1628093535042005 | 0.2607831078568183 | 0 | 5.66957313799503 | 0.4279862930034142 |

## Best MAE by slice

| dataset | frequency | best_model | best_mae |
| --- | --- | --- | --- |
| ECL | D | DLinear | 0.2489622905407869 |
| ECL | ME | AutoMFLES | 0.2384362190045449 |
| TrafficL | D | TimeBaseTrend | 0.1345640021674743 |
| TrafficL | ME | AutoMFLES | 0.1628093535042005 |
