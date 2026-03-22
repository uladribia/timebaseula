---
description: Model notes for TimeBase and TimeBaseTrend, including implementation and tuning details.
---

# Models

**TL;DR**
- `TimeBase` uses two linear layers: history segments → basis, then basis → forecast segments.
- `TimeBaseTrend` applies moving-average decomposition and adds a linear trend projection.
- Shared training logic lives in a common internal wrapper so both models behave consistently.

## `TimeBase`

The model assumes long histories contain repeated temporal structure. It:

1. pads the input to a multiple of `period_len`
2. reshapes the history into segments
3. optionally normalizes by period mean
4. extracts a low-dimensional basis
5. forecasts future segments from that basis
6. flattens the result back to horizon `h`

Core layers:

| Layer | Role |
|---|---|
| `ts2basis` | maps segment history to learned basis components |
| `basis2ts` | maps basis components to future segments |

## `TimeBaseTrend`

`TimeBaseTrend` adds a trend branch inspired by DLinear-style decomposition:

1. decompose the input with `SeriesDecomp`
2. pass the seasonal component through `TimeBaseCore`
3. pass the trend component through `linear_trend`
4. add both forecasts

## Orthogonal regularization

Both models can compute an orthogonal penalty on the learned basis matrix.

| Parameter | Meaning |
|---|---|
| `use_orthogonal` | enable orthogonal regularization |
| `orthogonal_weight` | weight of the penalty term |

## Recommendation helpers

Top-level helpers:

- `timebaseula.profile_dataset(...)`
- `timebaseula.recommend_timebase_kwargs(...)`
- `timebaseula.recommend_timebase_trend_kwargs(...)`

Class helpers:

- `TimeBase.profile_dataset(...)`
- `TimeBase.recommend_defaults(...)`
- `TimeBaseTrend.profile_dataset(...)`
- `TimeBaseTrend.recommend_defaults(...)`

These helpers inspect a long-format dataframe with `unique_id`, `ds`, and `y`, then derive compact defaults for:

- `input_size`
- `period_len`
- `basis_num`
- `moving_avg_window`
- `max_steps`
- `learning_rate`
- `early_stop_patience_steps`
- `val_check_steps`

## Testing policy

- unit tests cover shape, padding, recommendation logic, and synthetic helpers
- integration tests cover actual NeuralForecast fitting behavior
- benchmark-oriented runs stay outside the fast default suite
