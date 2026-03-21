---
description: Model notes for TimeBase and TimeBaseTrend, including how the repository implementation maps to the paper.
---

# Models

**TL;DR**
- `TimeBase` uses two linear layers: history segments → basis, then basis → forecast segments.
- `TimeBaseTrend` applies moving-average decomposition and adds a linear trend projection.
- The implementation lives in `timebaseula/models/timebase.py`.

## `TimeBase`

### Core idea

The model assumes long histories contain repeated temporal structure. It:

1. pads the input to a multiple of `period_len`
2. reshapes the history into segments
3. optionally normalizes by period mean
4. extracts a low-dimensional basis
5. forecasts future segments from that basis
6. flattens the result back to horizon `h`

### Repository implementation

The implementation uses:

- `TimeBaseConfig`
- `TimeBaseCore`
- `TimeBase(BaseWindows)`

The core layers are:

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

The implementation computes a Gram matrix from the basis tensor and penalizes off-diagonal energy.

## Shapes and behavior

| Stage | Shape |
|---|---|
| input window | `(batch, input_size)` |
| segmented input | `(batch, period_len, seg_num_x)` |
| basis | `(batch, period_len, basis_num)` |
| forecast | `(batch, h)` |

## What is tested

The test suite currently checks:

- output shapes
- padding behavior
- deterministic behavior under a fixed seed
- orthogonal loss sanity
- odd-window validation for `TimeBaseTrend`
- basic NeuralForecast fit/predict integration

See `tests/test_timebase.py`.
