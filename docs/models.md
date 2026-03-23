---
description: Reference for the TimeBaseUla model classes and their default behavior.
---

# Models

## TL;DR
- `TimeBase` is the explicit segmented-basis model.
- `TimeBaseTrend` adds a trend decomposition branch.
- `AutoTimeBase` and `AutoTimeBaseTrend` are thin Nixtla-style auto wrappers.
- The library no longer exposes dataset profilers or recommendation helpers.

## `TimeBase`

`TimeBase` reshapes the input window into temporal segments, projects them to a compact basis, and maps that basis to future segments.

### Explicit defaults

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| `period_len` with daily freq | `7` |
| `period_len` with monthly freq | `12` |
| `period_len` otherwise | `h` |
| `basis_num` | `6` |

## `TimeBaseTrend`

`TimeBaseTrend` keeps the TimeBase seasonal branch and adds a trend branch based on `SeriesDecomp` from Nixtla's DLinear implementation.

### Extra default

| Parameter | Default |
|---|---|
| `moving_avg_window` | `25` |

## Auto wrappers

| Wrapper | Base model | Notes |
|---|---|---|
| `AutoTimeBase` | `TimeBase` | Nixtla-style search over structural and training hyperparameters |
| `AutoTimeBaseTrend` | `TimeBaseTrend` | Same idea plus `moving_avg_window` |

These wrappers are intentionally thin:
- they subclass NeuralForecast's `BaseAuto`
- they expose a compact search space, similar in spirit to `AutoDLinear`
- they do not contain a handwritten profiling or local-search framework

## Example

```python
from timebaseula import TimeBase, AutoTimeBase

explicit_model = TimeBase(h=12, freq="D")
auto_model = AutoTimeBase(h=12, freq="D", num_samples=5, gpus=0)
```

## What is not part of the public API anymore

- synthetic-series helpers
- dataset profiling dataclasses
- recommendation helper functions

Those ideas added maintenance cost without enough library value, so the package API is now centered on the four model classes only.
