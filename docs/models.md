---
description: Reference for the TimeBaseUla model classes and their default behavior.
---

# Models

## TL;DR
- `TimeBase` is the explicit segmented-basis model.
- `TimeBaseTrend` adds a trend decomposition branch.
- `AutoTimeBase` and `AutoTimeBaseTrend` are thin Nixtla-style auto wrappers.
- The public API is centered on those four model classes.

## `TimeBase`

`TimeBase` reshapes the input window into temporal segments, projects them to a compact basis, and maps that basis to future segments.

### Explicit defaults

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| daily `period_len` | `7` |
| monthly `period_len` | `12` |
| other `period_len` | `h` |
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
- they expose a compact search space similar in spirit to `AutoDLinear`
- they do not introduce a custom orchestration layer around training

## Example

```python
from timebaseula import TimeBase, AutoTimeBase

explicit_model = TimeBase(h=12, freq="D")
auto_model = AutoTimeBase(h=12, freq="D", num_samples=5, gpus=0)
```
