---
description: Reference for the TimeBaseUla model classes and their parameters.
---

# Models

## TL;DR
- `TimeBase` is the explicit segmented-basis model.
- `TimeBaseTrend` adds a moving-average trend decomposition branch.
- The public API is centered on those two model classes.
- The implementation favors small readable building blocks.

## `TimeBase`

`TimeBase` reshapes the input window into temporal segments, projects them to a compact basis, and maps that basis to future segments.

### Core parameters

| Parameter | Effect |
|---|---|
| `h` | Forecast horizon. The model predicts this many future timestamps. |
| `input_size` | History window length used as model input. Larger values give the model more context but increase computation. Defaults to `max(2 * h, 8)`. |
| `period_len` | Segment length used to divide the history and reconstruct the forecast. Smaller values focus on short repeated patterns, while larger values encourage broader seasonal structure. |
| `basis_num` | Number of learned basis components. Higher values increase flexibility but also model capacity. Default: `6`. |
| `freq` | Used only to infer a default `period_len` when you do not set it manually. Daily data defaults to `7`, monthly data to `12`. |
| `use_period_norm` | If `True`, the model normalizes each segment by its own mean before learning the basis. This usually helps when level changes across periods are more important than absolute scale. |
| `use_orthogonal` | If `True`, adds a penalty that encourages basis components to be less redundant. |
| `orthogonal_weight` | Strength of that orthogonality penalty. Larger values push the basis toward more distinct components. |

### Training-related parameters

These are passed through the underlying `NeuralForecast` / Lightning training wrapper.

| Parameter | Effect |
|---|---|
| `loss` | Training loss. Defaults to `MAE()`. |
| `valid_loss` | Validation loss. If omitted, NeuralForecast uses its default behavior. |
| `max_steps` | Maximum number of optimization steps. Increase it if the model underfits. |
| `learning_rate` | Optimizer step size. Lower it if training is unstable; raise it cautiously if learning is too slow. |
| `val_check_steps` | How often validation is run during training. |
| `batch_size` | Number of series/windows per batch. Higher values can be faster but use more memory. |
| `windows_batch_size` | Number of sampled windows processed per optimization step. Larger values increase throughput but also memory use. |
| `inference_windows_batch_size` | Window batch size used during prediction. |
| `step_size` | Distance between consecutive sampled windows. `1` gives dense windows; larger values reduce overlap. |
| `scaler_type` | Target scaling strategy used by NeuralForecast. |
| `random_seed` | Controls stochastic training behavior. |
| `num_workers_loader` | Number of dataloader worker processes. |
| `trainer_kwargs` | Extra NeuralForecast/Lightning trainer settings. CPU defaults are enforced unless you override them. |

### Explicit defaults

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| daily `period_len` | `7` |
| monthly `period_len` | `12` |
| other `period_len` | `min(max(2, h), input_size)` |
| `basis_num` | `6` |

## `TimeBaseTrend`

`TimeBaseTrend` keeps the TimeBase seasonal branch and adds a trend branch based on `SeriesDecomp` from Nixtla's DLinear implementation.

### Extra parameter

| Parameter | Effect |
|---|---|
| `moving_avg_window` | Window used by the moving-average decomposition. Larger odd values produce a smoother trend and leave more variation in the seasonal branch. Smaller odd values produce a more reactive trend and leave less smoothing. It must be odd because the decomposition needs a centered window. Default: `25`. |

### How to think about `moving_avg_window`

| Setting | Practical effect |
|---|---|
| small odd window, such as `3` or `5` | trend reacts quickly to local changes; less aggressive smoothing |
| medium odd window, such as `11` or `25` | balanced smoothing for many daily or moderately noisy series |
| large odd window | trend becomes smoother and slower-moving; short-term fluctuations are pushed into the TimeBase seasonal branch |

If the trend forecast looks too wiggly, increase `moving_avg_window`.
If the trend is too flat and misses local changes, decrease it.

All other parameters behave the same as in `TimeBase`.

## Example

```python
from timebaseula import TimeBase, TimeBaseTrend

explicit_model = TimeBase(h=12, freq="D", period_len=7)
trend_model = TimeBaseTrend(h=12, freq="D", moving_avg_window=11)
```
