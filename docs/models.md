---
description: Reference for the TimeBaseUla model classes and their parameters.
---

# Models

## TL;DR
- `TimeBase` is the explicit segmented-basis model.
- `TimeBaseTrend` adds a moving-average trend decomposition branch.
- `AutoTimeBase` and `AutoTimeBaseTrend` wrap those models for NeuralForecast auto tuning.
- The implementation favors small readable building blocks.

## `TimeBase`

`TimeBase` reshapes the input window into temporal segments, projects them to a compact basis, and maps that basis to future segments.

### Core parameters

| Parameter | Effect |
|---|---|
| `h` | Forecast horizon. The model predicts this many future timestamps. |
| `input_size` | History window length used as model input. Defaults to `max(2 * h, 8)`. |
| `period_len` | Segment length used to divide the history and reconstruct the forecast. |
| `basis_num` | Number of learned basis components. Default: `6`. |
| `freq` | Used only to infer a default `period_len`. Daily data defaults to `7`, monthly data to `12`. |
| `use_period_norm` | Whether each segment is normalized before basis learning. |
| `use_orthogonal` | Whether to add an orthogonality penalty to the learned basis. |
| `orthogonal_weight` | Strength of the orthogonality penalty. |

### Training-related parameters

| Parameter | Effect |
|---|---|
| `loss` | Training loss. Defaults to `MAE()`. |
| `valid_loss` | Validation loss. |
| `max_steps` | Maximum number of optimization steps. |
| `learning_rate` | Optimizer step size. |
| `val_check_steps` | How often validation is run during training. |
| `batch_size` | Number of series or windows per batch. |
| `windows_batch_size` | Number of sampled windows processed per optimization step. |
| `inference_windows_batch_size` | Window batch size used during prediction. |
| `step_size` | Distance between consecutive sampled windows. |
| `scaler_type` | Target scaling strategy used by NeuralForecast. |
| `random_seed` | Controls stochastic training behavior. |
| `num_workers_loader` | Number of dataloader worker processes. |
| `trainer_kwargs` | Extra NeuralForecast or Lightning trainer settings. CPU defaults are enforced unless you override them. |

## `TimeBaseTrend`

`TimeBaseTrend` keeps the TimeBase seasonal branch and adds a trend branch based on `SeriesDecomp` from Nixtla's DLinear implementation.

### Extra parameter

| Parameter | Effect |
|---|---|
| `moving_avg_window` | Centered moving-average window used by the trend decomposition. It must be odd. Default: `25`. |

### Practical intuition for `moving_avg_window`

| Setting | Practical effect |
|---|---|
| small odd window | trend reacts quickly to local changes |
| medium odd window | balanced smoothing for many daily series |
| large odd window | smoother, slower trend that pushes more short-term variation into the TimeBase branch |

## Auto wrappers

Use the auto wrappers when you want NeuralForecast to tune the TimeBase family with Ray or Optuna backends.

```python
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

auto_timebase = AutoTimeBase(h=28, num_samples=3, cpus=1, gpus=0, backend="ray")
auto_timebasetrend = AutoTimeBaseTrend(
    h=28,
    num_samples=3,
    cpus=1,
    gpus=0,
    backend="ray",
)
```

## Example

```python
from timebaseula import TimeBase, TimeBaseTrend

explicit_model = TimeBase(h=12, freq="D", period_len=7)
trend_model = TimeBaseTrend(h=12, freq="D", moving_avg_window=11)
```
