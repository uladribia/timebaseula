---
description: Reference for the TimeBaseUla model classes and their parameters.
---

# Models

## TL;DR
- `TimeBase` is the explicit segmented-basis model.
- `TimeBaseTrend` adds a moving-average trend decomposition branch.
- `AutoTimeBase` and `AutoTimeBaseTrend` expose the same family through NeuralForecast auto tuning.
- The implementation separates the pure Torch core, pure Torch decomposition, shared wrapper base, defaults, and factories into smaller internal modules.
- The local decomposition is intentional to avoid coupling `TimeBaseTrend` to DLinear internals, but it can still be swapped back to the upstream helper later if needed.

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
| `use_period_norm` | If `True`, the model normalizes each segment by its own mean before learning the basis. |
| `use_orthogonal` | If `True`, adds a penalty that encourages basis components to be less redundant. |
| `orthogonal_weight` | Strength of that orthogonality penalty. |

### Training-related parameters

These are passed through the underlying `NeuralForecast` / Lightning training wrapper.

| Parameter | Effect |
|---|---|
| `loss` | Training loss. Defaults to `MAE()`. The explicit models also support multi-output losses such as `MQLoss()` and `DistributionLoss(...)`. |
| `valid_loss` | Validation loss. If omitted, NeuralForecast uses its default behavior. |
| `max_steps` | Maximum number of optimization steps. |
| `learning_rate` | Optimizer step size. |
| `val_check_steps` | How often validation is run during training. |
| `batch_size` | Number of series/windows per batch. |
| `windows_batch_size` | Number of sampled windows processed per optimization step. |
| `inference_windows_batch_size` | Window batch size used during prediction. |
| `step_size` | Distance between consecutive sampled windows. |
| `scaler_type` | Target scaling strategy used by NeuralForecast. |
| `random_seed` | Controls stochastic training behavior. |
| `num_workers_loader` | Number of dataloader worker processes. |
| `trainer_kwargs` | Extra NeuralForecast or Lightning trainer settings. CPU defaults are enforced unless you override them. |

## `TimeBaseTrend`

`TimeBaseTrend` keeps the TimeBase seasonal branch and adds a trend branch based on a local pure-Torch moving-average decomposition.

That local decomposition is intentional: it avoids coupling the explicit TimeBase family to `neuralforecast.models.dlinear`. The implementation keeps the same seasonal-plus-trend contract as the upstream DLinear helper, so maintainers can still revert to the dependency later if that becomes preferable.

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

## Internal module layout

| Module | Responsibility |
|---|---|
| `timebaseula/models/core.py` | pure Torch segmented-basis core |
| `timebaseula/models/decomposition.py` | pure Torch moving-average decomposition for `TimeBaseTrend` |
| `timebaseula/models/base.py` | shared NeuralForecast wrapper and training-step logic |
| `timebaseula/models/defaults.py` | explicit-model defaults and small validation helpers |
| `timebaseula/models/factories.py` | shared config resolution and core factory helpers |
| `timebaseula/models/timebase.py` | public `TimeBase` and `TimeBaseTrend` wrappers |

## Example

```python
from neuralforecast.losses.pytorch import DistributionLoss
from timebaseula import TimeBase, TimeBaseTrend

explicit_model = TimeBase(h=12, freq="D", period_len=7)
trend_model = TimeBaseTrend(h=12, freq="D", moving_avg_window=11)
probabilistic_model = TimeBase(
    h=12,
    freq="D",
    loss=DistributionLoss("Normal", level=[80, 95]),
)
```
