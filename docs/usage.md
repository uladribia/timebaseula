---
description: Usage guide for TimeBaseUla with basic training, multi-series forecasting, defaults, and practical convergence advice.
---

# Usage

**TL;DR**
- Instantiate `TimeBase` or `TimeBaseTrend` and pass it to `NeuralForecast(models=[...], freq=...)`.
- Always use a non-zero `val_size` during `fit(...)` so you can monitor validation loss.
- Prefer `recommend_timebase_kwargs(...)` and `recommend_timebase_trend_kwargs(...)` over hand-picked defaults.
- If training is unstable, first adjust **training budget**, **learning rate**, and **checkpointing**, not only model size.
- Track `ptl/val_loss` with a logger or callback so you can detect underfitting, overfitting, and non-convergence early.

## Basic univariate example

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 240,
        "ds": pd.date_range("2024-01-01", periods=240, freq="D"),
        "y": range(240),
    }
)

model = TimeBase(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
    max_steps=100,
    learning_rate=1e-3,
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Multi-series example

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBaseTrend

frames = []
for idx in range(3):
    frames.append(
        pd.DataFrame(
            {
                "unique_id": [f"series_{idx}"] * 240,
                "ds": pd.date_range("2024-01-01", periods=240, freq="D"),
                "y": [idx + step for step in range(240)],
            }
        )
    )

frame = pd.concat(frames, ignore_index=True)
model = TimeBaseTrend(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
    moving_avg_window=25,
    max_steps=100,
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

## Predict one series after multi-series training

NeuralForecast already supports this directly.

```python
single_series = frame[frame["unique_id"] == "series_0"].copy()
result = nf.predict(df=single_series)
```

This repository treats that as the supported flow and tests it in the integration suite.

## Automatic default selection

For most users, the easiest entry point is now the auto wrappers.

```python
from timebaseula import AutoTimeBase, AutoTimeBaseTrend

model = AutoTimeBase(h=24, freq="D", max_steps=150)
trend_model = AutoTimeBaseTrend(h=24, freq="D", max_steps=150)
```

These classes:

- start from the same recommendation helpers used elsewhere in the library
- trim validation and test tails from the internal recommendation frame to avoid leakage
- can run a short validation-guided local search before the final fit
- expose the selected configuration on `selected_config_`
- can expose `recommended_training_iterations_` when iteration guidance is enabled

If you want the recommendations without the auto wrapper, you can still call them directly.

```python
from timebaseula import (
    TimeBase,
    TimeBaseTrend,
    recommend_timebase_kwargs,
    recommend_timebase_trend_kwargs,
)

recommended_timebase = recommend_timebase_kwargs(
    frame=frame,
    freq="D",
    horizon=24,
    max_steps=150,
    include_iteration_recommendation=True,
)
recommended_timebase_trend = recommend_timebase_trend_kwargs(
    frame=frame,
    freq="D",
    horizon=24,
    max_steps=150,
    include_iteration_recommendation=True,
)

model = TimeBase(h=24, **recommended_timebase)
trend_model = TimeBaseTrend(h=24, **recommended_timebase_trend)
```

Why this helps:

- the helpers estimate a sensible `input_size`
- they choose a plausible `period_len` for the frequency
- they select a compact but usable `basis_num`
- they also return training defaults such as `max_steps`, `learning_rate`, and validation-check cadence

## How to think about training quality

When a model performs poorly, the cause is often one of three things:

| Problem | Typical symptom | First fixes to try |
|---|---|---|
| Underfitting | train and validation losses both remain high; validation still improving at the end | more `max_steps`, slightly larger `input_size`, possibly more `basis_num` |
| Overfitting | train loss keeps falling but validation loss worsens after a best point | stronger early stopping, best-checkpoint evaluation, smaller learning rate, simpler model |
| Non-convergence / unstable optimization | losses oscillate, jump, or validation is best very early and then drifts | lower learning rate, more frequent validation, checkpoint best model, inspect logs |

The most important practical point is:

> do not diagnose architecture problems before checking convergence behavior.

## Minimum good practice for fitting

At a minimum, always do these three things:

1. set a non-zero `val_size`
2. log validation loss
3. keep the best validation checkpoint when training budgets are non-trivial

### Example: tracked training with logger and checkpoints

```python
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from timebaseula import TimeBase, recommend_timebase_kwargs

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 240,
        "ds": pd.date_range("2024-01-01", periods=240, freq="D"),
        "y": range(240),
    }
)

checkpoints_dir = Path("logs/checkpoints")
logger = CSVLogger(save_dir="logs/training_logs", name="timebase_example")
checkpoint = ModelCheckpoint(
    dirpath=checkpoints_dir / "timebase_example",
    filename="best-{step}",
    monitor="ptl/val_loss",
    mode="min",
    save_top_k=1,
)

kwargs = recommend_timebase_kwargs(frame, freq="D", horizon=24, max_steps=150)
model = TimeBase(
    h=24,
    **kwargs,
    logger=logger,
    callbacks=[checkpoint],
    enable_checkpointing=True,
    enable_progress_bar=False,
    enable_model_summary=False,
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)

best_model = TimeBase.load_from_checkpoint(checkpoint.best_model_path)
best_nf = NeuralForecast(models=[best_model], freq="D")
forecast = best_nf.predict()
```

This pattern is often better than evaluating only the final in-memory weights, especially when validation improves early and then drifts.

## How to track underfitting or overfitting in practice

### 1. Inspect the logger output

With `CSVLogger`, PyTorch Lightning writes a `metrics.csv` file. Useful columns usually include:

- `train_loss_epoch`
- `valid_loss`
- `ptl/val_loss`
- `step`

A quick manual inspection often tells you most of what you need:

- if validation is still improving at the last step, the model may be undertrained
- if validation peaks early and then worsens, you are seeing overfitting or instability
- if both train and validation are noisy, the optimization setup may be too aggressive

### 2. Compare best validation step vs final step

If the best checkpoint happens much earlier than the final step, that is a strong signal that final-weight evaluation is misleading.

### 3. Compare train and validation qualitatively

A perfect train/validation gap heuristic does not always apply to forecasting, but the following still helps:

- **train down, validation down**: healthy
- **train down, validation flat**: likely underfitting or poor model match
- **train down, validation up**: overfitting
- **both noisy**: unstable optimization

## Common fixes

### Fix underfitting

Try one or more of:

```python
better_kwargs = recommend_timebase_kwargs(frame, freq="D", horizon=24, max_steps=250)

model = TimeBase(
    h=24,
    **better_kwargs,
    input_size=max(64, better_kwargs["input_size"]),
    basis_num=max(6, better_kwargs["basis_num"]),
)
```

Typical reasons this helps:

- more optimization steps let the model actually reach a good region
- larger `input_size` gives more history context
- larger `basis_num` can represent more repeated structure

### Fix overfitting or drift after the best validation point

Try:

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor="ptl/val_loss",
    mode="min",
    save_top_k=1,
)
early_stopping = EarlyStopping(
    monitor="ptl/val_loss",
    mode="min",
    patience=20,
)

model = TimeBaseTrend(
    h=24,
    **recommend_timebase_trend_kwargs(frame, freq="D", horizon=24, max_steps=200),
    learning_rate=1e-3,
    callbacks=[checkpoint, early_stopping],
    enable_checkpointing=True,
)
```

Typical reasons this helps:

- lower LR reduces drift and overshooting
- early stopping prevents wasting updates after validation has peaked
- checkpointing preserves the best state even if later steps are worse

### Fix non-convergence or unstable curves

If losses oscillate or validation is best immediately, try:

```python
stable_kwargs = recommend_timebase_kwargs(frame, freq="D", horizon=24, max_steps=200)

model = TimeBase(
    h=24,
    **stable_kwargs,
    learning_rate=min(stable_kwargs["learning_rate"], 1e-3),
    val_check_steps=10,
    early_stop_patience_steps=30,
)
```

Typical reasons this helps:

- smaller LR gives smoother optimization
- more frequent validation gives better visibility
- longer patience avoids stopping during harmless short-term noise

## Choosing between `TimeBase` and `TimeBaseTrend`

A practical rule of thumb:

- use `TimeBase` when repeated structure is the main signal
- use `TimeBaseTrend` when there is visible trend plus repeated structure

If `TimeBase` seems to fit the seasonal pattern but systematically misses broad movement, try `TimeBaseTrend`.

If `TimeBaseTrend` is not helping, check whether the decomposition branch is actually justified by the data before increasing model complexity further.

## Track results outside training loss too

Loss curves are necessary, but not sufficient. Also watch:

- holdout MAE / RMSE
- relative MAE if series scales vary a lot
- train time
- inference time
- representative forecast plots

This repository's benchmark scripts and HTML reports are useful for that broader inspection.

## Synthetic experiment assets

The repository exposes `timebaseula.make_synthetic_series(...)` as a deterministic generator reused by tests, scripts, and docs.

### Easy scenario

![Easy synthetic series](img/synthetic_series_easy.png)

### Medium scenario

![Medium synthetic series](img/synthetic_series_medium.png)

### Hard scenario

![Hard synthetic series](img/synthetic_series_hard.png)

## Long-horizon benchmark workflow

Prepare cached aggregates:

```bash
uv run --frozen python scripts/generate_datasets.py main
```

Quick verification benchmark:

```bash
uv run --frozen python scripts/benchmark_long_horizon.py run \
  --mode daily \
  --n-series 5 \
  --horizon 7 \
  --max-steps 10 \
  --skip-arima \
  --output logs/benchmark_results_smoke.csv \
  --html-report
```

## Final advice

If you are unsure whether the model is failing because of architecture or training setup, use this order:

1. recommendation helpers
2. non-zero `val_size`
3. validation logging
4. best-checkpoint evaluation
5. only then manual hyperparameter changes

That process will prevent many false conclusions about underfitting, overfitting, or model quality.
