---
description: Detailed explanation of the TimeBase and TimeBaseTrend models, their assumptions, architecture, training behavior, and practical tuning advice.
---

# Models

**TL;DR**
- `TimeBase` is a compact long-horizon forecasting model that works by **splitting history into periods**, **compressing them into a basis**, and **projecting that basis into future periods**.
- `TimeBaseTrend` extends `TimeBase` with a **trend decomposition branch**, making it more suitable when a series contains both repeated structure and a persistent low-frequency trend.
- In this repository, both models are implemented as **NeuralForecast-compatible wrappers**, so they can be trained like other NeuralForecast models.
- The recommendation helpers in `timebaseula.recommend` are the preferred way to choose defaults for `input_size`, `period_len`, `basis_num`, `moving_avg_window`, and training hyperparameters.

## Why these models exist

Long-horizon forecasting is difficult because the model must predict many future steps while preserving broad structure:

- seasonal repetition
- slow trend movement
- interactions between local shape and long-range horizon behavior

Many popular neural baselines handle this with deep architectures or many parameters. The TimeBase idea is different:

1. assume the recent history contains **repeating temporal blocks**
2. reshape those blocks explicitly
3. learn a **low-dimensional representation** of them
4. forecast future blocks from that compact representation

The result is a model that can stay surprisingly small while still being competitive on long-range tasks.

## High-level intuition

Imagine a monthly series with yearly seasonality. Instead of treating the input as one long flat vector, TimeBase asks:

- what if the last few years are naturally grouped into 12-step chunks?
- what if those chunks can be summarized by a few reusable basis components?
- what if forecasting the next year is easier in that chunked space than in the raw flat space?

That is the core idea.

A useful mental model is:

- **segment first**
- **compress second**
- **forecast in compressed space**
- **flatten at the end**

## Shared implementation structure

The repository implements both models through a common internal wrapper compatible with NeuralForecast's window-based training flow.

Main code paths:

| File | Role |
|---|---|
| `timebaseula/models/timebase.py` | model wrappers and core layers |
| `timebaseula/recommend.py` | dataset profiling and default selection |
| `tests/test_timebase.py` | unit and integration coverage |

Both exported models behave like standard NeuralForecast models:

```python
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

nf = NeuralForecast(models=[TimeBase(h=24, input_size=48)], freq="D")
```

## `TimeBase`

### Concept

`TimeBase` models the forecast through two learned linear transformations applied in a segmented representation.

At a high level:

1. take the input window of length `input_size`
2. pad if needed so its length is divisible by `period_len`
3. reshape it into `N` segments of length `period_len`
4. optionally normalize each segment by its mean
5. map the segmented history to a low-dimensional basis of size `basis_num`
6. map that basis back to predicted future segments
7. flatten those predicted segments to the target horizon `h`

### Core layers

| Layer | Meaning |
|---|---|
| `ts2basis` | history-segment → basis projection |
| `basis2ts` | basis → future-segment projection |

This is why the model is so lightweight: the architecture is intentionally structured around a simple low-rank transformation instead of a deep stack of nonlinear blocks.

### Detailed forward intuition

Suppose:

- `input_size = 48`
- `period_len = 12`
- `h = 24`

Then the input history can be viewed as 4 periods of length 12. TimeBase tries to learn:

- which latent basis components explain those 4 periods
- how those components should generate the next 2 future periods

This is especially attractive when the data has repeated shape at roughly the chosen `period_len`.

### When `TimeBase` tends to work well

`TimeBase` is most natural when:

- seasonality is strong
- the main forecast challenge is long-horizon repeated structure
- the series does not require a highly expressive nonlinear trend model
- a compact, interpretable model is preferred

Typical good cases:

- daily series with weekly repetition
- monthly series with yearly seasonality
- long synthetic seasonal series

### Failure modes

`TimeBase` can struggle when:

- the chosen `period_len` does not match the real structure well
- the series has large nonstationary trend changes
- the optimization settings are too aggressive for the dataset
- the model reaches a good validation point early and then drifts away

In practice, if `TimeBase` shows unstable validation behavior, the first things to check are:

- learning rate
- training budget
- early stopping and best-checkpoint selection
- whether `period_len` is plausible for the frequency

## `TimeBaseTrend`

### Concept

`TimeBaseTrend` keeps the same seasonal/basis idea, but adds a separate trend branch.

Instead of asking a single component to model both:

- repeated seasonal structure
- low-frequency trend

it first decomposes the input into:

- a **seasonal-like component**
- a **trend-like component**

Then it models them separately.

### Forward structure

1. decompose the input using `SeriesDecomp`
2. send the seasonal component through `TimeBaseCore`
3. send the trend component through a simple linear trend layer
4. add both outputs together

So the final forecast is:

- seasonal forecast from the TimeBase basis branch
- plus trend forecast from the linear branch

### Why this helps

Plain `TimeBase` is best when repeated structure dominates. But some datasets have an important low-frequency component that should not be squeezed into the same basis representation.

`TimeBaseTrend` can be a better fit when:

- there is obvious trend plus seasonality
- seasonal structure is still important
- the trend is smooth enough for a simple decomposition + linear branch to help

### Main extra parameter

| Parameter | Meaning |
|---|---|
| `moving_avg_window` | odd window size used by the moving-average decomposition |

This must be odd because of how the centered moving average is constructed.

## Orthogonal regularization

Both models support an optional orthogonal penalty on the learned basis matrix.

### Motivation

If the basis vectors become too redundant, the low-dimensional representation may waste capacity by learning overlapping directions.

The orthogonal penalty encourages basis components to be more distinct.

### Parameters

| Parameter | Meaning |
|---|---|
| `use_orthogonal` | whether to include the orthogonal penalty |
| `orthogonal_weight` | strength of the penalty term |

### Practical advice

This is an advanced option. It is usually reasonable to start with it disabled unless you are explicitly experimenting with basis diversity.

## Recommendation helpers

This repository includes lightweight data profiling so users do not have to guess every parameter by hand.

Top-level helpers:

- `timebaseula.profile_dataset(...)`
- `timebaseula.recommend_timebase_kwargs(...)`
- `timebaseula.recommend_timebase_trend_kwargs(...)`

Class-level helpers:

- `TimeBase.profile_dataset(...)`
- `TimeBase.recommend_defaults(...)`
- `TimeBaseTrend.profile_dataset(...)`
- `TimeBaseTrend.recommend_defaults(...)`

### What they inspect

The profiler looks at a long-format frame with:

- `unique_id`
- `ds`
- `y`

and estimates:

- history lengths
- dominant period candidates
- seasonality strength
- trend strength
- a rough scale summary

### What they return

Depending on the helper, they recommend values for:

- `input_size`
- `period_len`
- `basis_num`
- `moving_avg_window`
- `max_steps`
- `learning_rate`
- `early_stop_patience_steps`
- `val_check_steps`

### Why this matters

The benchmarks in this repository repeatedly showed that poor training defaults can look like a model problem when they are really an optimization problem. The helpers are therefore not cosmetic: they are part of making these compact models usable across different regimes.

## Relationship to DLinear and other baselines

A useful comparison is:

| Model | Main bias |
|---|---|
| `DLinear` | decomposition + direct linear projection |
| `NLinear` | direct normalized linear projection |
| `TimeBase` | segmented low-rank basis forecast |
| `TimeBaseTrend` | segmented low-rank seasonal branch + linear trend branch |

So TimeBase is not just “another linear model”. Its distinguishing feature is the **explicit segment-and-basis view** of the history.

## Practical model selection advice

### Prefer `TimeBase` when

- repeated structure is the main signal
- you want a small model
- the frequency has a natural period candidate
- trend is present but not dominant

### Prefer `TimeBaseTrend` when

- the series has both repeated structure and visible trend
- plain `TimeBase` underfits the trend part
- decomposition is likely to help separate low- and high-frequency behavior

### Prefer recommendation helpers over fixed defaults

This is especially important when changing:

- frequency
- horizon
- history length
- dataset family

Blindly reusing one set of hyperparameters across daily, monthly, synthetic, and long-horizon benchmarks is usually a bad idea.

## How these models are trained here

In this repository, the models are typically trained through NeuralForecast's windowing pipeline, which means:

- data is converted into rolling training windows
- validation is monitored through NeuralForecast / Lightning metrics
- best-checkpoint evaluation is used in benchmark settings where convergence matters

That last point is important. Some models, especially compact ones, can reach their best validation point before the final optimization step. Benchmarks should therefore avoid evaluating only the last state when a better checkpoint is available.

## Testing and maintenance policy

This repository treats the models as library code, not notebook-only experiments.

### Unit tests cover

- shape behavior
- decomposition constraints
- recommendation logic
- helper functions

### Integration tests cover

- actual NeuralForecast fitting
- prediction behavior after training
- multi-series training with subset prediction

### Benchmark scripts cover

- real-data comparisons
- synthetic comparisons
- HTML report generation

## Suggested reading order

If you are new to the repository:

1. read `docs/paper-for-agents.md`
2. read this page
3. inspect `timebaseula/models/timebase.py`
4. inspect `timebaseula/recommend.py`
5. run one of the benchmark scripts to see the models in context

That sequence gives both the conceptual motivation and the implementation reality.
