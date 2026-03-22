---
description: Extended markdown digest of the TimeBase paper, with plain-language explanations, implementation mapping, and reading guidance for agents and human readers.
---

# TimeBase paper for agents

**TL;DR**
- This page is the best starting point if you want to understand the **idea of TimeBase** without reading the full PDF first.
- The central proposal is simple: **reshape the history into periods, learn a compact basis over those periods, forecast future periods, and flatten back to the horizon**.
- The paper's pitch is that long-horizon forecasting does not always require a large deep model; a carefully structured low-rank model can work surprisingly well.
- In this repository, the implementation follows the paper's main intuition, but the code should be treated as the source of truth for exact exported behavior.

## Who this page is for

This page is written for:

- contributors trying to understand the model before editing code
- AI agents or LLMs that need a fast conceptual digest
- readers who want the main paper ideas in plain English before reading the PDF

If you need the exact original language, equations, or figures, use:

- `docs/huang25az.pdf`

If you need to understand what is actually implemented here, prefer:

- `timebaseula/models/timebase.py`
- `timebaseula/recommend.py`
- `docs/models.md`

## How to use this page as an agent or LLM

When reasoning about this repository:

1. use this page for the paper's conceptual summary
2. use the repository code for implementation details
3. do not assume every claim from the paper is fully reproduced here
4. describe `TimeBaseTrend` as a **decomposition-based extension with a trend branch**
5. for actual behavior exposed to users, trust tested code over paper wording

That distinction matters because papers often describe a broader family of ideas than what a practical library exports.

## The forecasting problem the paper is trying to solve

Long-horizon forecasting is hard because the model must predict many future steps while preserving broad temporal structure. In practice, this means the model must understand:

- repeated patterns
- slow trend changes
- how local windows connect to much longer future ranges

A common approach is to increase model depth or width. The TimeBase paper instead asks whether a more **structured representation** can reduce the need for large capacity.

The key observation is that many time series contain repeated temporal units:

- days in weekly data
- months in yearly seasonal data
- recurring blocks in long synthetic or industrial signals

If those blocks repeat, maybe the right representation is not one long flat vector, but a matrix of aligned periods.

## The central idea in one sentence

TimeBase turns a long history into a sequence of periods, compresses those periods into a small basis, forecasts future periods from that basis, and then flattens the result into the final horizon.

## The paper's intuition, step by step

### 1. Segment the history

Instead of viewing the input as:

- one vector of length `T`

TimeBase views it as:

- `N` segments of length `P`

where `P` is the period length and `N` is the number of segments that fit in the input window.

Example:

- daily data with weekly structure: `P = 7`
- monthly data with yearly structure: `P = 12`

This is the step that injects the paper's main inductive bias.

### 2. Learn a compact basis

Once history is represented as aligned segments, the model asks:

- can a small number of latent basis components summarize these repeated blocks?

This basis is meant to capture recurring shape patterns more efficiently than a generic dense representation.

### 3. Forecast future segments in basis space

Instead of forecasting every future time step independently from the raw history, the model predicts future segments from the compressed basis.

This is meant to make long-horizon forecasting easier because the model is operating in a representation aligned with periodic structure.

### 4. Flatten back to the target horizon

The model finally converts future predicted segments back into the usual forecast vector of length `L` or `h`.

So the user still receives a standard horizon forecast, even though the internal representation is segmented.

## The key equations in plain English

This section maps paper-style notation to practical meaning.

| Paper concept | Plain-language meaning |
|---|---|
| `Segment[N, P](X)` | reshape the input history into `N` chunks of length `P` |
| `BasisExtract(Xhis)` | project segmented history into a low-dimensional basis |
| `SegmentForecast(Xbasis)` | map the basis representation into future segments |
| `Flatten(Xpred)[:L]` | flatten predicted segments back into the final forecast horizon |
| `Lorth` | regularization term that discourages redundant basis vectors |

The important point is not the notation itself, but the representation change:

- from raw history space
- to segmented period space
- to basis space
- back to forecast space

## Why this can work

The paper's argument is that many long-horizon tasks contain enough repeated structure that the model can be both:

- compact
- accurate

if it is biased toward the right temporal decomposition.

This is attractive because it suggests a path between two extremes:

- naive linear models that may be too simple
- large deep models that may be expensive or harder to train

TimeBase tries to keep the model small while still encoding something domain-relevant about temporal repetition.

## What makes TimeBase different from a generic linear model

At first glance, the implementation can look linear or lightweight. But the paper's innovation is not “just use linear layers.”

The important difference is the **structure of the transformation**:

- the model is built around segmented periods
- it forecasts through a low-rank basis
- the architecture is designed around repeated temporal blocks

So the right comparison is not simply:

- TimeBase versus deep nonlinear networks

but also:

- TimeBase versus flat direct linear projections that do not explicitly respect periodic segmentation

## The role of orthogonal regularization

The paper includes an orthogonal penalty to encourage the basis vectors to remain diverse rather than collapsing into redundant directions.

Intuitively, this means:

- different basis components should explain genuinely different patterns
- the latent representation should not waste capacity by learning nearly identical vectors

In this repository, that idea is available as optional orthogonal regularization.

## TimeBaseTrend in the paper context

The repository also exposes `TimeBaseTrend`, which is not just “TimeBase with more layers.”

Its motivation is practical:

- some series contain repeated structure
- but also contain a trend component that should be modeled separately

So `TimeBaseTrend` applies a decomposition-style split and lets:

- the seasonal/repeating part go through the TimeBase branch
- the smoother trend part go through a linear trend branch

This is especially useful when the repeated temporal block bias is still helpful, but not sufficient on its own.

## How the repository maps the paper to code

| Concept | Repository location |
|---|---|
| exported wrappers | `timebaseula/models/timebase.py` |
| dataset-aware defaults | `timebaseula/recommend.py` |
| synthetic benchmarking | `scripts/check_forecast_mae.py` |
| long-horizon real-data benchmarking | `scripts/benchmark_long_horizon.py` |
| custom HTML benchmark reporting | `scripts/reporting.py` and `scripts/benchmark_custom_dataset.py` |

## What is implemented here

| Item | Status in this repository |
|---|---|
| TimeBase core segmented-basis idea | implemented |
| TimeBaseTrend decomposition extension | implemented |
| orthogonal penalty option | implemented |
| NeuralForecast compatibility | implemented |
| recommendation helpers | implemented |
| benchmark scripts and reports | implemented |
| ad hoc single-series helper API | intentionally not exported |

## Where to be careful

When explaining this project, avoid overclaiming.

### Do not assume

- that every detail from the paper is reproduced exactly
- that paper-level claims automatically transfer to every dataset here
- that all good benchmark behavior is architectural rather than optimization-related

### Do say

- the repository follows the paper's central segmented-basis intuition
- the exported models are practical NeuralForecast-compatible implementations
- recommendation helpers and checkpoint-aware benchmarking are important for getting stable results

## Practical reading advice for humans

A good reading order is:

1. this page
2. `docs/models.md`
3. `timebaseula/models/timebase.py`
4. `timebaseula/recommend.py`
5. the original PDF if you want the full formal presentation

That order helps because:

- this page gives the motivation
- `docs/models.md` explains the implemented models more concretely
- the code shows what really exists today
- the PDF adds full paper context afterward

## Practical reading advice for contributors

If you are about to modify the implementation, keep these questions in mind:

- what period structure is the model assuming?
- does the chosen `period_len` make sense for the frequency?
- are optimization settings masking the architecture's real behavior?
- is the benchmark evaluating the final weights or the best validation checkpoint?

Those questions matter because compact structured models can look worse than they are if the training loop is not aligned with how they converge.

## Paper ideas translated into engineering questions

Here is a useful mental bridge from the paper to the repository.

| Paper idea | Engineering question in this repo |
|---|---|
| repeated temporal blocks matter | what should `period_len` be? |
| low-rank basis can summarize history | how many basis components should `basis_num` use? |
| trend can interfere with seasonal modeling | should this dataset use `TimeBaseTrend` instead of `TimeBase`? |
| compact models can still work well | are we training long enough and evaluating the right checkpoint? |

## What an interested reader should conclude

The paper is not arguing that every forecasting problem is simple. It is arguing something narrower and more interesting:

- when the data contains repeated temporal structure,
- and when the model is designed to respect that structure,
- a compact basis model may be enough for strong long-horizon performance.

That is the lens through which this repository should be understood.

## If you only remember five things

1. TimeBase is a **segmented low-rank forecasting model**.
2. Its main bias is to treat long histories as **repeated periods**.
3. `TimeBaseTrend` adds a separate **trend branch** through decomposition.
4. The recommendation helpers are part of making the model usable, not just convenience sugar.
5. The implementation and tests in this repository are the source of truth for current behavior.
