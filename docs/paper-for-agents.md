---
description: Agent-friendly markdown digest of the TimeBase paper with instructions for LLM readers.
---

# TimeBase paper for agents

**TL;DR**
- Read this page first if you need the paper's main ideas quickly.
- Use `docs/huang25az.pdf` only when you need the original wording or figures.
- The method is: segment history, learn a compact basis, forecast future segments, flatten back to the target horizon.

## LLM instructions

When using this repository as an agent or LLM:

1. Prefer this markdown digest over the PDF for quick reasoning.
2. Treat the repository code as the source of truth for implementation details.
3. Do not claim support for features that are not exported or tested.
4. For single-series prediction after multi-series training, recommend the native NeuralForecast `predict(df=...)` flow.
5. When describing TimeBaseTrend, say it adds a decomposition-based trend branch; do not describe a learnable trend blend because this repo does not implement one.

## Core paper ideas

The TimeBase paper argues that long-horizon forecasting can be done with a very small model because many time series contain repeated temporal patterns.

Core steps:

1. split the input history into segments of length `P`
2. represent the segmented history with a low-rank basis of size `R`
3. forecast future segments from that basis
4. flatten the predicted segments back to horizon `L`

## Key equations in plain English

| Paper concept | Plain-language meaning |
|---|---|
| `Segment[N, P](X)` | reshape the history into `N` segments of length `P` |
| `BasisExtract(Xhis)` | linear map from segmented history to basis space |
| `SegmentForecast(Xbasis)` | linear map from basis space to future segments |
| `Flatten(Xpred)[:L]` | turn segment forecasts back into the final horizon |
| `Lorth` | penalty that discourages redundant basis vectors |

## What this repository implements

| Item | Status |
|---|---|
| TimeBase core idea | implemented |
| orthogonal penalty | implemented |
| NeuralForecast wrapper | implemented |
| recommendation helpers | implemented |
| markdown paper digest | implemented |
| custom single-series helper | intentionally removed |

## Mapping to repository code

| Concept | File |
|---|---|
| model wrappers | `timebaseula/models/timebase.py` |
| recommendation helpers | `timebaseula/recommend.py` |
| synthetic generator | `timebaseula/synthetic.py` |
| benchmark workflow | `scripts/benchmark_long_horizon.py` |

## Practical reading advice

- Use `TimeBase` when repeated seasonal structure dominates.
- Use `TimeBaseTrend` when the series has both trend and repeating structure.
- Use the recommendation helpers for dataset-dependent defaults instead of copying fixed hyperparameters blindly.
