---
description: Extended markdown digest of the TimeBase paper, with plain-language explanations and implementation mapping.
---

# TimeBase paper for agents

## TL;DR
- This page is the fastest way to understand the idea behind TimeBase without starting from the PDF.
- TimeBase reshapes history into periods, learns a compact basis over those periods, forecasts future periods, and flattens them back to the horizon.
- The repository implementation should be treated as the source of truth for exported behavior.

## Who this page is for

This page is written for:
- contributors trying to understand the model before editing code
- AI agents that need a fast conceptual digest
- readers who want the main paper ideas in plain English before reading the PDF

If you need the original source, use `docs/huang25az.pdf`.
If you need the implemented behavior, prefer:
- `timebaseula/models/timebase.py`
- `docs/models.md`
- `docs/usage.md`

## Central idea in one sentence

TimeBase turns a long history into a sequence of periods, compresses those periods into a small basis, forecasts future periods from that basis, and then flattens the result into the final horizon.

## The paper's intuition

### 1. Segment the history

Instead of one long vector, the model treats the input as aligned segments of length `P`.

Examples:
- daily data with weekly structure: `P = 7`
- monthly data with yearly structure: `P = 12`

### 2. Learn a compact basis

The segmented history is projected into a small basis that summarizes recurring shapes.

### 3. Forecast future segments in basis space

The model predicts future segments from the compressed representation instead of forecasting every step independently from raw history.

### 4. Flatten back to the target horizon

The predicted future segments are flattened into the usual forecast vector of length `h`.

## Paper concepts in plain English

| Paper concept | Plain-language meaning |
|---|---|
| `Segment[N, P](X)` | reshape history into `N` chunks of length `P` |
| `BasisExtract(Xhis)` | project segmented history into a low-dimensional basis |
| `SegmentForecast(Xbasis)` | map the basis representation into future segments |
| `Flatten(Xpred)[:L]` | flatten predicted segments back into the final horizon |
| `Lorth` | regularization that discourages redundant basis vectors |

## Why this can work

The paper argues that many long-horizon tasks contain repeated temporal structure. If the architecture is biased toward that structure, a compact model can still be effective.

## TimeBaseTrend in repository context

The repository also exposes `TimeBaseTrend`, which adds a decomposition-style trend branch:
- the repeating component goes through the TimeBase branch
- the smoother trend component goes through a linear trend branch

## How the repository maps the paper to code

| Concept | Repository location |
|---|---|
| exported models | `timebaseula/models/timebase.py` |
| explicit defaults | `timebaseula/models/timebase.py` |
| auto wrappers | `timebaseula/models/timebase.py` |
| usage examples | `docs/usage.md` |

## What is implemented here

| Item | Status in this repository |
|---|---|
| TimeBase core segmented-basis idea | implemented |
| TimeBaseTrend decomposition extension | implemented |
| orthogonal penalty option | implemented |
| NeuralForecast compatibility | implemented |
| deterministic explicit defaults | implemented |

## Where to be careful

Do not assume:
- that every detail from the paper is reproduced exactly
- that paper-level claims automatically transfer to every dataset
- that the paper is a substitute for reading the code

Do say:
- the repository follows the paper's central segmented-basis intuition
- the exported models are practical NeuralForecast-compatible implementations
- the implementation in this repository is the source of truth for current behavior
