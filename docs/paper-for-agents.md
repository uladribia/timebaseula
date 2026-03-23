---
description: Agent-friendly summary of the TimeBase paper and how this repository maps it to code.
---

# TimeBase paper for agents

## TL;DR
- TimeBase models long-horizon forecasting by learning a compact basis over temporal segments.
- The core idea is to capture repeated structure instead of relying on very large sequence models.
- This repository implements `TimeBase` and `TimeBaseTrend` for `NeuralForecast`.
- The code favors explicit defaults and readable model wrappers.

## Paper intuition

TimeBase divides an input window into repeated temporal segments, compresses those segments into a basis representation, and reconstructs future segments from that basis.

This makes the architecture a good fit for data with repeated structure across time.

## Core idea in plain language

1. split the history into segments
2. encode the segments into a compact basis
3. decode the basis into future segments
4. reshape the decoded segments back into a forecast horizon

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
| model wrappers | `timebaseula/models/timebase.py` |
| usage examples | `docs/usage.md` |

## What is implemented here

| Item | Status in this repository |
|---|---|
| TimeBase core segmented-basis idea | implemented |
| TimeBaseTrend decomposition extension | implemented |
| orthogonal penalty option | implemented |
| NeuralForecast compatibility | implemented |
| deterministic explicit defaults | implemented |

## What to inspect first as an agent

1. `timebaseula/models/timebase.py`
2. `timebaseula/__init__.py`
3. `docs/models.md`
4. `tests/unit/library/test_timebase.py`
