---
description: Short release-style summary of the latest simplification pass.
---

# Release notes

## TL;DR
- simplified the public API to four model classes
- removed synthetic helpers and custom recommendation utilities from the package surface
- replaced the handwritten auto-search layer with thin Nixtla-style auto wrappers
- simplified benchmark tooling to CSV and markdown outputs

## Library changes

- kept `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`
- added deterministic defaults to `TimeBase` and `TimeBaseTrend`
- rebuilt the auto wrappers on top of NeuralForecast's `BaseAuto`
- removed package-level synthetic utilities and recommendation helpers

## Devtools changes

- removed synthetic benchmark and plotting scripts
- removed the custom HTML reporting layer
- kept only dataset preparation and the two benchmark entrypoints
- standardized benchmark output around CSV and markdown

## Why this change happened

The repository had accumulated too much custom orchestration around features that already exist in the Nixtla stack. The current direction is:
- use Nixtla primitives first
- keep the publishable library small
- keep devtools internal and simple
