---
description: Paper summary and references for TimeBaseUla, based on the original TimeBase PDF shipped with the repository.
---

# Paper notes and references

**TL;DR**
- The repository includes the original paper as `docs/huang25az.pdf`.
- The paper presents **TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting**.
- Its central claim is that long-horizon series often have repeated patterns and approximate low-rank structure, so a tiny model can still work well.

## Original paper in this repo

- `docs/huang25az.pdf`

## What the paper argues

After checking the PDF in this repository, the main ideas are:

1. long-term time series often show **temporal pattern similarity**
2. those repeated patterns can be modeled with a small number of **basis components**
3. forecasting can be done at the **segment level** instead of the point level
4. an **orthogonal restriction** can encourage basis diversity
5. the approach can also be used as a **plug-in reducer** for patch-based models

## High-level method summary

The paper describes TimeBase as a pipeline with:

- segmentation using a period length `P`
- basis extraction from segmented history
- segment-level forecasting from the learned basis
- flattening back to the final horizon `L`

This matches the structure implemented in `timebaseula/models/timebase.py`.

## Notes relevant to this repository

The repository implementation follows the paper in spirit, especially on:

- compact basis learning
- segment-level forecasting
- optional orthogonal regularization
- channel-independent handling of multiple series through the forecasting interface

It also adds a practical variant, `TimeBaseTrend`, that combines the TimeBase seasonal branch with a decomposition-based trend branch.

## Citation-style reference

```text
Huang, Q., Zhou, Z., Yang, K., Yi, Z., Wang, X., & Wang, Y. (2025).
TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting.
Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267.
```

## Related libraries used here

- NeuralForecast: training and prediction framework
- StatsForecast: classical baselines used in scripts
- PyTorch: model implementation backend
