---
description: Paper summary and references for TimeBaseUla, including the original PDF and markdown digest.
---

# Paper notes and references

**TL;DR**
- The repository includes the original paper as `docs/huang25az.pdf`.
- A readable markdown digest for agents is available at `paper-for-agents.md`.
- The core claim is that long-horizon series often show repeated patterns that can be forecast with a compact basis.

## Original paper in this repo

- `docs/huang25az.pdf`

## Readable markdown export

- [paper-for-agents.md](paper-for-agents.md)

## High-level method summary

The paper describes TimeBase as a pipeline with:

- segmentation using a period length `P`
- basis extraction from segmented history
- segment-level forecasting from the learned basis
- flattening back to the final horizon `L`
- optional orthogonal regularization on the basis matrix

## Repository-specific note

This repository adds a practical `TimeBaseTrend` variant that combines the TimeBase seasonal branch with a decomposition-based trend branch.

## Citation-style reference

```text
Huang, Q., Zhou, Z., Yang, K., Yi, Z., Wang, X., & Wang, Y. (2025).
TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting.
Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267.
```

## Related libraries used here

- NeuralForecast
- StatsForecast
- PyTorch
