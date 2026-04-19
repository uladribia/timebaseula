---
description: Contribution guide for TimeBaseUla development, testing, and documentation updates.
---

# Contribute

## TL;DR
- Clone the repository and use `uv sync`.
- Run `make format`, `make lint`, and `make test` before opening a PR.
- Run `make test-integration` only when your change affects real fit or predict behavior.
- Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/); keep
  reusable strategies in `tests/property_strategies.py` instead of duplicating them
  across test files.
- Keep changes small, readable, CPU-friendly, and documented.

## Development setup

```bash
git clone https://github.com/uladribia/timebaseula.git
cd timebaseula
uv sync
```

## Required checks

```bash
make format
make lint
make test
```

## Optional heavier check

```bash
make test-integration
```

## Testing policy

| Suite | Purpose |
|---|---|
| `make test` | fast default non-integration suite |
| `make test-integration` | real NeuralForecast fitting behavior |

## Contribution expectations

- follow the existing Python style
- keep identifiers and docs in English
- prefer clear, small changes
- update documentation when behavior changes
- preserve CPU-first compatibility
- avoid putting reusable package logic under `tests/`

## Library architecture

| Module | Responsibility |
|---|---|
| `timebaseula/models/core.py` | pure Torch segmented-basis model code |
| `timebaseula/models/decomposition.py` | pure Torch moving-average decomposition for `TimeBaseTrend` |
| `timebaseula/models/base.py` | shared NeuralForecast wrapper behavior |
| `timebaseula/models/defaults.py` | defaults and small validation helpers |
| `timebaseula/models/factories.py` | shared explicit-model construction helpers |
| `timebaseula/models/timebase.py` | public `TimeBase` and `TimeBaseTrend` wrappers |

Keep pure model math out of the NeuralForecast wrapper layer when possible. The local decomposition module is intentional so `TimeBaseTrend` does not depend on DLinear internals, but it should stay close to the upstream contract so a future revert remains easy.

## Documentation commands

```bash
make docs
make docs-serve
```
