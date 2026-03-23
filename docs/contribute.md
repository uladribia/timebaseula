---
description: Contribution guide for TimeBaseUla development, testing, and documentation updates.
---

# Contribute

## TL;DR
- Clone the repository and use `uv sync`.
- Run `make format`, `make lint`, and `make test` before opening a PR.
- Run `make test-integration` only when your change affects real fit or predict behavior.
- Keep changes small, readable, CPU-friendly, and documented.

## Development setup

```bash
git clone https://github.com/dribia/timebaseula.git
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

## Documentation commands

```bash
make docs
make docs-serve
```
