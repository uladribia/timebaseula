---
description: Contribution guide for TimeBaseUla development, testing, and documentation updates.
---

# Contribute

**TL;DR**
- Clone the repository and use `uv sync`.
- Run `make format`, `make lint`, and `make test` before opening a PR.
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

Optional heavier checks:

```bash
make test-integration
make test-benchmark
```

## Testing policy

| Suite | Purpose |
|---|---|
| `make test` | fast default unit suite |
| `make test-integration` | actual NeuralForecast fitting behavior |
| `make test-benchmark` | benchmark-oriented checks when present |

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
