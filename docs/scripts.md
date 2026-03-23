---
description: Current repository scope after the cleanup to a library-first package.
---

# Repository scope

## TL;DR
- This repository is now focused on the `timebaseula` Python package.
- Historical operational scripts are no longer part of the tracked source tree.
- Use the library through Python imports and `NeuralForecast`.
- Package metadata, tests, and docs remain in the repository.

## What is in the repository

| Path | Purpose |
|---|---|
| `timebaseula/` | library source code |
| `tests/` | unit and integration tests |
| `docs/` | MkDocs documentation |
| `pyproject.toml` | package metadata and dependencies |
| `Makefile` | common development commands |

## What is not part of the tracked package workflow

The repository no longer documents or ships operational benchmark or dataset-preparation entrypoints as part of the current library workflow.

If you want to use TimeBaseUla, start from the package API instead:

```python
from timebaseula import TimeBase, TimeBaseTrend, AutoTimeBase, AutoTimeBaseTrend
```

## Main development commands

```bash
make format
make lint
make test
make docs
```
