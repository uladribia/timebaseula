---
description: Contribution guide for TimeBaseUla development, testing, and documentation updates.
---

# Contribute

**TL;DR**
- Clone the repository and use `uv sync`.
- Run `make format`, `make lint`, and `make test` before opening a PR.
- Keep changes small, readable, and CPU-friendly.

## Maintainers

TimeBaseUla is maintained by Dribia.

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

If you need integration coverage:

```bash
make test-integration
```

## Documentation commands

```bash
make docs
```

To preview the site locally:

```bash
make docs-serve
```

## Contribution expectations

- follow the existing Python style
- keep identifiers and docs in English
- prefer clear, small changes
- update documentation when behavior changes
- preserve CPU-first compatibility

## Project tools

| Task | Tool |
|---|---|
| dependency management | `uv` |
| formatting and linting | `ruff`, `tombi`, `ty` |
| tests | `pytest` |
| docs | `mkdocs-material` |

## Reporting issues

When filing a bug, include:

- package version
- Python version
- operating system
- minimal reproduction

```bash
python -c "import timebaseula; print(timebaseula.__version__)"
```
