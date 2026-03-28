---
description: Installation guide for TimeBaseUla with source and branch-aware setup instructions.
---

# Install TimeBaseUla

## TL;DR
- Use `uv sync` from a local checkout.
- The main package includes the dependencies needed for `AutoTimeBase` and `AutoTimeBaseTrend`.
- Use `uv sync --group benchmark` only on the `benchmark` branch when you want benchmark workflows.
- Python requirement: `>=3.10,<3.15`.
- This repository does not publish `timebaseula` to PyPI.

## Source install

```bash
git clone https://github.com/uladribia/timebaseula.git
cd timebaseula
uv sync
```

## Benchmark workflow setup

```bash
uv sync --group benchmark
```

The benchmark group is intended for Python 3.12+ on non-Windows environments and is primarily useful on the `benchmark` branch.

## Main runtime dependencies

| Package | Why it is used |
|---|---|
| `neuralforecast` | training and forecasting interface |
| `torch` | model implementation |
| `pandas` / `numpy` | tabular and numerical processing |
| `ray[tune]` | backend for `AutoTimeBase` and `AutoTimeBaseTrend` |

## Benchmark-only dependencies

| Package | Why it is used |
|---|---|
| `statsforecast` | statistical baselines |
| `scikit-learn` | `AutoMFLES` requirement |
| `matplotlib` | benchmark plot generation |
| `typer` / `rich` | CLI and terminal rendering |

## Verify the install

```bash
python -c "import timebaseula; print(timebaseula.__version__)"
```

## Data format expected by the models

The examples in this repository use the standard `NeuralForecast` long format:

| Column | Meaning |
|---|---|
| `unique_id` | series identifier |
| `ds` | timestamp |
| `y` | target value |

## Build the HTML docs

```bash
uv run --group docs mkdocs build --strict
```
