---
description: Installation guide for TimeBaseUla with runtime, source, and benchmark setup instructions.
---

# Install TimeBaseUla

## TL;DR
- Use `pip install timebaseula` to consume the package.
- Use `uv sync` to work from a local checkout.
- Use `uv sync --group benchmark` when you want to run the AirPassengers benchmark script.
- Python requirement: `>=3.10,<3.15`.

## Runtime install

```bash
pip install timebaseula
```

## Source install

```bash
git clone https://github.com/dribia/timebaseula.git
cd timebaseula
uv sync
```

## Benchmark script setup

```bash
uv sync --group benchmark
```

The benchmark group is intended for Python 3.12+ on non-Windows environments.

## Main runtime dependencies

| Package | Why it is used |
|---|---|
| `neuralforecast` | training and forecasting interface |
| `torch` | model implementation |
| `pandas` / `numpy` | tabular and numerical processing |

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
