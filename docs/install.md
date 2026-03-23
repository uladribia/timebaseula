---
description: Installation guide for TimeBaseUla with runtime and source setup instructions.
---

# Install TimeBaseUla

## TL;DR
- Use `pip install timebaseula` to consume the package.
- Use `uv sync` to work from a local checkout.
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

## Main runtime dependencies

| Package | Why it is used |
|---|---|
| `neuralforecast` | training and forecasting interface |
| `torch` | model implementation |
| `pandas` / `numpy` | tabular and numerical processing |
| `ray[tune]` | auto-model hyperparameter search |

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
