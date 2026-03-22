---
description: Installation guide for TimeBaseUla with runtime and development setup instructions.
---

# Install TimeBaseUla

**TL;DR**
- Use `pip install timebaseula` to consume the package.
- Use `uv sync` to work on the repository.
- Python requirement: `>=3.10,<3.15`.

## Runtime install

```bash
pip install timebaseula
```

## Development install

```bash
git clone https://github.com/dribia/timebaseula.git
cd timebaseula
uv sync
```

## Main dependencies

| Package | Why it is used |
|---|---|
| `neuralforecast` | training and forecasting interface |
| `torch` | model implementation |
| `pandas` / `numpy` | tabular and numerical processing |
| `statsforecast` | classical baselines in scripts |
| `typer` + `rich` | command-line tooling |
| `mkdocs-material` | project documentation |

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

## Contributor quality gates

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

## Build the HTML docs

```bash
make docs
make docs-serve
```
