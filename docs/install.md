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

Example:

```python
import pandas as pd

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 5,
        "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
        "y": [10.0, 11.0, 12.0, 13.0, 14.0],
    }
)
```

## Quality commands for contributors

```bash
make format
make lint
make test
```

## Build the HTML docs

Build the static site:

```bash
make docs
```

Generated files are written to:

```text
site/
```

Run the local docs server:

```bash
make docs-serve
```

Open:

```text
http://127.0.0.1:8000
```
