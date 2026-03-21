# TimeBaseUla

<p style="text-align: center; padding-bottom: 1rem;">
    <a href="/timebaseula">
        <img
            src="../img/logo_dribia_blau_cropped.png"
            alt="Dribia"
            style="display: block; margin-left: auto; margin-right: auto; width: 40%;"
        >
    </a>
</p>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CI/CD   | [![Tests](https://github.com/dribia/timebaseula/actions/workflows/test.yml/badge.svg)](https://github.com/dribia/timebaseula/actions/workflows/test.yml) [![Coverage Status](https://img.shields.io/codecov/c/github/dribia/timebaseula)](https://codecov.io/gh/dribia/timebaseula) [![Tests](https://github.com/dribia/timebaseula/actions/workflows/lint.yml/badge.svg)](https://github.com/dribia/timebaseula/actions/workflows/lint.yml) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Package | [![PyPI](https://img.shields.io/pypi/v/timebaseula)](https://pypi.org/project/timebaseula/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/timebaseula?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/timebaseula?logo=python&logoColor=gold) [![GitHub](https://img.shields.io/github/license/dribia/timebaseula?color=blue)](https://github.com/dribia/timebaseula/blob/main/LICENSE) |

<p style="text-align: center;">
    <em>Port of TimeBase to pythonic and Dribia standards to be used with Nixtla's NeuralForecast</em>
</p>

---

**Documentation**: <a href="https://dribia.github.io/timebaseula" target="_blank">https://dribia.github.io/timebaseula</a>

**Source Code**: <a href="https://github.com/dribia/timebaseula" target="_blank">https://github.com/dribia/timebaseula</a>

---

## Overview

**TimeBaseUla** is a Python library implementing the TimeBase forecasting method, ported to Pythonic and Dribia standards for use with [Nixtla's NeuralForecast](https://nixtla.github.io/neuralforecast/).

TimeBase is a minimalistic LTSF (Long-Term Sequence Forecasting) model that leverages segment-level forecasting and basis extraction with two linear layers.

## Key Features

- **TimeBase**: Core model using segment-level forecasting with learned basis components
- **TimeBaseTrend**: Enhanced model combining trend decomposition with TimeBase basis forecasting
- **NeuralForecast Compatible**: Full integration with Nixtla's NeuralForecast API
- **Orthogonal Regularization**: Optional basis orthogonalization for improved representation learning
- **CPU-First Design**: Optimized for CPU execution with no GPU dependencies
- **Multivariate Support**: Train on multiple series simultaneously with channel independence

## Installation

**timebaseula** is available on PyPI:

```shell
pip install timebaseula
```

Or install from source:

```shell
git clone https://github.com/dribia/timebaseula.git
cd timebaseula
uv sync
```

## Quickstart

### Univariate Forecasting

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

# Load your data (requires 'ds', 'y', 'unique_id' columns)
df = pd.read_csv('your_data.csv')

model = TimeBase(
    h=24,              # Forecast horizon
    input_size=48,     # Input window size
    period_len=24,     # Period length
    basis_num=6,       # Basis components
)

nf = NeuralForecast(models=[model], freq='D')
nf.fit(df)
predictions = nf.predict()
```

### Multivariate Forecasting

```python
from neuralforecast import NeuralForecast
from timebaseula import TimeBase, TimeBaseTrend

# Multiple series with same timestamps
df = pd.read_csv('multivariate_data.csv')

# Train multiple models
models = [
    TimeBase(h=24, input_size=48, period_len=24, basis_num=6),
    TimeBaseTrend(h=24, input_size=48, period_len=24, moving_avg_window=25),
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df)
predictions = nf.predict()  # Predictions for all series
```

### Single Series Prediction from Multivariate Model

```python
from timebaseula import predict_single_series

# After training on multivariate data
single_series = df[df['unique_id'] == 'series_1']

pred = predict_single_series(
    model=model,
    series=single_series,
    h=24,
    input_size=48,
    freq='D'
)
```

## Examples

See [Usage Guide](usage.md) for detailed examples including:

- Synthetic series generation for testing
- MAE benchmark results
- Model configuration options

## Contributing

Contributions are welcome! Please see [Contributing Guide](contribute.md) for details.

## License

MIT License. See [LICENSE](https://github.com/dribia/timebaseula/blob/main/LICENSE) for details.
