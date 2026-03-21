# TimeBaseUla

<p align="center">
    <a href="https://dribia.github.io/timebaseula">
    <picture style="display: block; margin-left: auto; margin-right: auto; width: 40%;">
            <source
                media="(prefers-color-scheme: dark)"
                srcset="docs/img/logo_dribia_blanc_cropped.png"
            >
            <source
                media="(prefers-color-scheme: light)"
                srcset="docs/img/logo_dribia_blau_cropped.png"
            >
            <img
                alt="timebaseula"
                src="docs/img/logo_dribia_blau_cropped.png"
            >
        </picture>
    </a>
</p>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CI/CD   | [![Tests](https://github.com/dribia/timebaseula/actions/workflows/test.yml/badge.svg)](https://github.com/dribia/timebaseula/actions/workflows/test.yml) [![Coverage Status](https://img.shields.io/codecov/c/github/dribia/timebaseula)](https://codecov.io/gh/dribia/timebaseula) [![Tests](https://github.com/dribia/timebaseula/actions/workflows/lint.yml/badge.svg)](https://github.com/dribia/timebaseula/actions/workflows/lint.yml) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Package | [![PyPI](https://img.shields.io/pypi/v/timebaseula)](https://pypi.org/project/timebaseula/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/timebaseula?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/timebaseula?logo=python&logoColor=gold) [![GitHub](https://img.shields.io/github/license/dribia/timebaseula?color=blue)](LICENSE)                                                                                                                                                                                                                                                                                                         |

---

**Documentation**: <a href="https://dribia.github.io/timebaseula" target="_blank">https://dribia.github.io/timebaseula</a>

**Source Code**: <a href="https://github.com/dribia/timebaseula" target="_blank">https://github.com/dribia/timebaseula</a>

---

## Overview

**TimeBaseUla** is a Python library implementing the TimeBase forecasting method, ported to Pythonic and Dribia standards for use with [Nixtla's NeuralForecast](https://nixtla.github.io/neuralforecast/).

TimeBase is a minimalistic LTSF (Long-Term Sequence Forecasting) model that leverages segment-level forecasting and basis extraction with two linear layers. It's designed to be ultra-lightweight while maintaining competitive accuracy.

## Key Features

- **TimeBase**: Core model using segment-level forecasting with learned basis components
- **TimeBaseTrend**: Enhanced model combining trend decomposition with TimeBase basis forecasting
- **NeuralForecast Compatible**: Full integration with Nixtla's NeuralForecast API
- **Orthogonal Regularization**: Optional basis orthogonalization for improved representation learning
- **CPU-First Design**: Optimized for CPU execution with no GPU dependencies
- **Multivariate Support**: Train on multiple series simultaneously with channel independence

## Installation

### From PyPI (recommended)

```shell
pip install timebaseula
```

### From source with uv

```shell
# Clone the repository
git clone https://github.com/dribia/timebaseula.git
cd timebaseula

# Install with uv
uv sync
```

## Quickstart

### Training a Univariate Model

Train TimeBase on a single time series:

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

# Load your univariate data
df = pd.read_csv('your_data.csv')  # Must have 'ds', 'y', 'unique_id' columns

# Create and train model
model = TimeBase(
    h=24,              # Forecast horizon
    input_size=48,     # Input window size
    period_len=24,     # Period length for segmentation
    basis_num=6,       # Number of basis components
    max_steps=500,
    learning_rate=1e-3,
)

nf = NeuralForecast(models=[model], freq='D')
nf.fit(df)

# Predict
predictions = nf.predict()
```

### Training a Multivariate Model

Train on multiple aligned series simultaneously:

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase, TimeBaseTrend

# Load multivariate data (multiple series with same timestamps)
df = pd.read_csv('multivariate_data.csv')
# Columns: 'ds', 'unique_id', 'y'
# Multiple 'unique_id' values for different series

# TimeBase for multivariate forecasting
model = TimeBase(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
)

# Or use TimeBaseTrend for series with strong trend components
trend_model = TimeBaseTrend(
    h=24,
    input_size=48,
    period_len=24,
    basis_num=6,
    moving_avg_window=25,  # For trend extraction
)

nf = NeuralForecast(models=[model, trend_model], freq='D')
nf.fit(df)

# Predict all series
predictions = nf.predict()
```

### Predicting a Single Series from Multivariate Model

After training on multivariate data, predict for a specific series:

```python
from timebaseula import predict_single_series

# After training (see above)
# Extract predictions for a specific series
single_series_df = df[df['unique_id'] == 'series_1'].copy()

pred = predict_single_series(
    model=model,           # Trained model
    series=single_series_df,
    h=24,
    input_size=48,
    freq='D'
)
```

## API Reference

### TimeBase

```python
TimeBase(
    h: int,                    # Forecast horizon
    input_size: int,           # Input window size
    period_len: int = 24,      # Segment period length
    basis_num: int = 6,        # Number of basis components
    use_period_norm: bool = True,    # Normalize per period
    use_orthogonal: bool = False,    # Enable orthogonal regularization
    orthogonal_weight: float = 0.0,  # Orthogonal loss weight
    # ... NeuralForecast standard params
)
```

### TimeBaseTrend

```python
TimeBaseTrend(
    h: int,
    input_size: int,
    period_len: int = 24,
    basis_num: int = 6,
    moving_avg_window: int = 25,  # Must be odd
    use_period_norm: bool = True,
    use_orthogonal: bool = False,
    orthogonal_weight: float = 0.0,
    # ... NeuralForecast standard params
)
```

### predict_single_series

```python
predict_single_series(
    model: TimeBase | TimeBaseTrend,
    series: pd.DataFrame,
    h: int,
    input_size: int,
    freq: str = "D",
) -> pd.DataFrame
```

## License

timebaseula is distributed under the terms of the [MIT](https://opensource.org/license/mit) license.
