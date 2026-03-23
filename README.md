---
description: TimeBaseUla README with installation, public API, defaults, and package usage.
---

# TimeBaseUla

> Compact TimeBase-style forecasting models for NeuralForecast.

## TL;DR
- `timebaseula` is a Python library, not an application repository.
- Public API: `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, `AutoTimeBaseTrend`.
- The core implementation lives in `timebaseula/models/timebase.py`.
- The package is designed for CPU-first usage and plugs into `NeuralForecast`.
- This README was updated after an agent-made repository cleanup.

## Installation

### From PyPI

```bash
pip install timebaseula
```

### From source

```bash
git clone https://github.com/dribia/timebaseula.git
cd timebaseula
uv sync
```

## What this package provides

| Object | Purpose |
|---|---|
| `TimeBase` | Explicit TimeBase model with simple deterministic defaults |
| `TimeBaseTrend` | TimeBase model with an additional trend decomposition branch |
| `AutoTimeBase` | Nixtla-style auto wrapper for `TimeBase` |
| `AutoTimeBaseTrend` | Nixtla-style auto wrapper for `TimeBaseTrend` |

## Quickstart

```python
import pandas as pd
from neuralforecast import NeuralForecast
from timebaseula import TimeBase

frame = pd.DataFrame(
    {
        "unique_id": ["series_1"] * 200,
        "ds": pd.date_range("2024-01-01", periods=200, freq="D"),
        "y": range(200),
    }
)

model = TimeBase(h=24, freq="D", max_steps=100)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(frame, val_size=24)
forecast = nf.predict()
```

The expected data format is the standard `NeuralForecast` long format:

| Column | Meaning |
|---|---|
| `unique_id` | series identifier |
| `ds` | timestamp |
| `y` | target value |

## Default behavior

### `TimeBase` and `TimeBaseTrend`

| Parameter | Default |
|---|---|
| `input_size` | `max(2 * h, 8)` |
| daily `period_len` | `7` |
| monthly `period_len` | `12` |
| other `period_len` | `min(max(2, h), input_size)` |
| `basis_num` | `6` |
| `moving_avg_window` (`TimeBaseTrend`) | `25` |

### `AutoTimeBase` and `AutoTimeBaseTrend`
- subclass NeuralForecast's `BaseAuto`
- use Ray Tune through NeuralForecast's native auto infrastructure
- default to CPU execution with `gpus=0`
- search over a compact set of structural and training hyperparameters

## Repository layout

| Path | Role |
|---|---|
| `timebaseula/` | publishable library code |
| `docs/` | MkDocs documentation |
| `tests/` | repository test suite |
| `pyproject.toml` | package metadata and dependencies |

## Documentation

The documentation site covers:
- `docs/index.md`
- `docs/install.md`
- `docs/usage.md`
- `docs/models.md`
- `docs/release-notes.md`
- `docs/paper-for-agents.md`
- `docs/references.md`

Build the docs locally with:

```bash
uv run --group docs mkdocs build --strict
```

## Notes on repository scope

This repository is now library-first. It does not ship a user-facing CLI or the benchmark scripts that existed in earlier iterations, so package usage should start from Python imports such as:

```python
from timebaseula import TimeBase, TimeBaseTrend, AutoTimeBase, AutoTimeBaseTrend
```

## License

MIT. See [LICENSE](LICENSE).
