---
description: Script reference for the AirPassengers benchmark workflow.
---

# Scripts

## TL;DR
- The repository includes one benchmark script: `scripts/benchmark_airpassengers.py`.
- It benchmarks `TimeBase`, `TimeBaseTrend`, `NLinear`, `DLinear`, `AutoMFLES`, and `Naive`.
- It uses small model-specific neural settings tuned for `AirPassengersPanel`.
- It writes a docs-ready markdown report and a Matplotlib plot.
- It uses Typer, Rich, and rotating logs.

## Install script dependencies

```bash
uv sync --group benchmark
```

The benchmark group is intended for Python 3.12+ on non-Windows environments.

## Run the benchmark

```bash
uv run --group benchmark python scripts/benchmark_airpassengers.py run \
  --output-markdown docs/benchmark.md \
  --output-plot docs/img/airpassengers-benchmark.png
```

## Outputs

| Path | Purpose |
|---|---|
| `docs/benchmark.md` | markdown report with metrics and embedded image |
| `docs/img/airpassengers-benchmark.png` | forecast plot for both series |
| `logs/benchmark_airpassengers.log` | rotating execution log |

## Metrics in the report

| Metric | Meaning |
|---|---|
| `MAE` | mean absolute error |
| `RMSE` | root mean squared error |
| `RMAE` | model MAE divided by the `Naive` MAE |
| `parameters` | trainable parameter count for neural models |
| `runtime_seconds` | end-to-end fit and predict time |
