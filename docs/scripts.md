---
description: Reference for the internal benchmark and dataset-preparation scripts.
---

# Scripts

## TL;DR
- Operational scripts are thin Typer wrappers over internal `devtools/` modules.
- Only the dataset-preparation and benchmark entrypoints remain.
- Benchmarks write CSV, markdown, plot, and optional PDF outputs.
- Neural benchmark entries are `DLinear`, `NLinear`, `AutoTimeBase`, and `AutoTimeBaseTrend`.
- `--auto-num-samples` controls how many Ray Tune trials the auto wrappers sample.

## Available scripts

| Script | Purpose |
|---|---|
| `scripts/generate_datasets.py` | prepare cached long-horizon datasets |
| `scripts/benchmark_long_horizon.py` | benchmark ECL and TrafficL and write CSV/markdown/plot/PDF outputs |
| `scripts/benchmark_custom.py` | benchmark the custom monthly dataset and write CSV/markdown/plot/PDF outputs |

## Prepare cached benchmark datasets

```bash
uv run --frozen python scripts/generate_datasets.py main
```

Force refresh:

```bash
uv run --frozen python scripts/generate_datasets.py main --force-download
```

## Benchmark long-horizon datasets

```bash
uv run --frozen python scripts/benchmark_long_horizon.py run \
  --mode daily \
  --n-series 50 \
  --auto-num-samples 1 \
  --output logs/benchmark_long_horizon_daily.csv \
  --output-md logs/benchmark_long_horizon_daily.md \
  --output-pdf logs/benchmark_long_horizon_daily.pdf
```

This command writes:
- a CSV leaderboard with `mae`, `rmse`, `rmae`, `params`, and `execution_time`
- a markdown report with metric notes, a data summary, and representative forecast plots
- a sibling plot directory next to the markdown report
- an optional PDF export rendered with headless Chrome and embedded plot images

The auto wrappers use their Ray Tune search spaces with CPU-safe benchmark overrides.
Preset targets are:
- `smoke`: `max_steps=1`, `auto_num_samples=1`
- `normal`: `max_steps=10`, `auto_num_samples=2` (~2 CPU minutes)
- `thorough`: `max_steps=20`, `auto_num_samples=4` (~5 CPU minutes)

## Benchmark the custom dataset

```bash
uv run --frozen python scripts/benchmark_custom.py \
  --max-steps 30 \
  --auto-num-samples 1 \
  --output-dir logs/custom_dataset_benchmark \
  --output-pdf logs/custom_dataset_benchmark/report.pdf
```

This command writes:
- `leaderboard.csv`
- `report.md`
- `plots/*.png`
- `report.pdf`

Both benchmark CLIs always run cross-validation with `refit=True`.

## Logging

These scripts use rotating log files with a 5 MB limit:
- `logs/generate_datasets.log`
- `logs/benchmark_long_horizon.log`
- `logs/benchmark_custom.log`
