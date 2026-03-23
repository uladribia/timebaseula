---
description: Reference for the internal benchmark and dataset-preparation scripts.
---

# Scripts

## TL;DR
- Operational scripts are thin Typer wrappers over internal `devtools/` modules.
- Only the dataset-preparation and benchmark entrypoints remain.
- Benchmarks now write simple CSV and markdown outputs.

## Available scripts

| Script | Purpose |
|---|---|
| `scripts/generate_datasets.py` | prepare cached long-horizon datasets |
| `scripts/benchmark_long_horizon.py` | benchmark ECL and TrafficL and write CSV/markdown outputs |
| `scripts/benchmark_custom.py` | benchmark the custom monthly dataset and write CSV/markdown outputs |

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
  --output logs/benchmark_long_horizon_daily.csv \
  --output-md logs/benchmark_long_horizon_daily.md \
  --output-pdf logs/benchmark_long_horizon_daily.pdf
```

This command writes:
- a CSV leaderboard with `mae`, `rmse`, `rmae`, `params`, and `execution_time`
- a markdown report with metric notes, a data summary, and representative forecast plots
- a sibling plot directory next to the markdown report

## Benchmark the custom dataset

```bash
uv run --frozen python scripts/benchmark_custom.py \
  --max-steps 30 \
  --output-dir logs/custom_dataset_benchmark
```

This command writes:
- `leaderboard.csv`
- `report.md`
- `plots/*.png`
- `report.pdf` when `--output-pdf` is provided

Both benchmark CLIs always run cross-validation with `refit=True`.

## Logging

These scripts use rotating log files with a 5 MB limit:
- `logs/generate_datasets.log`
- `logs/benchmark_long_horizon.log`
- `logs/benchmark_custom.log`
stom.log`
g`
