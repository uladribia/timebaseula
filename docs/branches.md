---
description: Repository branch strategy for the release-oriented and benchmark-oriented branches.
---

# Repository branches

## TL;DR
- `main` is the release-oriented library branch.
- `benchmark` is the full benchmarking and tuning branch.
- Published benchmark reports may exist on both branches.
- Benchmark-generation scripts and experiment scaffolding live on `benchmark`.

## Branch roles

| Branch | Purpose | Keep | Avoid |
|---|---|---|---|
| `main` | library release and docs | package code, core tests, user docs, benchmark result pages | benchmark-generation scripts, tuning scripts, benchmark-only scaffolding |
| `benchmark` | research, benchmarking, tuning, reproducibility | scripts, tuning workflows, benchmark tests, workflow docs, benchmark artifacts | local data and transient logs |

## How to use them

- Use `benchmark` for benchmark and tuning development.
- Use `main` for release-oriented library work and publishable docs.
- When preparing `main`, curate from `benchmark` rather than maintaining two independent implementations.

## Documentation policy

- `main` keeps concise benchmark conclusions and published benchmark pages.
- `benchmark` keeps the full workflow documentation needed to reproduce those results.
- The `README.md` on `main` should point users to `benchmark` for benchmarking and tuning machinery.
