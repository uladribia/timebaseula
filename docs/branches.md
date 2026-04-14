---
description: Repository branch strategy for the release-oriented, benchmark-oriented, and deprecated historical branches.
---

# Repository branches

## TL;DR
- `benchmark` is the canonical source branch.
- `main` is a curated subset for release-oriented library usage.
- `deprecated/library-v0.3.4` preserves the pre-multivariate library state.
- Only `benchmark` should carry full benchmark-generation workflows.

## Branch roles

| Branch | Purpose | Keep | Avoid | Status |
|---|---|---|---|---|
| `main` | release-oriented library branch | package code, curated docs, published benchmark result pages | benchmark-generation scripts, tuning scripts, benchmark-only scaffolding | active |
| `benchmark` | reproducible benchmark and tuning branch | scripts, tuning workflows, benchmark tests, workflow docs, artifacts | local data and transient logs | active |
| `deprecated/library-v0.3.4` | historical snapshot of the pre-multivariate library release | old library state for inspection and comparison | new features, fixes, or release work | deprecated |

## How to use them

- Use `benchmark` for benchmark and tuning development.
- Use `main` for release-oriented library work.
- Use `deprecated/library-v0.3.4` only when you need the old library layout or historical comparisons.
- Do not implement new shared functionality directly on `main`.

## Curation rule

When preparing `main`:
1. update and validate shared changes on `benchmark` first
2. sync shared files from `benchmark` into `main`
3. remove benchmark-only files from `main`
4. refresh `main` docs so they point users back to `benchmark` for reproducible workflows

## Documentation policy

- `benchmark` keeps the full workflow documentation and script reference.
- `main` keeps concise library docs and published benchmark result pages.
- Both active branches should mention the deprecated historical branch explicitly.
