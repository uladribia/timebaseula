---
description: Release-style summary for the benchmark branch split, anonymized benchmark docs, and version 0.3.0.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.0`
- formalized the long-lived `main` and `benchmark` branch split
- kept `benchmark` as the reproducible benchmarking and tuning branch
- refreshed benchmark docs to remove direct business labels from the internal daily-panel write-up
- preserved the public package API around `TimeBase` and `TimeBaseTrend`

## What changed

| Area | Summary |
|---|---|
| Branches | documented the `main` vs `benchmark` workflow clearly for users and agents |
| Benchmark workflows | kept the full preparation, benchmarking, and tuning scripts on `benchmark` |
| Documentation | replaced direct business-facing labels in benchmark docs with anonymized wording |
| Release metadata | bumped the package version to `0.3.0` |

## Branch roles

| Branch | Role |
|---|---|
| `main` | release-oriented library branch with curated benchmark reports |
| `benchmark` | full benchmarking and tuning branch with reproducibility tooling |

## Anonymization policy

The internal daily-panel benchmark pages now use anonymized wording in user-facing documentation.
Published plots use generic series aliases and anonymized units.
Workflow examples also use generic internal dataset paths in documentation.

## Public API status

The exported package API remains:
- `TimeBase`
- `TimeBaseTrend`
