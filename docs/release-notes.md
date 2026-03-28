---
description: Release-style summary for the branch split, anonymized benchmark docs, and version 0.3.0.
---

# Release notes

## TL;DR
- bumped the package version to `0.3.0`
- split the repository into long-lived `main` and `benchmark` branches
- kept `main` focused on the library plus curated benchmark result pages
- kept `benchmark` focused on reproducible benchmarking and tuning workflows
- reviewed benchmark docs to keep the internal daily-panel write-up anonymized

## What changed

| Area | Summary |
|---|---|
| Branches | documented and enforced the `main` vs `benchmark` split |
| Documentation | added branch-strategy guidance and refreshed benchmark wording |
| Benchmark reporting | kept published result pages while removing direct business labels from docs |
| Release metadata | bumped the package version to `0.3.0` |

## Branch roles

| Branch | Role |
|---|---|
| `main` | release-oriented library branch with curated benchmark reports |
| `benchmark` | full benchmarking and tuning branch with reproducibility tooling |

## Anonymization policy

The internal daily-panel benchmark pages now use anonymized wording in user-facing documentation.
The published plots use generic series aliases and anonymized units.

## Public API status

The exported package API remains:
- `TimeBase`
- `TimeBaseTrend`
