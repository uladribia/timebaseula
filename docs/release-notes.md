---
description: Short release-style summary of the repository cleanup that kept the library and docs aligned.
---

# Release notes

## TL;DR
- kept the public API centered on four model classes
- retained the library source, tests, and documentation
- removed stale references to non-existent operational tooling
- simplified the repository to focus on the publishable package

## Repository changes

- kept `timebaseula/` as the library source
- kept `tests/` as the validation suite
- kept `docs/` plus MkDocs configuration
- removed stale documentation that implied tracked operational tooling still existed

## Library status

The exported package API remains:
- `TimeBase`
- `TimeBaseTrend`
- `AutoTimeBase`
- `AutoTimeBaseTrend`

## Why this change happened

The goal of this cleanup was to leave a repository centered on the reusable library and its documentation, without stale references to removed tooling.
