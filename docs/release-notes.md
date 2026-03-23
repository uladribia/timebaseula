---
description: Short release-style summary of the current library-focused TimeBaseUla repository.
---

# Release notes

## TL;DR
- kept the public API centered on `TimeBase` and `TimeBaseTrend`
- retained the library source, tests, and documentation
- simplified the implementation to favor readability over feature breadth
- removed stale references to removed tooling and auto wrappers

## Repository changes

- kept `timebaseula/` as the library source
- kept `tests/` as the validation suite
- kept `docs/` plus MkDocs configuration
- removed stale documentation and package surface that implied extra orchestration layers

## Library status

The exported package API is:
- `TimeBase`
- `TimeBaseTrend`

## Why this change happened

The goal of the cleanup and refactor was to keep the repository centered on a readable reusable library with a small explicit API.
