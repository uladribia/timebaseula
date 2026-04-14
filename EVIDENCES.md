---
description: Evidence log for the BaseMultivariate TimeBase feature branch.
---

# Evidence: BaseMultivariate TimeBase branch

## TL;DR
- Branch: `feature/base-multivariate-timebase`
- Source branch: `benchmark`
- Goal: keep the public `TimeBase` / `TimeBaseTrend` API unchanged while switching internal training to joint multivariate windows closer to original TimeBase `individual=0` behavior.
- Result: implemented, unit tests pass, integration tests pass, docs updated, AirPassengers benchmark reproduced before/after, stronger normal-profile scope benchmarks were reproduced, and strict reruns were executed on the published daily benchmark settings.
- Main benchmark takeaway:
  - `TimeBase` improved on AirPassengers (`MAE 17.0719 -> 16.8449`).
  - `TimeBaseTrend` regressed on AirPassengers (`MAE 17.2453 -> 21.7723`).
  - On strict published-setting reruns, the TimeBase family remains strong:
    - all-scope heavy-256: `TimeBaseTrend` becomes the best overall model and improves `avg_mae` from `70.4985` to `66.1189`
    - detailed heavy-256: `TimeBaseTrend` stays the best overall model and improves `avg_mae` from `4.5542` to `4.5316`
    - aggregated published rerun: `AutoTheta` stays best overall, while `TimeBase`, `TimeBaseTrend`, and `AutoTimeBaseTrend` improve their published `avg_mae`
  - The earlier normal-64 all-scope and aggregated runs selected the same aggregate `unique_id` set, which explains why those two intermediate reruns were numerically identical.

## Requested workflow checklist
- [x] Created a dedicated feature branch from `benchmark`
- [x] Ran a baseline subset before code changes
- [x] Added tests first and confirmed a red run
- [x] Implemented the multivariate wrapper
- [x] Re-ran the same subset after the change
- [x] Ran `make format`
- [x] Ran `make lint`
- [x] Ran `make test`
- [x] Ran `make test-integration` (justified by NeuralForecast wrapper changes)
- [x] Rebuilt docs with `mkdocs build --strict`
- [x] Reproduced benchmarks and recorded results
- [x] Documented evidence in this file

## Implementation scope
Target equivalence is the original TimeBase shared-weight path:
- equivalent intent: original TimeBase with `individual=0`
- not implemented: original `individual=1` per-channel heads
- public constructors kept unchanged for `TimeBase`, `TimeBaseTrend`, `AutoTimeBase`, and `AutoTimeBaseTrend`

### Files changed
- `timebaseula/models/base.py`
- `timebaseula/models/core.py`
- `timebaseula/models/decomposition.py`
- `timebaseula/models/timebase.py`
- `tests/unit/test_model_core.py`
- `tests/unit/test_model_decomposition.py`
- `tests/unit/test_timebase.py`
- `README.md`
- `docs/usage.md`
- `docs/models.md`

### What changed
1. `TimeBase` / `TimeBaseTrend` now use NeuralForecast multivariate sampling internally.
2. `TimeBaseCore` now accepts:
   - univariate input: `[batch, time]`
   - multivariate input: `[batch, time, n_series]`
3. Multivariate forward follows the original shared-weight pattern:
   - flatten series into batch
   - apply one shared `ts2basis`
   - apply one shared `basis2ts`
   - restore `[batch, horizon, n_series]`
4. `SeriesDecomposition` now supports multivariate tensors.
5. Prediction logic was adapted so long-format `NeuralForecast.fit/predict` still works, including subset prediction and probabilistic outputs.

## Baseline subset before any code changes
Command log:
- `logs/baseline_subset_tests.log`

Command:
```bash
uv run --frozen pytest tests/unit/test_timebase.py tests/integration/test_neuralforecast_fit_predict.py
```

Result:
- `24 passed, 6 skipped`
- the integration tests were skipped in this baseline command because the default test config does not run `integration` marks without `--run-integration`

## TDD red -> green evidence
### Red
Command log:
- `logs/red_multivariate_unit_tests.log`

Command:
```bash
uv run --frozen pytest tests/unit/test_model_core.py tests/unit/test_model_decomposition.py tests/unit/test_timebase.py
```

Result:
- failed as expected
- failing areas:
  - `TimeBaseCore` rejected 3D input
  - `SeriesDecomposition` rejected 3D input
  - `TimeBase` / `TimeBaseTrend` were still `SAMPLING_TYPE="windows"`

### Green
Command log:
- `logs/green_multivariate_unit_tests.log`

Result:
- `33 passed`

## Same subset after the change
Command log:
- `logs/postchange_subset_tests.log`

Result:
- `28 passed, 6 skipped`
- same command path preserved
- extra passing unit tests come from the newly added multivariate coverage

## Quality gates after implementation
### Format
- log: `logs/main_make_format.log`
- result: passed

### Lint
- log: `logs/main_make_lint.log`
- result: passed

### Unit suite
- log: `logs/main_make_test.log`
- result: `75 passed, 6 deselected`

### Integration suite
- log: `logs/main_make_test_integration.log`
- result: `6 passed, 75 deselected`
- rationale: this change directly affects NeuralForecast batching, fit/predict behavior, subset prediction, conformal intervals, and distribution-loss prediction

### Docs build
- log: `logs/main_mkdocs_build.log`
- result: passed

## Benchmark evidence
### 1) AirPassengers benchmark before vs after
Before log:
- `logs/benchmark_airpassengers_before_stdout.log`

After log:
- `logs/benchmark_airpassengers_after_stdout.log`

Comparison:

| Model | MAE before | MAE after | Delta | Runtime before | Runtime after | Runtime delta |
|---|---:|---:|---:|---:|---:|---:|
| TimeBase | 17.0719 | 16.8449 | -0.2270 | 0.2227 | 0.1176 | -0.1051 |
| TimeBaseTrend | 17.2453 | 21.7723 | +4.5270 | 0.5731 | 0.3348 | -0.2383 |
| NLinear | 12.3282 | 12.3282 | +0.0000 | 0.4380 | 0.2824 | -0.1556 |
| DLinear | 18.5919 | 18.5919 | +0.0000 | 0.4155 | 0.2716 | -0.1439 |
| AutoMFLES | 18.3310 | 18.3310 | +0.0000 | 4.6884 | 4.0880 | -0.6004 |
| Naive | 76.0000 | 76.0000 | +0.0000 | 0.0015 | 0.0015 | +0.0000 |

Interpretation:
- `TimeBase` improved slightly and ran faster.
- `TimeBaseTrend` got materially worse on this tiny benchmark, despite running faster.
- Baselines stayed numerically identical, which is a good sanity check that the benchmark harness itself was not disturbed.

### 2) Daily aggregated smoke benchmark after the change
Preparation log:
- `logs/prepare_nixtla_panel_evidence_stdout.log`

Benchmark log:
- `logs/benchmark_nixtla_panel_smoke_after_stdout.log`

Configuration used:
- prepared panel from `/home/newuser/Repositories/ula/timebaseula/data/danone_subset_250_pdvs_all_history.parquet.gzip`
- `profile=smoke`
- `max-series=8`
- `series-scope=aggregated`
- `horizon=28`

Shared-model comparison against the published benchmark-branch smoke report in `docs/daily-panel-aggregated-benchmark-smoke.md`:

| Model | Published avg_mae | Current avg_mae | Delta |
|---|---:|---:|---:|
| AutoTheta | 609.0243 | 609.0243 | +0.0000 |
| TimeBase | 749.4134 | 747.1265 | -2.2869 |
| TimeBaseTrend | 806.8331 | 799.9444 | -6.8887 |
| AutoMFLES | 1400.1926 | 1400.1926 | +0.0000 |
| DLinear | 1027.9420 | 1027.9420 | +0.0000 |
| NLinear | 1227.5166 | 1227.5166 | +0.0000 |
| Naive | 1928.9120 | 1928.9120 | +0.0000 |

Interpretation:
- on this smoke benchmark, shared-model `avg_mae` for `TimeBase` and `TimeBaseTrend` improved slightly versus the published benchmark-branch report
- rank values are not directly comparable here because this reproduction did not include the auto TimeBase/NLinear/DLinear variants present in the published page

### 3) Stronger normal-profile scope benchmarks
Preparation log for the stronger runs:
- `logs/prepare_nixtla_panel_normal_scopes_stdout.log`

Shared panel output used by all three runs:
- `logs/evidence_danone_panel/panel.parquet`

Scope benchmark logs:
- all scope: `logs/benchmark_nixtla_panel_all_normal_stdout.log`
- aggregated scope: `logs/benchmark_nixtla_panel_aggregated_normal_stdout.log`
- detailed scope: `logs/benchmark_nixtla_panel_detailed_normal_stdout.log`

Common benchmark settings:
- `profile=normal`
- `horizon=28`
- `test_ratio=0.2`
- `max-series=64`
- detailed run used `--no-include-autotheta` to stay aligned with the repository's detailed benchmark convention

Current winners:

| Scope | Winner | Winner avg_rank | Winner avg_mae |
|---|---|---:|---:|
| all | AutoTheta | 2.2995 | 149.9750 |
| aggregated | AutoTheta | 2.2995 | 149.9750 |
| detailed | TimeBaseTrend | 2.3724 | 5.7126 |

### Why the all-scope and aggregated metrics match
A direct selection check showed that the all-scope normal-64 run chose exactly the same `64` `unique_id` values as the aggregated normal-64 run.
That evidence is stored in:
- `logs/normal_scope_selection_compare.log`

Key result:
- `same_ids: True`

Interpretation:
- under this normalized 64-series budget, the highest-coverage, highest-volume candidates in the mixed scope are all aggregate series
- therefore the all-scope benchmark collapses to the aggregated benchmark at this specific budget

### 4) Strict reruns on the published benchmark settings
This section is the final apples-to-apples check against the published benchmark pages.

Shared prepared panel used by the strict reruns:
- `logs/evidence_danone_panel/panel.parquet`

Exhaustive metric deltas for all common models and all reported metrics are stored in:
- `logs/strict_published_comparison.log`

#### 4.1 All-scope published rerun
Reference page:
- `docs/daily-panel-benchmark.md`

Rerun log:
- `logs/benchmark_nixtla_panel_all_heavy256_exact_stdout.log`

Exact rerun settings:
```bash
uv run --frozen python scripts/benchmark_nixtla_panel.py run \
  --input-path logs/evidence_danone_panel/panel.parquet \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile heavy \
  --max-series 256 \
  --no-include-autotheta
```

Published winner vs current winner:
- published winner: `TimeBase` (`avg_rank=2.0625`)
- current winner: `TimeBaseTrend` (`avg_rank=2.1940`)

Headline deltas:

| Model | Published avg_mae | Current avg_mae | Delta | Published avg_rank | Current avg_rank | Delta |
|---|---:|---:|---:|---:|---:|---:|
| TimeBase | 67.9572 | 67.7587 | -0.1985 | 2.0625 | 2.3203 | +0.2578 |
| TimeBaseTrend | 70.4985 | 66.1189 | -4.3796 | 2.7884 | 2.1940 | -0.5944 |
| AutoMFLES | 99.9432 | 99.9432 | +0.0000 | 3.1133 | 3.2396 | +0.1263 |
| NLinear | 74.9211 | 74.9211 | +0.0000 | 3.2533 | 3.3919 | +0.1386 |
| DLinear | 84.7788 | 84.7788 | +0.0000 | 4.2995 | 4.3353 | +0.0358 |
| Naive | 152.6325 | 152.6325 | +0.0000 | 5.4831 | 5.5189 | +0.0358 |

Interpretation:
- this rerun reproduces the published accuracy metrics exactly for `AutoMFLES`, `NLinear`, `DLinear`, and `Naive`
- `TimeBase` improves slightly on `avg_mae`, but loses ground on `avg_rank` and `wins`
- `TimeBaseTrend` improves substantially and becomes the best overall all-scope model in the rerun
- because the non-TimeBase baselines reproduce exactly, the changed behavior is strongly localized to the TimeBase family rather than the benchmark harness

#### 4.2 Aggregated published rerun
Reference page:
- `docs/daily-panel-aggregated-benchmark.md`

Rerun log:
- `logs/benchmark_nixtla_panel_aggregated_published_exact_stdout.log`

Exact rerun settings:
```bash
uv run --frozen python scripts/benchmark_nixtla_panel.py run \
  --input-path logs/evidence_danone_panel/panel.parquet \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile normal \
  --series-scope aggregated \
  --max-series 64 \
  --tuned-config-path artifacts/tuning/aggregated/best_configs.json
```

Published winner vs current winner:
- published winner: `AutoTheta` (`avg_rank=3.5885`)
- current winner: `AutoTheta` (`avg_rank=3.7240`)

Headline deltas for the TimeBase family and tuned auto variants:

| Model | Published avg_mae | Current avg_mae | Delta | Published avg_rank | Current avg_rank | Delta |
|---|---:|---:|---:|---:|---:|---:|
| AutoTheta | 149.9750 | 149.9750 | +0.0000 | 3.5885 | 3.7240 | +0.1355 |
| AutoTimeBaseTrend | 155.3247 | 153.6379 | -1.6868 | 4.4089 | 3.8438 | -0.5651 |
| AutoNLinear | 167.8676 | 167.8676 | +0.0000 | 4.4010 | 4.5964 | +0.1954 |
| TimeBase | 169.0332 | 166.6526 | -2.3806 | 4.7266 | 4.8099 | +0.0833 |
| AutoTimeBase | 169.2781 | 168.5292 | -0.7489 | 4.9036 | 5.1771 | +0.2735 |
| TimeBaseTrend | 177.8933 | 167.0068 | -10.8865 | 5.5729 | 4.8438 | -0.7291 |
| AutoMFLES | 268.0997 | 268.0997 | +0.0000 | 5.9141 | 6.0729 | +0.1588 |
| AutoDLinear | 211.8093 | 211.8093 | +0.0000 | 6.5208 | 6.7031 | +0.1823 |
| NLinear | 193.3910 | 193.3910 | +0.0000 | 7.4141 | 7.5651 | +0.1510 |
| DLinear | 204.2335 | 204.2335 | +0.0000 | 8.1406 | 8.2292 | +0.0886 |
| Naive | 398.1608 | 398.1608 | +0.0000 | 10.4089 | 10.4349 | +0.0260 |

Interpretation:
- this rerun reproduces the published `avg_mae` values exactly for `AutoTheta`, `AutoNLinear`, `AutoMFLES`, `AutoDLinear`, `NLinear`, `DLinear`, and `Naive`
- `AutoTimeBaseTrend`, `TimeBase`, `AutoTimeBase`, and especially `TimeBaseTrend` improve their published `avg_mae`
- the overall winner does not change: `AutoTheta` remains first on average rank
- again, the fact that the non-TimeBase models reproduce exactly makes the benchmark comparison very strong and isolates the effect to the TimeBase family

#### 4.3 Detailed published rerun
Reference page:
- `docs/daily-panel-detailed-benchmark.md`

Rerun log:
- `logs/benchmark_nixtla_panel_detailed_heavy256_exact_stdout.log`

Exact rerun settings:
```bash
uv run --frozen python scripts/benchmark_nixtla_panel.py run \
  --input-path logs/evidence_danone_panel/panel.parquet \
  --horizon 28 \
  --test-ratio 0.2 \
  --profile heavy \
  --series-scope detailed \
  --no-include-autotheta \
  --max-series 256
```

Published winner vs current winner:
- published winner: `TimeBaseTrend` (`avg_rank=2.5944`)
- current winner: `TimeBaseTrend` (`avg_rank=2.5814`)

Headline deltas:

| Model | Published avg_mae | Current avg_mae | Delta | Published avg_rank | Current avg_rank | Delta |
|---|---:|---:|---:|---:|---:|---:|
| TimeBaseTrend | 4.5542 | 4.5316 | -0.0226 | 2.5944 | 2.5814 | -0.0130 |
| DLinear | 4.5996 | 4.5996 | +0.0000 | 2.9434 | 2.9460 | +0.0026 |
| TimeBase | 4.7074 | 4.7008 | -0.0066 | 3.0286 | 3.0599 | +0.0313 |
| NLinear | 4.6677 | 4.6677 | +0.0000 | 3.1556 | 3.1419 | -0.0137 |
| AutoMFLES | 5.1535 | 5.1535 | +0.0000 | 3.8444 | 3.8340 | -0.0104 |
| Naive | 6.9157 | 6.9157 | +0.0000 | 5.4336 | 5.4368 | +0.0032 |

Interpretation:
- this rerun reproduces the published `avg_mae` values exactly for `DLinear`, `NLinear`, `AutoMFLES`, and `Naive`
- `TimeBaseTrend` improves slightly and remains the best detailed-scope model
- `TimeBase` improves slightly on `avg_mae` but remains behind `DLinear` and `TimeBaseTrend` on the published headline ordering
- the detailed rerun is therefore the most stable strict-check result: the winner is preserved and the TimeBase family moves only modestly

### Strict published-setting conclusion
Across the strict reruns:
- `TimeBaseTrend` is the main model whose behavior changes materially
- the benchmark harness itself looks stable because most non-TimeBase baselines reproduce exactly on their published `avg_mae`
- the change appears beneficial on the daily panel benchmarks:
  - all-scope: `TimeBaseTrend` becomes best overall
  - detailed: `TimeBaseTrend` stays best overall and improves slightly
  - aggregated: `AutoTheta` stays best, but the TimeBase family improves
- the main unresolved negative signal remains the small AirPassengers benchmark, where `TimeBaseTrend` still regresses

## Public API compatibility evidence
- benchmark scripts still instantiate `TimeBase(...)` and `TimeBaseTrend(...)` without adding new required arguments
- existing integration coverage passed for:
  - `NeuralForecast.fit` / `predict`
  - multi-series training with subset prediction
  - conformal intervals
  - distribution-loss fit/predict
- docs were updated to explain the new internal multivariate batching without changing the external long-format API

## Notes and caveats
- I intentionally targeted the original TimeBase shared-weight path (`individual=0`). I did **not** add a new public `individual` parameter.
- `scripts/check_forecast_mae.py`, referenced by `AGENTS.md`, is not present on this benchmark branch checkout, so that exact script could not be run.
- The strongest negative signal is the AirPassengers `TimeBaseTrend` regression. That should be part of the merge decision.

## Decision support summary
Merge looks technically safe if the priority is:
- matching original TimeBase-style multivariate batching more closely
- keeping the public API unchanged
- preserving NeuralForecast compatibility

Merge is riskier if the priority is:
- preserving the current AirPassengers `TimeBaseTrend` behavior exactly
- minimizing any benchmark movement on very small datasets
