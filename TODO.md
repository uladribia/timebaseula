# TODO (Plan Progress)

## Done
- Added synthetic series generator with trend/seasonality/noise, amplitude modulation, and amplitude growth.
- Added tests for synthetic series generator (determinism, validation, modulation behavior).
- Added plotting script (Typer + Rich + logging) to generate synthetic series images.
- Generated and documented easy/medium/hard synthetic series plots in `docs/usage.md`.
- Added DLinear MAE evaluation script for multivariate synthetic scenarios.
- Updated MAE thresholds based on DLinear baselines (+25% margin).
- Added DLinear and TimeBase reference overlays with MAE in plot legends.
- Added MAE comparison check script for naive/DLinear/TimeBase.
- Updated docs dependencies to include matplotlib.
- Implemented TimeBase core and BaseWindows model with orthogonal loss support.
- Added TimeBase unit tests (shape, padding, orthogonal loss).
- Wired TimeBase into package exports.
- Added TimeBaseTrend (trend + TimeBase) with learnable trend_weight and moving average decomposition.
- Added TimeBaseTrend tests (shape, trend_weight, moving_avg_window validation).

## Next
- Add multivariate TimeBase variants and single-series inference helper.
- Add MAE threshold tests for easy/medium/hard multivariate scenarios.
- Review TimeBase/TimeBaseTrend hard-case MAE - note TimeBaseTrend improves hard case (0.80 vs 1.54) but still underperforms DLinear.
- Write model tests (shape, orthogonal loss, MAE thresholds for easy/medium/hard synthetic cases).
- Update README/docs with usage examples.
- Run `make format`, `make lint`, `make test` and commit.
