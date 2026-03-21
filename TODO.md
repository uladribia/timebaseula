# TODO (Plan Progress)

## Done
- Added synthetic series generator with trend/seasonality/noise, amplitude modulation, and amplitude growth.
- Added tests for synthetic series generator (determinism, validation, modulation behavior).
- Added plotting script (Typer + Rich + logging) to generate synthetic series images.
- Generated and documented easy/medium/hard synthetic series plots in `docs/usage.md`.
- Added DLinear MAE evaluation script for multivariate synthetic scenarios.
- Updated MAE thresholds based on DLinear baselines (+25% margin).
- Added DLinear, TimeBase, and TimeBaseTrend reference overlays with MAE and params in plot legends.
- Added MAE comparison check script for naive/DLinear/TimeBase/TimeBaseTrend.
- Updated docs dependencies to include matplotlib.
- Implemented TimeBase core and BaseWindows model with orthogonal loss support.
- Added TimeBase unit tests (shape, padding, orthogonal loss, deterministic behavior).
- Wired TimeBase into package exports.
- Added TimeBaseTrend (trend + TimeBase) with DLinear-style addition and moving average decomposition.
- Added TimeBaseTrend tests (shape, linear_trend, moving_avg_window validation, decomp presence).
- Added integration tests for NeuralForecast (marked as `@pytest.mark.integration`).
- Added predict_single_series utility for single series prediction from multivariate models.
- Updated README.md with comprehensive quickstart and API documentation.
- Updated docs/index.md and docs/usage.md with full usage guide.

## Implementation Complete
All planned features have been implemented:
- TimeBase model with segment-level forecasting and basis extraction
- TimeBaseTrend model with trend decomposition
- Multivariate training support via NeuralForecast
- Single series prediction utility
- Comprehensive tests and documentation
