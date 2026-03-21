"""Utility functions for TimeBaseUla models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from timebaseula.models.timebase import TimeBase, TimeBaseTrend


def predict_single_series(
    model: TimeBase | TimeBaseTrend,
    series: pd.DataFrame,
    h: int,
    input_size: int,
    freq: str = "D",
) -> pd.DataFrame:
    """Predict a single series using a trained model.

    This utility function allows prediction on a single time series,
    even if the model was trained on multivariate data.

    Args:
        model: A trained TimeBase or TimeBaseTrend model.
        series: DataFrame with 'ds' (datestamps) and 'y' (values) columns.
        h: Forecast horizon.
        input_size: Input window size used during training.
        freq: Frequency string (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        DataFrame with 'ds', 'unique_id', and prediction columns.

    Example:
        >>> from neuralforecast import NeuralForecast
        >>> from timebaseula import TimeBase
        >>> # Train on multivariate data
        >>> nf = NeuralForecast(models=[TimeBase(h=24, input_size=48, ...)])
        >>> nf.fit(train_df)
        >>> # Predict single series
        >>> single_df = train_df[train_df['unique_id'] == 'series1']
        >>> pred = predict_single_series(nf.models[0], single_df, h=24, input_size=48)
    """
    from neuralforecast import NeuralForecast

    if "unique_id" not in series.columns:
        series = series.copy()
        series["unique_id"] = "series_0"

    nf = NeuralForecast(models=[model], freq=freq)
    forecast = nf.predict()

    unique_id = series["unique_id"].iloc[0]
    if unique_id in forecast.columns:
        result = forecast[forecast["unique_id"] == unique_id].copy()
    else:
        result = forecast.head(h).copy()
        result["unique_id"] = unique_id

    return result
