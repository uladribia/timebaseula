"""Synthetic time series helpers used by tests, scripts, and docs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SyntheticSeriesError(ValueError):
    """Base error for synthetic series generation."""


class InvalidLengthError(SyntheticSeriesError):
    """Raised when the synthetic series length is invalid."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("length must be positive")


class InvalidNoiseError(SyntheticSeriesError):
    """Raised when the noise standard deviation is invalid."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("noise_std must be non-negative")


class InvalidSeasonPeriodError(SyntheticSeriesError):
    """Raised when the seasonality period is invalid."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("season_period must be positive")


class InvalidAmplitudePeriodError(SyntheticSeriesError):
    """Raised when the amplitude modulation period is invalid."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("amplitude_period must be positive")


class InvalidAmplitudeStrengthError(SyntheticSeriesError):
    """Raised when the amplitude modulation strength is invalid."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("amplitude_strength must be non-negative")


class InvalidAmplitudeGrowthRateError(SyntheticSeriesError):
    """Raised when the amplitude growth rate is invalid."""

    def __init__(self) -> None:
        """Initialize the error message."""
        super().__init__("amplitude_growth_rate must be non-negative")


@dataclass(frozen=True)
class SyntheticSeriesConfig:
    """Configuration for synthetic series generation."""

    length: int
    noise_std: float
    include_trend: bool = True
    include_seasonality: bool = True
    season_period: int = 24
    amplitude_period: int | None = None
    amplitude_strength: float = 0.0
    amplitude_growth_rate: float = 0.0
    seed: int = 0
    unique_id: str = "series_1"


def make_synthetic_series(
    length: int,
    noise_std: float,
    include_trend: bool = True,
    include_seasonality: bool = True,
    season_period: int = 24,
    amplitude_period: int | None = None,
    amplitude_strength: float = 0.0,
    amplitude_growth_rate: float = 0.0,
    seed: int = 0,
    unique_id: str = "series_1",
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate a deterministic synthetic time series for testing.

    Args:
        length: Number of observations to generate.
        noise_std: Standard deviation of the Gaussian noise.
        include_trend: Whether to add a linear trend component.
        include_seasonality: Whether to add a sinusoidal seasonal component.
        season_period: Period, in steps, for the seasonal component.
        amplitude_period: Optional period for amplitude modulation.
        amplitude_strength: Non-negative amplitude modulation strength.
        amplitude_growth_rate: Non-negative linear growth rate for amplitude.
        seed: Seed for deterministic noise generation.
        unique_id: Identifier for the series.
        start: Optional start timestamp for the time index.

    Returns:
        DataFrame with columns ``unique_id``, ``ds``, and ``y``.
    """
    if length <= 0:
        raise InvalidLengthError
    if noise_std < 0:
        raise InvalidNoiseError
    if season_period <= 0:
        raise InvalidSeasonPeriodError
    if amplitude_period is not None and amplitude_period <= 0:
        raise InvalidAmplitudePeriodError
    if amplitude_strength < 0:
        raise InvalidAmplitudeStrengthError
    if amplitude_growth_rate < 0:
        raise InvalidAmplitudeGrowthRateError

    rng = np.random.default_rng(seed)
    time_idx = np.arange(length)
    values = np.zeros(length, dtype=float)

    if include_trend:
        values += 0.01 * time_idx

    if include_seasonality:
        seasonal = np.sin(2 * np.pi * time_idx / season_period)
        modulation = 1.0
        if amplitude_period is not None and amplitude_strength > 0:
            modulation += amplitude_strength * np.sin(
                2 * np.pi * time_idx / amplitude_period
            )
        if amplitude_growth_rate > 0:
            growth = (
                1.0
                if length == 1
                else 1.0 + amplitude_growth_rate * (time_idx / (length - 1))
            )
            modulation *= growth
        values += seasonal * modulation

    if noise_std > 0:
        values += rng.normal(0.0, noise_std, size=length)

    timestamps = pd.date_range(
        start=pd.Timestamp("2020-01-01") if start is None else start,
        periods=length,
        freq="D",
    )
    return pd.DataFrame({"unique_id": unique_id, "ds": timestamps, "y": values})
