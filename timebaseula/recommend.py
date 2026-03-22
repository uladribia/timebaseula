"""Dataset profiling and default recommendation helpers for TimeBase models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Lightweight dataset statistics for parameter recommendation."""

    n_series: int
    min_length: int
    median_length: int
    max_length: int
    train_length_estimate: int
    dominant_period: int
    seasonality_strength: float
    trend_strength: float
    scale_median: float
    short_history: bool
    long_history: bool


def infer_test_size(
    series_length: int, horizon: int, test_fraction: float = 0.2
) -> int:
    """Infer an approximate holdout size while respecting the horizon."""
    inferred = max(horizon, round(series_length * test_fraction))
    return min(inferred, series_length - 1)


def infer_season_length(freq: str) -> int:
    """Choose a standard seasonal length for a frequency string."""
    return 7 if freq.upper() == "D" else 12


def candidate_periods_for_frequency(freq: str, input_size: int) -> tuple[int, ...]:
    """Return feasible candidate periods for lightweight profiling."""
    raw_candidates = (7, 14, 28) if freq.upper() == "D" else (3, 6, 12)
    feasible = [
        candidate for candidate in raw_candidates if 2 <= candidate <= input_size
    ]
    fallback = max(2, min(infer_season_length(freq), input_size))
    return tuple(feasible or [fallback])


def estimate_lag_correlation(values: np.ndarray, lag: int) -> float:
    """Estimate lag correlation cheaply, returning zero when undefined."""
    if lag <= 0 or len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def profile_dataset(frame: pd.DataFrame, freq: str, horizon: int) -> DatasetProfile:
    """Build a lightweight dataset profile for model and training defaults."""
    lengths = frame.groupby("unique_id").size()
    sampled_ids = frame["unique_id"].drop_duplicates().head(64)
    sampled = frame[frame["unique_id"].isin(sampled_ids)]
    median_length = int(lengths.median())
    train_length_estimate = max(
        1, median_length - infer_test_size(median_length, horizon)
    )
    max_input_size = max(4, train_length_estimate - horizon)
    candidate_periods = candidate_periods_for_frequency(freq, max_input_size)

    seasonality_scores = {candidate: [] for candidate in candidate_periods}
    trend_scores: list[float] = []
    scales: list[float] = []

    for _, series in sampled.groupby("unique_id"):
        values = series.sort_values("ds")["y"].to_numpy(dtype=float)
        if len(values) < 3:
            continue
        scales.append(float(np.std(values)))
        for candidate in candidate_periods:
            seasonality_scores[candidate].append(
                max(0.0, estimate_lag_correlation(values, candidate))
            )

        x_axis = np.arange(len(values), dtype=float)
        if np.std(values) == 0:
            trend_scores.append(0.0)
        else:
            slope = float(np.polyfit(x_axis, values, 1)[0])
            trend_scores.append(min(1.0, abs(slope) * len(values) / np.std(values)))

    dominant_period = max(
        candidate_periods,
        key=lambda candidate: float(np.mean(seasonality_scores[candidate] or [0.0])),
    )
    seasonality_strength = float(np.mean(seasonality_scores[dominant_period] or [0.0]))
    trend_strength = float(np.mean(trend_scores or [0.0]))
    scale_median = float(np.median(scales or [0.0]))

    return DatasetProfile(
        n_series=int(lengths.shape[0]),
        min_length=int(lengths.min()),
        median_length=median_length,
        max_length=int(lengths.max()),
        train_length_estimate=train_length_estimate,
        dominant_period=int(dominant_period),
        seasonality_strength=seasonality_strength,
        trend_strength=trend_strength,
        scale_median=scale_median,
        short_history=train_length_estimate < 32,
        long_history=train_length_estimate >= 128,
    )


def recommend_training_kwargs(
    profile: DatasetProfile, horizon: int, max_steps: int
) -> dict[str, int | float]:
    """Recommend neural training defaults from the dataset profile."""
    recommended_max_steps = max_steps
    if profile.short_history:
        recommended_max_steps = min(max_steps, 100)
    elif profile.long_history:
        if horizon >= 24:
            recommended_max_steps = max(max_steps, 200)
        elif horizon >= 14:
            recommended_max_steps = max(max_steps, 150)
        elif horizon >= 12:
            recommended_max_steps = max(max_steps, 120)
        elif horizon >= 6:
            recommended_max_steps = max(max_steps, 80)

    learning_rate = 5e-3
    if profile.long_history and horizon >= 12:
        learning_rate = 3e-3
    early_stop_patience_steps = max(10, recommended_max_steps // 5)
    val_check_steps = max(10, recommended_max_steps // 4)
    return {
        "max_steps": int(recommended_max_steps),
        "learning_rate": learning_rate,
        "early_stop_patience_steps": int(early_stop_patience_steps),
        "val_check_steps": int(val_check_steps),
    }


def recommend_timebase_model_kwargs(
    profile: DatasetProfile, horizon: int
) -> dict[str, int]:
    """Recommend TimeBase architectural defaults from the dataset profile."""
    max_input_size = max(4, profile.train_length_estimate - horizon)
    input_size = min(max(horizon * 2, 16), 96, max_input_size)
    if profile.short_history:
        input_size = min(input_size, 24)
    period_len = min(profile.dominant_period, max(2, input_size))
    basis_num = (
        3 if profile.short_history else 8 if profile.seasonality_strength >= 0.4 else 6
    )
    return {
        "input_size": int(input_size),
        "period_len": int(max(2, period_len)),
        "basis_num": int(min(basis_num, max(2, input_size // 2))),
    }


def recommend_timebase_trend_model_kwargs(
    profile: DatasetProfile, horizon: int
) -> dict[str, int]:
    """Recommend TimeBaseTrend defaults from the dataset profile."""
    timebase_kwargs = recommend_timebase_model_kwargs(profile, horizon)
    input_size = int(timebase_kwargs["input_size"])
    period_len = int(timebase_kwargs["period_len"])
    moving_avg_window = min(max(5, 2 * period_len + 1), input_size - 1)
    if moving_avg_window % 2 == 0:
        moving_avg_window -= 1
    if profile.short_history and profile.trend_strength < 0.2:
        moving_avg_window = min(moving_avg_window, 5)
    return {**timebase_kwargs, "moving_avg_window": int(max(3, moving_avg_window))}


def recommend_timebase_kwargs(
    frame: pd.DataFrame,
    freq: str,
    horizon: int,
    max_steps: int,
) -> dict[str, Any]:
    """Profile a dataset and return recommended TimeBase kwargs."""
    profile = profile_dataset(frame, freq=freq, horizon=horizon)
    training_kwargs = recommend_training_kwargs(
        profile,
        horizon=horizon,
        max_steps=max_steps,
    )
    if profile.long_history and horizon >= 12:
        training_kwargs["learning_rate"] = min(
            float(training_kwargs["learning_rate"]),
            1e-3,
        )
    return {
        **recommend_timebase_model_kwargs(profile, horizon=horizon),
        **training_kwargs,
    }


def recommend_timebase_trend_kwargs(
    frame: pd.DataFrame,
    freq: str,
    horizon: int,
    max_steps: int,
) -> dict[str, Any]:
    """Profile a dataset and return recommended TimeBaseTrend kwargs."""
    profile = profile_dataset(frame, freq=freq, horizon=horizon)
    training_kwargs = recommend_training_kwargs(
        profile,
        horizon=horizon,
        max_steps=max_steps,
    )
    if profile.long_history and horizon >= 12:
        training_kwargs["learning_rate"] = min(
            float(training_kwargs["learning_rate"]),
            1e-3,
        )
    return {
        **recommend_timebase_trend_model_kwargs(profile, horizon=horizon),
        **training_kwargs,
    }
