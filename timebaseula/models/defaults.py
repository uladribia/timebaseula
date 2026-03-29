"""Default values and small helper utilities for TimeBase models."""

from __future__ import annotations

from typing import Any

from neuralforecast.losses.pytorch import MAE

DEFAULT_LOSS = MAE()
DEFAULT_BASIS_NUM = 6
DEFAULT_MOVING_AVG_WINDOW = 25


def _normalize_frequency(freq: str | None) -> str | None:
    """Normalize a frequency string when one is provided."""
    if freq is None:
        return None
    return freq.upper()


def _default_input_size(horizon: int) -> int:
    """Return the default input window size."""
    return max(2 * horizon, 8)


def _default_period_len(horizon: int, input_size: int, freq: str | None) -> int:
    """Return a simple seasonal default for the explicit models."""
    normalized_freq = _normalize_frequency(freq)
    if normalized_freq == "D":
        return min(7, input_size)
    if normalized_freq in {"M", "ME", "MS"}:
        return min(12, input_size)
    return min(max(2, horizon), input_size)


def _default_trainer_kwargs(trainer_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Force CPU-first trainer defaults while preserving user overrides."""
    resolved = dict(trainer_kwargs)
    resolved.setdefault("accelerator", "cpu")
    resolved.setdefault("devices", 1)
    return resolved


def resolve_moving_avg_window(moving_avg_window: int | None) -> int:
    """Return a valid moving average window for trend decomposition."""
    resolved_window = (
        DEFAULT_MOVING_AVG_WINDOW
        if moving_avg_window is None
        else int(moving_avg_window)
    )
    if resolved_window % 2 == 0:
        msg = "moving_avg_window must be odd for moving average decomposition"
        raise ValueError(msg)
    return resolved_window
