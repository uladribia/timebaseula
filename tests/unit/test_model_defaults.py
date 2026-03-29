"""Tests for TimeBase default-resolution helpers."""

from __future__ import annotations

from timebaseula.models.defaults import (
    _default_input_size,
    _default_period_len,
    _default_trainer_kwargs,
)


def test_default_input_size_is_horizon_based_with_floor() -> None:
    """Default input size should follow the documented horizon rule."""
    assert _default_input_size(2) == 8
    assert _default_input_size(12) == 24


def test_default_period_len_uses_frequency_specific_defaults() -> None:
    """Default period length should prefer daily and monthly seasonality."""
    assert _default_period_len(horizon=14, input_size=28, freq="D") == 7
    assert _default_period_len(horizon=6, input_size=12, freq="ME") == 12
    assert _default_period_len(horizon=5, input_size=9, freq=None) == 5


def test_default_trainer_kwargs_force_cpu_defaults_without_overrides() -> None:
    """Trainer defaults should remain CPU-first while preserving caller settings."""
    assert _default_trainer_kwargs({}) == {"accelerator": "cpu", "devices": 1}
    assert _default_trainer_kwargs({"devices": 2}) == {
        "accelerator": "cpu",
        "devices": 2,
    }
