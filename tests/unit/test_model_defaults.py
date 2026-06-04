"""Tests for TimeBase default-resolution helpers."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from timebaseula.models.defaults import (
    _default_input_size,
    _default_period_len,
    _default_trainer_kwargs,
)


@given(horizon=st.integers(min_value=1, max_value=128))
def test_default_input_size_is_horizon_based_with_floor(horizon: int) -> None:
    """Default input size should follow the documented horizon rule."""
    resolved = _default_input_size(horizon)

    assert resolved == max(2 * horizon, 8)
    assert resolved >= 8


@given(
    horizon=st.integers(min_value=1, max_value=32),
    input_size=st.integers(min_value=1, max_value=32),
)
def test_default_period_len_uses_frequency_specific_defaults(
    horizon: int,
    input_size: int,
) -> None:
    """Default period length should respect the documented frequency-specific rules."""
    daily_period = _default_period_len(horizon=horizon, input_size=input_size, freq="D")
    monthly_period = _default_period_len(
        horizon=horizon,
        input_size=input_size,
        freq="ME",
    )
    fallback_period = _default_period_len(
        horizon=horizon,
        input_size=input_size,
        freq=None,
    )

    assert daily_period == min(7, input_size)
    assert monthly_period == min(12, input_size)
    assert fallback_period == min(max(2, horizon), input_size)


@given(
    devices=st.integers(min_value=1, max_value=8),
    accelerator=st.sampled_from(["cpu", "gpu", "tpu"]),
)
def test_default_trainer_kwargs_force_cpu_defaults_without_overrides(
    devices: int,
    accelerator: str,
) -> None:
    """Trainer defaults should remain CPU-first while preserving caller settings."""
    assert _default_trainer_kwargs({}) == {"accelerator": "cpu", "devices": 1}
    assert _default_trainer_kwargs({"devices": devices}) == {
        "accelerator": "cpu",
        "devices": devices,
    }
    assert _default_trainer_kwargs({"accelerator": accelerator}) == {
        "accelerator": accelerator,
        "devices": 1,
    }
