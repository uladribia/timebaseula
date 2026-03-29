"""Tests for the pure TimeBase core module."""

from __future__ import annotations

import torch

from timebaseula.models.core import TimeBaseConfig, TimeBaseCore


def test_timebase_core_forward_returns_forecast_and_basis() -> None:
    """The pure TimeBase core should keep the documented tensor contract."""
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=24,
            period_len=6,
            basis_num=4,
            use_period_norm=True,
        ),
        horizon=12,
    )

    forecast, basis = core(torch.ones((2, 24)))

    assert forecast.shape == (2, 12)
    assert basis.shape == (2, 6, 4)
