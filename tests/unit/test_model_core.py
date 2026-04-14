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


def test_timebase_core_multivariate_forward_matches_channelwise_projection() -> None:
    """Multivariate inputs should match the shared per-channel projection path."""
    torch.manual_seed(0)
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=24,
            period_len=6,
            basis_num=4,
            use_period_norm=True,
        ),
        horizon=12,
    )
    multivariate_series = torch.randn(2, 24, 3)

    forecast, basis = core(multivariate_series)

    assert forecast.shape == (2, 12, 3)
    assert basis.shape == (2, 3, 6, 4)

    expected_forecasts = []
    expected_bases = []
    for channel_idx in range(multivariate_series.shape[-1]):
        channel_forecast, channel_basis = core(multivariate_series[:, :, channel_idx])
        expected_forecasts.append(channel_forecast.unsqueeze(-1))
        expected_bases.append(channel_basis.unsqueeze(1))

    assert torch.allclose(forecast, torch.cat(expected_forecasts, dim=-1))
    assert torch.allclose(basis, torch.cat(expected_bases, dim=1))
