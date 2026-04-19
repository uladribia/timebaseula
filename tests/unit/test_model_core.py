"""Tests for the pure TimeBase core module."""

from __future__ import annotations

from hypothesis import given
import torch

from tests.property_strategies import (
    CoreCase,
    CoreMultivariateCase,
    core_multivariate_cases,
    core_univariate_cases,
)


@given(case=core_univariate_cases())
def test_timebase_core_forward_returns_forecast_and_basis(
    case: CoreCase,
) -> None:
    """The pure TimeBase core should keep the documented univariate tensor contract."""
    core = case.core
    series = case.series

    forecast, basis = core(series)

    assert forecast.shape == (series.shape[0], case.horizon)
    assert basis.shape == (series.shape[0], case.period_len, core.basis_num)


@given(case=core_multivariate_cases())
def test_timebase_core_multivariate_forward_matches_channelwise_projection(
    case: CoreMultivariateCase,
) -> None:
    """Multivariate inputs should match the shared per-channel projection path."""
    core = case.core
    multivariate_series = case.series

    forecast, basis = core(multivariate_series)

    assert forecast.shape == (
        multivariate_series.shape[0],
        case.horizon,
        multivariate_series.shape[-1],
    )
    assert basis.shape == (
        multivariate_series.shape[0],
        multivariate_series.shape[-1],
        case.period_len,
        case.basis_num,
    )

    expected_forecasts = []
    expected_bases = []
    for channel_idx in range(multivariate_series.shape[-1]):
        channel_forecast, channel_basis = core(multivariate_series[:, :, channel_idx])
        expected_forecasts.append(channel_forecast.unsqueeze(-1))
        expected_bases.append(channel_basis.unsqueeze(1))

    assert torch.allclose(
        forecast,
        torch.cat(expected_forecasts, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        basis,
        torch.cat(expected_bases, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
