"""Tests for the pure TimeBase core module."""

from __future__ import annotations

from typing import Any, cast

from hypothesis import given
from hypothesis import strategies as st
import torch

from timebaseula.models.core import TimeBaseConfig, TimeBaseCore


FINITE_FLOATS = st.floats(
    min_value=-100,
    max_value=100,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@st.composite
def _core_univariate_cases(
    draw: st.DrawFn,
) -> tuple[TimeBaseCore, torch.Tensor, int, int]:
    input_size = draw(st.integers(min_value=1, max_value=24))
    period_len = draw(st.integers(min_value=1, max_value=min(input_size, 8)))
    basis_num = draw(st.integers(min_value=1, max_value=6))
    horizon = draw(st.integers(min_value=1, max_value=12))
    batch_size = draw(st.integers(min_value=1, max_value=4))
    values = draw(
        st.lists(
            st.lists(FINITE_FLOATS, min_size=input_size, max_size=input_size),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            use_period_norm=True,
        ),
        horizon=horizon,
    )
    return core, torch.tensor(values, dtype=torch.float32), horizon, period_len


@st.composite
def _core_multivariate_cases(
    draw: st.DrawFn,
) -> tuple[TimeBaseCore, torch.Tensor, int, int, int]:
    input_size = draw(st.integers(min_value=1, max_value=18))
    period_len = draw(st.integers(min_value=1, max_value=min(input_size, 6)))
    basis_num = draw(st.integers(min_value=1, max_value=5))
    horizon = draw(st.integers(min_value=1, max_value=10))
    batch_size = draw(st.integers(min_value=1, max_value=3))
    n_series = draw(st.integers(min_value=1, max_value=4))
    values = draw(
        st.lists(
            st.lists(
                st.lists(FINITE_FLOATS, min_size=n_series, max_size=n_series),
                min_size=input_size,
                max_size=input_size,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    torch.manual_seed(0)
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            use_period_norm=draw(st.booleans()),
        ),
        horizon=horizon,
    )
    return (
        core,
        torch.tensor(values, dtype=torch.float32),
        horizon,
        period_len,
        basis_num,
    )


def _core_univariate_case_strategy() -> st.SearchStrategy[
    tuple[TimeBaseCore, torch.Tensor, int, int]
]:
    return cast(Any, _core_univariate_cases)()


def _core_multivariate_case_strategy() -> st.SearchStrategy[
    tuple[TimeBaseCore, torch.Tensor, int, int, int]
]:
    return cast(Any, _core_multivariate_cases)()


@given(case=_core_univariate_case_strategy())
def test_timebase_core_forward_returns_forecast_and_basis(
    case: tuple[TimeBaseCore, torch.Tensor, int, int],
) -> None:
    """The pure TimeBase core should keep the documented univariate tensor contract."""
    core, series, horizon, period_len = case

    forecast, basis = core(series)

    assert forecast.shape == (series.shape[0], horizon)
    assert basis.shape == (series.shape[0], period_len, core.basis_num)


@given(case=_core_multivariate_case_strategy())
def test_timebase_core_multivariate_forward_matches_channelwise_projection(
    case: tuple[TimeBaseCore, torch.Tensor, int, int, int],
) -> None:
    """Multivariate inputs should match the shared per-channel projection path."""
    core, multivariate_series, horizon, period_len, basis_num = case

    forecast, basis = core(multivariate_series)

    assert forecast.shape == (
        multivariate_series.shape[0],
        horizon,
        multivariate_series.shape[-1],
    )
    assert basis.shape == (
        multivariate_series.shape[0],
        multivariate_series.shape[-1],
        period_len,
        basis_num,
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
