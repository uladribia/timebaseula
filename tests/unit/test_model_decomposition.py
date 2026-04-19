"""Tests for the pure Torch decomposition helpers."""

from __future__ import annotations

from typing import Any, cast

from hypothesis import given
from hypothesis import strategies as st
import torch

from timebaseula.models.decomposition import MovingAverage, SeriesDecomposition

FINITE_FLOATS = st.floats(
    min_value=-100,
    max_value=100,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@st.composite
def _tensor_2d(draw: st.DrawFn) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=1, max_value=4))
    time_steps = draw(st.integers(min_value=1, max_value=12))
    values = draw(
        st.lists(
            st.lists(FINITE_FLOATS, min_size=time_steps, max_size=time_steps),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


@st.composite
def _tensor_3d(draw: st.DrawFn) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=1, max_value=3))
    time_steps = draw(st.integers(min_value=1, max_value=10))
    n_series = draw(st.integers(min_value=1, max_value=4))
    values = draw(
        st.lists(
            st.lists(
                st.lists(FINITE_FLOATS, min_size=n_series, max_size=n_series),
                min_size=time_steps,
                max_size=time_steps,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


def _tensor_2d_strategy() -> st.SearchStrategy[torch.Tensor]:
    return cast(Any, _tensor_2d)()


def _tensor_3d_strategy() -> st.SearchStrategy[torch.Tensor]:
    return cast(Any, _tensor_3d)()


@given(
    kernel_size=st.integers(min_value=1, max_value=9).map(lambda value: 2 * value - 1),
    series=_tensor_2d_strategy(),
)
def test_moving_average_preserves_expected_shape(
    kernel_size: int,
    series: torch.Tensor,
) -> None:
    """The moving-average block should preserve univariate batch and time shape."""
    moving_average = MovingAverage(kernel_size=kernel_size)

    trend = moving_average(series)

    assert trend.shape == series.shape


@given(
    kernel_size=st.integers(min_value=1, max_value=9).map(lambda value: 2 * value - 1),
    series=_tensor_2d_strategy(),
)
def test_series_decomposition_reconstructs_original_signal(
    kernel_size: int,
    series: torch.Tensor,
) -> None:
    """Seasonal and trend components should reconstruct the original univariate input."""
    decomposition = SeriesDecomposition(kernel_size=kernel_size)

    seasonal, trend = decomposition(series)

    assert seasonal.shape == series.shape
    assert trend.shape == series.shape
    assert torch.allclose(seasonal + trend, series, atol=1e-4, rtol=1e-4)


@given(
    kernel_size=st.integers(min_value=1, max_value=9).map(lambda value: 2 * value - 1),
    series=_tensor_3d_strategy(),
)
def test_series_decomposition_supports_multivariate_inputs(
    kernel_size: int,
    series: torch.Tensor,
) -> None:
    """The decomposition should preserve multivariate shape and reconstruct inputs."""
    decomposition = SeriesDecomposition(kernel_size=kernel_size)

    seasonal, trend = decomposition(series)

    assert seasonal.shape == series.shape
    assert trend.shape == series.shape
    assert torch.allclose(seasonal + trend, series, atol=1e-4, rtol=1e-4)
