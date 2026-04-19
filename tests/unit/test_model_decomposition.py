"""Tests for the pure Torch decomposition helpers."""

from __future__ import annotations

from hypothesis import given
import torch

from tests.property_strategies import odd_integers, tensor_2d, tensor_3d
from timebaseula.models.decomposition import MovingAverage, SeriesDecomposition


@given(
    kernel_size=odd_integers(1, 17),
    series=tensor_2d(),
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
    kernel_size=odd_integers(1, 17),
    series=tensor_2d(),
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
    kernel_size=odd_integers(1, 17),
    series=tensor_3d(),
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
