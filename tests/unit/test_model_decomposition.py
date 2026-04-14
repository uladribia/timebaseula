"""Tests for the pure Torch decomposition helpers."""

from __future__ import annotations

import torch

from timebaseula.models.decomposition import MovingAverage, SeriesDecomposition


def test_moving_average_preserves_expected_shape() -> None:
    """The moving-average block should keep the original batch and time shape."""
    moving_average = MovingAverage(kernel_size=5)

    trend = moving_average(torch.arange(12, dtype=torch.float32).reshape(2, 6))

    assert trend.shape == (2, 6)


def test_series_decomposition_reconstructs_original_signal() -> None:
    """Seasonal and trend components should sum back to the input signal."""
    decomposition = SeriesDecomposition(kernel_size=3)
    series = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])

    seasonal, trend = decomposition(series)

    assert seasonal.shape == series.shape
    assert trend.shape == series.shape
    assert torch.allclose(seasonal + trend, series)


def test_series_decomposition_supports_multivariate_inputs() -> None:
    """The decomposition should preserve a trailing series dimension."""
    decomposition = SeriesDecomposition(kernel_size=3)
    series = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)

    seasonal, trend = decomposition(series)

    assert seasonal.shape == series.shape
    assert trend.shape == series.shape
    assert torch.allclose(seasonal + trend, series)
