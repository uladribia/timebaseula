"""Pure Torch decomposition helpers for TimeBaseTrend.

This module intentionally reimplements the tiny moving-average decomposition used by
NeuralForecast's DLinear model so that `TimeBaseTrend` does not depend on
`neuralforecast.models.dlinear.SeriesDecomp`.

Keeping the logic local makes the TimeBase family easier to reason about and reduces
cross-model coupling. If future compatibility needs make it preferable, the wrapper can
still switch back to the upstream dependency because this module follows the same
seasonal-plus-trend contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Centered moving average with edge-value padding.

    This implementation is intentionally local and Torch-only. It mirrors the behavior
    of NeuralForecast's DLinear moving average while keeping the TimeBaseTrend
    decomposition independent from that model module. If maintainers later prefer to
    reuse the upstream helper again, this block can be replaced without changing the
    public TimeBaseTrend API.
    """

    def __init__(self, kernel_size: int) -> None:
        """Initialize the moving-average block."""
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the moving-average trend estimate for a batch of series."""
        padding = (self.kernel_size - 1) // 2
        x = x.unsqueeze(1)
        front = x[:, :, :1].repeat(1, 1, padding)
        end = x[:, :, -1:].repeat(1, 1, padding)
        padded = torch.cat([front, x, end], dim=-1)
        return self.avg_pool(padded).squeeze(1)


class SeriesDecomposition(nn.Module):
    """Split a series into seasonal residual and moving-average trend.

    The implementation is intentionally local even though NeuralForecast exposes a
    similar `SeriesDecomp` helper. The goal is to keep the explicit TimeBase models
    decoupled from unrelated model modules while preserving an easy revert path to the
    upstream implementation if needed later.
    """

    def __init__(self, kernel_size: int) -> None:
        """Initialize the decomposition module."""
        super().__init__()
        self.moving_average = MovingAverage(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return seasonal residual and trend components."""
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend
