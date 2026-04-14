"""Pure Torch components for the TimeBase model family."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TimeBaseConfig:
    """Configuration for the segmented TimeBase core."""

    input_size: int
    period_len: int
    basis_num: int
    use_period_norm: bool


class TimeBaseCore(nn.Module):
    """Segmented basis projection used by the public TimeBase models.

    The core accepts either a univariate batch ``[batch, time]`` or a multivariate
    batch ``[batch, time, n_series]``. For multivariate inputs, the model follows the
    original TimeBase ``individual=0`` path: each series is projected independently
    with shared ``ts2basis`` and ``basis2ts`` layers.
    """

    def __init__(self, config: TimeBaseConfig, horizon: int) -> None:
        """Build the TimeBase projection layers."""
        super().__init__()
        self.input_size = config.input_size
        self.period_len = config.period_len
        self.pred_len = horizon
        self.basis_num = config.basis_num
        self.use_period_norm = config.use_period_norm

        self.seg_num_x = (self.input_size + self.period_len - 1) // self.period_len
        self.seg_num_y = (self.pred_len + self.period_len - 1) // self.period_len
        self.pad_seq_len = self.seg_num_x * self.period_len - self.input_size

        self.ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
        self.basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def _pad_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the input sequence following the original TimeBase convention."""
        if self.pad_seq_len == 0:
            return x

        pad_start = (self.seg_num_x - 1) * self.period_len
        if x.ndim == 2:
            if pad_start > 0:
                pad_values = x[:, pad_start - self.pad_seq_len : pad_start]
            else:
                pad_values = x[:, -self.pad_seq_len :]
            return torch.cat([x, pad_values], dim=-1)

        transposed = x.permute(0, 2, 1)
        if pad_start > 0:
            pad_values = transposed[:, :, pad_start - self.pad_seq_len : pad_start]
        else:
            pad_values = transposed[:, :, -self.pad_seq_len :]
        padded = torch.cat([transposed, pad_values], dim=-1)
        return padded.permute(0, 2, 1)

    @staticmethod
    def _reshape_input(
        x: torch.Tensor, seg_num_x: int, period_len: int
    ) -> torch.Tensor:
        """Reshape univariate or multivariate inputs into segment batches."""
        if x.ndim == 2:
            batch_size = x.shape[0]
            return x.reshape(batch_size, seg_num_x, period_len).permute(0, 2, 1)

        batch_size, _, n_series = x.shape
        reshaped = x.permute(0, 2, 1).reshape(
            batch_size, n_series, seg_num_x, period_len
        )
        return reshaped.permute(0, 1, 3, 2).reshape(-1, period_len, seg_num_x)

    def _normalize_segments(
        self,
        segments: torch.Tensor,
        batch_size: int,
        n_series: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize segmented inputs before basis projection."""
        if self.use_period_norm:
            period_mean = segments.mean(dim=-1, keepdim=True)
            return segments - period_mean, period_mean

        reshaped = segments.reshape(batch_size, n_series, -1)
        series_mean = reshaped.mean(dim=-1, keepdim=True)
        normalized_segments = reshaped - series_mean
        normalized_segments = normalized_segments.reshape(
            -1, self.period_len, self.seg_num_x
        )
        return normalized_segments, series_mean

    def _denormalize_segments(
        self,
        forecast_segments: torch.Tensor,
        norm_stats: torch.Tensor,
        batch_size: int,
        n_series: int,
    ) -> torch.Tensor:
        """Restore the original scale after basis decoding."""
        if self.use_period_norm:
            return forecast_segments + norm_stats

        restored = forecast_segments.reshape(batch_size, n_series, -1)
        restored = restored + norm_stats
        return restored.reshape(-1, self.period_len, self.seg_num_y)

    @staticmethod
    def _restore_output_shape(
        forecast_segments: torch.Tensor,
        basis: torch.Tensor,
        batch_size: int,
        pred_len: int,
        n_series: int,
        is_multivariate_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return forecasts and basis tensors with their public shapes."""
        if not is_multivariate_input:
            forecast = forecast_segments.permute(0, 2, 1).reshape(batch_size, -1)
            return forecast[:, :pred_len].contiguous(), basis

        forecast = forecast_segments.reshape(
            batch_size,
            n_series,
            forecast_segments.shape[1],
            forecast_segments.shape[2],
        )
        forecast = forecast.permute(0, 1, 3, 2).reshape(batch_size, n_series, -1)
        forecast = forecast[:, :, :pred_len].permute(0, 2, 1).contiguous()
        basis = basis.reshape(batch_size, n_series, basis.shape[1], basis.shape[2])
        return forecast, basis

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project an input window to a forecast and the learned basis."""
        if x.ndim not in {2, 3}:
            msg = "TimeBaseCore expects [batch, time] or [batch, time, n_series] input"
            raise ValueError(msg)

        batch_size = x.shape[0]
        is_multivariate_input = x.ndim == 3
        n_series = 1 if x.ndim == 2 else x.shape[-1]
        x = self._pad_sequence(x)
        segments = self._reshape_input(x, self.seg_num_x, self.period_len)
        normalized_segments, norm_stats = self._normalize_segments(
            segments=segments,
            batch_size=batch_size,
            n_series=n_series,
        )
        basis = self.ts2basis(normalized_segments)
        forecast_segments = self.basis2ts(basis)
        forecast_segments = self._denormalize_segments(
            forecast_segments=forecast_segments,
            norm_stats=norm_stats,
            batch_size=batch_size,
            n_series=n_series,
        )
        return self._restore_output_shape(
            forecast_segments=forecast_segments,
            basis=basis,
            batch_size=batch_size,
            pred_len=self.pred_len,
            n_series=n_series,
            is_multivariate_input=is_multivariate_input,
        )
