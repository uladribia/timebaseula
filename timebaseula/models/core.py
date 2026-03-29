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
    """Segmented basis projection used by the public TimeBase models."""

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project an input window to a forecast and the learned basis."""
        batch_size, _ = x.shape

        if self.pad_seq_len > 0:
            x = torch.cat([x, x[:, -self.pad_seq_len :]], dim=-1)

        segments = x.reshape(batch_size, self.seg_num_x, self.period_len).permute(
            0, 2, 1
        )

        if self.use_period_norm:
            period_mean = segments.mean(dim=-1, keepdim=True)
            normalized_segments = segments - period_mean
            basis = self.ts2basis(normalized_segments)
            forecast_segments = self.basis2ts(basis) + period_mean
        else:
            series_mean = segments.reshape(batch_size, -1).mean(dim=-1, keepdim=True)
            normalized_segments = segments.reshape(batch_size, -1) - series_mean
            normalized_segments = normalized_segments.reshape(
                batch_size,
                self.period_len,
                self.seg_num_x,
            )
            basis = self.ts2basis(normalized_segments)
            forecast_segments = self.basis2ts(basis).reshape(batch_size, -1)
            forecast_segments = forecast_segments + series_mean
            forecast_segments = forecast_segments.reshape(
                batch_size,
                self.period_len,
                self.seg_num_y,
            )

        forecast = forecast_segments.permute(0, 2, 1).reshape(batch_size, -1)
        return forecast[:, : self.pred_len].contiguous(), basis
