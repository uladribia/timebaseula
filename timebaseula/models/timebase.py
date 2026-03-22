"""TimeBase model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.dlinear import SeriesDecomp

from timebaseula.recommend import (
    DatasetProfile,
    profile_dataset,
    recommend_timebase_kwargs,
    recommend_timebase_trend_kwargs,
)

DEFAULT_LOSS = MAE()


class TrainingLossNaNError(RuntimeError):
    """Raised when the training loss becomes NaN."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Loss is NaN, training stopped.")


@dataclass(frozen=True)
class TimeBaseConfig:
    """Configuration for the TimeBase core components."""

    period_len: int
    basis_num: int
    use_period_norm: bool


class TimeBaseCore(nn.Module):
    """Core TimeBase operations for basis extraction and segment forecasting."""

    def __init__(self, config: TimeBaseConfig, input_size: int, pred_len: int) -> None:
        """Initialize the core TimeBase components."""
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.pred_len = pred_len
        self.period_len = config.period_len
        self.basis_num = config.basis_num
        self.use_period_norm = config.use_period_norm

        self.seg_num_x = (input_size + self.period_len - 1) // self.period_len
        self.seg_num_y = (pred_len + self.period_len - 1) // self.period_len
        self.pad_seq_len = self.seg_num_x * self.period_len - input_size

        self.ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
        self.basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the TimeBase forecast and basis components."""
        batch_size, _ = x.shape

        if self.pad_seq_len > 0:
            x = torch.cat([x, x[:, -self.pad_seq_len :]], dim=-1)

        x = x.reshape(batch_size, self.seg_num_x, self.period_len).permute(0, 2, 1)

        if self.use_period_norm:
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            x_norm = x - period_mean
            basis = self.ts2basis(x_norm)
            forecast_segments = self.basis2ts(basis) + period_mean
        else:
            series_mean = torch.mean(x.reshape(batch_size, -1), dim=-1, keepdim=True)
            x_norm = x.reshape(batch_size, -1) - series_mean
            x_norm = x_norm.reshape(batch_size, self.period_len, self.seg_num_x)
            basis = self.ts2basis(x_norm)
            forecast_segments = self.basis2ts(basis).reshape(batch_size, -1)
            forecast_segments = forecast_segments + series_mean
            forecast_segments = forecast_segments.reshape(
                batch_size, self.period_len, self.seg_num_y
            )

        forecast = forecast_segments.permute(0, 2, 1).reshape(batch_size, -1)
        return forecast[:, : self.pred_len].contiguous(), basis


class BaseTimeBaseWindows(BaseWindows):
    """Shared NeuralForecast wrapper logic for TimeBase-style models."""

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    use_orthogonal: bool
    orthogonal_weight: float
    _last_orthogonal_loss: torch.Tensor | None

    @staticmethod
    def _compute_orthogonal_loss(basis: torch.Tensor) -> torch.Tensor:
        """Compute the orthogonal regularization term for basis components."""
        gram = torch.matmul(basis.transpose(-2, -1), basis)
        diag = torch.diagonal(gram, dim1=-2, dim2=-1)
        off_diag = gram - torch.diag_embed(diag)
        return torch.norm(off_diag, dim=(-2, -1)).mean()

    def _build_windows_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create and normalize the windows batch used during training."""
        windows = self._create_windows(batch, step="train")
        y_idx = batch["y_idx"]
        original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
        windows = self._normalization(windows=windows, y_idx=y_idx)
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)
        windows_batch = {
            "insample_y": insample_y,
            "insample_mask": insample_mask,
            "futr_exog": futr_exog,
            "hist_exog": hist_exog,
            "stat_exog": stat_exog,
        }
        return windows_batch, outsample_y, outsample_mask, original_outsample_y

    def _compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        windows_batch: dict[str, torch.Tensor],
        outsample_y: torch.Tensor,
        outsample_mask: torch.Tensor,
        original_outsample_y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the forecast loss, including optional denormalization."""
        output = self(windows_batch)
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y,
                temporal_cols=batch["temporal_cols"],
                y_idx=batch["y_idx"],
            )
            distr_args = self.loss.scale_decouple(
                output=output,
                loc=y_loc,
                scale=y_scale,
            )
            return self.loss(
                y=original_outsample_y,
                distr_args=distr_args,
                mask=outsample_mask,
            )
        return self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

    def _apply_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply the optional orthogonal penalty to the loss."""
        if self.use_orthogonal and self._last_orthogonal_loss is not None:
            return loss + self.orthogonal_weight * self._last_orthogonal_loss
        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one training step with shared loss bookkeeping."""
        del batch_idx
        windows_batch, outsample_y, outsample_mask, original_outsample_y = (
            self._build_windows_batch(batch)
        )
        loss = self._compute_batch_loss(
            batch=batch,
            windows_batch=windows_batch,
            outsample_y=outsample_y,
            outsample_mask=outsample_mask,
            original_outsample_y=original_outsample_y,
        )
        loss = self._apply_regularization(loss)

        if torch.isnan(loss):
            raise TrainingLossNaNError

        self.log(
            "train_loss",
            loss.item(),
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, loss.item()))
        return loss


class TimeBase(BaseTimeBaseWindows):
    """TimeBase model compatible with NeuralForecast BaseWindows."""

    @staticmethod
    def profile_dataset(
        frame: pd.DataFrame,
        freq: str,
        horizon: int,
    ) -> DatasetProfile:
        """Profile a dataset for TimeBase-oriented default selection."""
        return profile_dataset(frame=frame, freq=freq, horizon=horizon)

    @classmethod
    def recommend_defaults(
        cls,
        frame: pd.DataFrame,
        freq: str,
        horizon: int,
        max_steps: int = 100,
    ) -> dict[str, Any]:
        """Return recommended initialization kwargs for TimeBase."""
        return recommend_timebase_kwargs(
            frame=frame,
            freq=freq,
            horizon=horizon,
            max_steps=max_steps,
        )

    def __init__(
        self,
        h: int,
        input_size: int,
        period_len: int = 24,
        basis_num: int = 6,
        use_period_norm: bool = True,
        use_orthogonal: bool = False,
        orthogonal_weight: float = 0.0,
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int | None = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer: type[torch.optim.Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        **trainer_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the TimeBase wrapper."""
        super().__init__(
            h=h,
            input_size=input_size,
            loss=DEFAULT_LOSS if loss is None else loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **trainer_kwargs,
        )
        self.core = TimeBaseCore(
            TimeBaseConfig(
                period_len=period_len,
                basis_num=basis_num,
                use_period_norm=use_period_norm,
            ),
            input_size=input_size,
            pred_len=h,
        )
        self.use_orthogonal = use_orthogonal
        self.orthogonal_weight = orthogonal_weight
        self._last_orthogonal_loss: torch.Tensor | None = None

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for BaseWindows training and inference."""
        forecast, basis = self.core(windows_batch["insample_y"])
        self._last_orthogonal_loss = (
            self._compute_orthogonal_loss(basis) if self.use_orthogonal else None
        )
        forecast = forecast.reshape(-1, self.h, self.loss.outputsize_multiplier)
        return self.loss.domain_map(forecast)


class TimeBaseTrend(BaseTimeBaseWindows):
    """TimeBase with trend decomposition, compatible with NeuralForecast BaseWindows."""

    @staticmethod
    def profile_dataset(
        frame: pd.DataFrame,
        freq: str,
        horizon: int,
    ) -> DatasetProfile:
        """Profile a dataset for TimeBaseTrend-oriented default selection."""
        return profile_dataset(frame=frame, freq=freq, horizon=horizon)

    @classmethod
    def recommend_defaults(
        cls,
        frame: pd.DataFrame,
        freq: str,
        horizon: int,
        max_steps: int = 100,
    ) -> dict[str, Any]:
        """Return recommended initialization kwargs for TimeBaseTrend."""
        return recommend_timebase_trend_kwargs(
            frame=frame,
            freq=freq,
            horizon=horizon,
            max_steps=max_steps,
        )

    def __init__(
        self,
        h: int,
        input_size: int,
        period_len: int = 24,
        basis_num: int = 6,
        moving_avg_window: int = 25,
        use_period_norm: bool = True,
        use_orthogonal: bool = False,
        orthogonal_weight: float = 0.0,
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int | None = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer: type[torch.optim.Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        **trainer_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the TimeBaseTrend wrapper."""
        if moving_avg_window % 2 == 0:
            msg = "moving_avg_window must be odd for moving average decomposition"
            raise ValueError(msg)

        super().__init__(
            h=h,
            input_size=input_size,
            loss=DEFAULT_LOSS if loss is None else loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **trainer_kwargs,
        )
        self.moving_avg_window = moving_avg_window
        self.decomp = SeriesDecomp(moving_avg_window)
        self.core = TimeBaseCore(
            TimeBaseConfig(
                period_len=period_len,
                basis_num=basis_num,
                use_period_norm=use_period_norm,
            ),
            input_size=input_size,
            pred_len=h,
        )
        self.linear_trend = nn.Linear(input_size, h)
        self.use_orthogonal = use_orthogonal
        self.orthogonal_weight = orthogonal_weight
        self._last_orthogonal_loss: torch.Tensor | None = None

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for BaseWindows training and inference."""
        seasonal, trend = self.decomp(windows_batch["insample_y"])
        timebase_forecast, basis = self.core(seasonal)
        trend_forecast = self.linear_trend(trend)
        forecast = trend_forecast + timebase_forecast
        self._last_orthogonal_loss = (
            self._compute_orthogonal_loss(basis) if self.use_orthogonal else None
        )
        forecast = forecast.reshape(-1, self.h, self.loss.outputsize_multiplier)
        return self.loss.domain_map(forecast)
