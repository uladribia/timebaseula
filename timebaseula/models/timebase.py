"""TimeBase model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.dlinear import SeriesDecomp

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
        """Initialize the core TimeBase components.

        Args:
            config: TimeBase configuration values.
            input_size: Input window length.
            pred_len: Forecast horizon length.
        """
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
        """Compute the TimeBase forecast and basis components.

        Args:
            x: Tensor of shape (batch, time).

        Returns:
            Tuple of (forecast, basis) where forecast has shape (batch, pred_len).
        """
        batch_size, _ = x.shape

        if self.pad_seq_len > 0:
            pad_values = x[:, -self.pad_seq_len :]
            x = torch.cat([x, pad_values], dim=-1)

        x = x.reshape(batch_size, self.seg_num_x, self.period_len).permute(0, 2, 1)

        if self.use_period_norm:
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            x_norm = x - period_mean
            norm_stats = {"period_mean": period_mean}
        else:
            series_mean = torch.mean(x.reshape(batch_size, -1), dim=-1, keepdim=True)
            x_norm = x.reshape(batch_size, -1) - series_mean
            x_norm = x_norm.reshape(batch_size, self.period_len, self.seg_num_x)
            norm_stats = {"mean": series_mean}

        basis = self.ts2basis(x_norm)
        forecast_segments = self.basis2ts(basis)

        if self.use_period_norm:
            forecast_segments = forecast_segments + norm_stats["period_mean"]
        else:
            forecast_segments = forecast_segments.reshape(batch_size, -1)
            forecast_segments = forecast_segments + norm_stats["mean"]
            forecast_segments = forecast_segments.reshape(
                batch_size, self.period_len, self.seg_num_y
            )

        forecast = (
            forecast_segments.permute(0, 2, 1).reshape(batch_size, -1).contiguous()
        )
        return forecast[:, : self.pred_len], basis


class TimeBase(BaseWindows):
    """TimeBase model compatible with NeuralForecast BaseWindows."""

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

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
        """Initialize the TimeBase wrapper.

        Args:
            h: Forecast horizon.
            input_size: Input window length.
            period_len: Segment length.
            basis_num: Number of basis components.
            use_period_norm: Whether to normalize per period.
            use_orthogonal: Whether to enable orthogonal regularization.
            orthogonal_weight: Weight for the orthogonal loss term.
            loss: Training loss module.
            valid_loss: Optional validation loss module.
            max_steps: Maximum training steps.
            learning_rate: Optimizer learning rate.
            num_lr_decays: Number of learning rate decays.
            early_stop_patience_steps: Early stopping patience.
            val_check_steps: Validation check interval.
            batch_size: Training batch size.
            valid_batch_size: Validation batch size.
            windows_batch_size: Number of windows per batch.
            inference_windows_batch_size: Number of windows per inference batch.
            start_padding_enabled: Enable start padding for windows.
            step_size: Window step size.
            scaler_type: Temporal scaler type.
            random_seed: Random seed.
            num_workers_loader: DataLoader worker count.
            drop_last_loader: Drop last incomplete batch.
            optimizer: Optimizer class.
            optimizer_kwargs: Optimizer keyword arguments.
            lr_scheduler: Learning rate scheduler class.
            lr_scheduler_kwargs: Scheduler keyword arguments.
            trainer_kwargs: Additional trainer arguments.
        """
        if loss is None:
            loss = DEFAULT_LOSS
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
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
        insample_y = windows_batch["insample_y"]
        forecast, basis = self.core(insample_y)
        if self.use_orthogonal:
            self._last_orthogonal_loss = self._compute_orthogonal_loss(basis)
        else:
            self._last_orthogonal_loss = None
        forecast = forecast.reshape(-1, self.h, self.loss.outputsize_multiplier)
        return self.loss.domain_map(forecast)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with optional orthogonal regularization."""
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

        output = self(windows_batch)
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        if self.use_orthogonal and self._last_orthogonal_loss is not None:
            loss = loss + self.orthogonal_weight * self._last_orthogonal_loss

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

    @staticmethod
    def _compute_orthogonal_loss(basis: torch.Tensor) -> torch.Tensor:
        """Compute the orthogonal regularization term for basis components."""
        gram = torch.matmul(basis.transpose(-2, -1), basis)
        diag = torch.diagonal(gram, dim1=-2, dim2=-1)
        diag_embed = torch.diag_embed(diag)
        off_diag = gram - diag_embed
        loss = torch.norm(off_diag, dim=(-2, -1))
        return loss.mean()


class TimeBaseTrend(BaseWindows):
    """TimeBase with trend decomposition, compatible with NeuralForecast BaseWindows."""

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

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
        """Initialize the TimeBaseTrend wrapper.

        Args:
            h: Forecast horizon.
            input_size: Input window length.
            period_len: Segment length for TimeBase core.
            basis_num: Number of basis components.
            moving_avg_window: Window size for trend extraction (must be odd).
            use_period_norm: Whether to normalize per period in TimeBase core.
            use_orthogonal: Whether to enable orthogonal regularization.
            orthogonal_weight: Weight for the orthogonal loss term.
            loss: Training loss module.
            valid_loss: Optional validation loss module.
            max_steps: Maximum training steps.
            learning_rate: Optimizer learning rate.
            num_lr_decays: Number of learning rate decays.
            early_stop_patience_steps: Early stopping patience.
            val_check_steps: Validation check interval.
            batch_size: Training batch size.
            valid_batch_size: Validation batch size.
            windows_batch_size: Number of windows per batch.
            inference_windows_batch_size: Number of windows per inference batch.
            start_padding_enabled: Enable start padding for windows.
            step_size: Window step size.
            scaler_type: Temporal scaler type.
            random_seed: Random seed.
            num_workers_loader: DataLoader worker count.
            drop_last_loader: Drop last incomplete batch.
            optimizer: Optimizer class.
            optimizer_kwargs: Optimizer keyword arguments.
            lr_scheduler: Learning rate scheduler class.
            lr_scheduler_kwargs: Scheduler keyword arguments.
            trainer_kwargs: Additional trainer arguments.
        """
        if moving_avg_window % 2 == 0:
            msg = "moving_avg_window must be odd for moving average decomposition"
            raise ValueError(msg)

        if loss is None:
            loss = DEFAULT_LOSS
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
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

        self.trend_linear = nn.Linear(input_size, h)

        self.trend_weight = nn.Parameter(torch.tensor(0.5))

        self.use_orthogonal = use_orthogonal
        self.orthogonal_weight = orthogonal_weight
        self._last_orthogonal_loss: torch.Tensor | None = None

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for BaseWindows training and inference."""
        insample_y = windows_batch["insample_y"]

        trend, seasonal = self.decomp(insample_y)

        timebase_forecast, basis = self.core(seasonal)

        trend_forecast = self.trend_linear(trend)

        w = torch.sigmoid(self.trend_weight)
        forecast = w * trend_forecast + (1 - w) * timebase_forecast

        if self.use_orthogonal:
            self._last_orthogonal_loss = self._compute_orthogonal_loss(basis)
        else:
            self._last_orthogonal_loss = None

        forecast = forecast.reshape(-1, self.h, self.loss.outputsize_multiplier)
        return self.loss.domain_map(forecast)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with optional orthogonal regularization."""
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

        output = self(windows_batch)
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        if self.use_orthogonal and self._last_orthogonal_loss is not None:
            loss = loss + self.orthogonal_weight * self._last_orthogonal_loss

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

    @staticmethod
    def _compute_orthogonal_loss(basis: torch.Tensor) -> torch.Tensor:
        """Compute the orthogonal regularization term for basis components."""
        gram = torch.matmul(basis.transpose(-2, -1), basis)
        diag = torch.diagonal(gram, dim1=-2, dim2=-1)
        diag_embed = torch.diag_embed(diag)
        off_diag = gram - diag_embed
        loss = torch.norm(off_diag, dim=(-2, -1))
        return loss.mean()
