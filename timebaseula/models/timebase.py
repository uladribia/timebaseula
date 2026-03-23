"""Readable TimeBase model implementations for NeuralForecast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.dlinear import SeriesDecomp

DEFAULT_LOSS = MAE()
DEFAULT_BASIS_NUM = 6
DEFAULT_MOVING_AVG_WINDOW = 25


@dataclass(frozen=True)
class TimeBaseConfig:
    """Configuration for the segmented TimeBase core."""

    input_size: int
    period_len: int
    basis_num: int
    use_period_norm: bool


@dataclass(frozen=True)
class _ModelSettings:
    """Resolved settings shared by TimeBase and TimeBaseTrend."""

    input_size: int
    period_len: int
    trainer_kwargs: dict[str, Any]


class TrainingLossNaNError(RuntimeError):
    """Raised when the training loss becomes NaN."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Loss is NaN, training stopped.")


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

        segments = x.reshape(batch_size, self.seg_num_x, self.period_len).permute(0, 2, 1)

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


def _normalize_frequency(freq: str | None) -> str | None:
    """Normalize a frequency string when one is provided."""
    if freq is None:
        return None
    return freq.upper()


def _default_input_size(horizon: int) -> int:
    """Return the default input window size."""
    return max(2 * horizon, 8)


def _default_period_len(horizon: int, input_size: int, freq: str | None) -> int:
    """Return a simple seasonal default for the explicit models."""
    normalized_freq = _normalize_frequency(freq)
    if normalized_freq == "D":
        return min(7, input_size)
    if normalized_freq in {"M", "ME", "MS"}:
        return min(12, input_size)
    return min(max(2, horizon), input_size)


def _default_trainer_kwargs(trainer_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Force CPU-first trainer defaults while preserving user overrides."""
    resolved = dict(trainer_kwargs)
    resolved.setdefault("accelerator", "cpu")
    resolved.setdefault("devices", 1)
    return resolved


def _resolve_model_settings(
    h: int,
    input_size: int | None,
    period_len: int | None,
    freq: str | None,
    trainer_kwargs: dict[str, Any],
) -> _ModelSettings:
    """Resolve shared constructor defaults for public TimeBase models."""
    resolved_input_size = _default_input_size(h) if input_size is None else int(input_size)
    resolved_period_len = (
        _default_period_len(h, resolved_input_size, freq)
        if period_len is None
        else int(period_len)
    )
    return _ModelSettings(
        input_size=resolved_input_size,
        period_len=resolved_period_len,
        trainer_kwargs=_default_trainer_kwargs(trainer_kwargs),
    )


class _BaseTimeBaseModel(BaseWindows):
    """Shared NeuralForecast wrapper logic for the public TimeBase models."""

    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    use_orthogonal: bool
    orthogonal_weight: float
    _last_orthogonal_loss: torch.Tensor | None

    def __init__(
        self,
        h: int,
        input_size: int | None,
        period_len: int | None,
        freq: str | None,
        loss: nn.Module | None,
        valid_loss: nn.Module | None,
        max_steps: int,
        learning_rate: float,
        num_lr_decays: int,
        early_stop_patience_steps: int,
        val_check_steps: int,
        batch_size: int,
        valid_batch_size: int | None,
        windows_batch_size: int,
        inference_windows_batch_size: int,
        start_padding_enabled: bool,
        step_size: int,
        scaler_type: str,
        random_seed: int,
        num_workers_loader: int,
        drop_last_loader: bool,
        optimizer: type[torch.optim.Optimizer] | None,
        optimizer_kwargs: dict[str, Any] | None,
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None,
        lr_scheduler_kwargs: dict[str, Any] | None,
        **trainer_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the shared NeuralForecast wrapper state."""
        settings = _resolve_model_settings(
            h=h,
            input_size=input_size,
            period_len=period_len,
            freq=freq,
            trainer_kwargs=trainer_kwargs,
        )

        super().__init__(
            h=h,
            input_size=settings.input_size,
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
            **settings.trainer_kwargs,
        )
        self.freq = freq
        self.period_len = settings.period_len
        self.use_orthogonal = False
        self.orthogonal_weight = 0.0
        self._last_orthogonal_loss = None

    @staticmethod
    def _compute_orthogonal_loss(basis: torch.Tensor) -> torch.Tensor:
        """Compute the orthogonal regularization term for basis components."""
        gram = torch.matmul(basis.transpose(-2, -1), basis)
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        off_diagonal = gram - torch.diag_embed(diagonal)
        return torch.norm(off_diagonal, dim=(-2, -1)).mean()

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
            distribution_args = self.loss.scale_decouple(
                output=output,
                loc=y_loc,
                scale=y_scale,
            )
            return self.loss(
                y=original_outsample_y,
                distr_args=distribution_args,
                mask=outsample_mask,
            )
        return self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

    def _apply_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply the optional orthogonal penalty to the loss."""
        if self.use_orthogonal and self._last_orthogonal_loss is not None:
            return loss + self.orthogonal_weight * self._last_orthogonal_loss
        return loss

    def _set_regularization(
        self,
        use_orthogonal: bool,
        orthogonal_weight: float,
    ) -> None:
        """Store orthogonal regularization settings."""
        self.use_orthogonal = use_orthogonal
        self.orthogonal_weight = orthogonal_weight
        self._last_orthogonal_loss = None

    def _finalize_forecast(
        self,
        forecast: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """Apply shared post-processing for a model forecast."""
        self._last_orthogonal_loss = (
            self._compute_orthogonal_loss(basis) if self.use_orthogonal else None
        )
        forecast = forecast.reshape(-1, self.h, self.loss.outputsize_multiplier)
        return self.loss.domain_map(forecast)

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


class TimeBase(_BaseTimeBaseModel):
    """Compact segmented-basis model for NeuralForecast."""

    def __init__(
        self,
        h: int,
        input_size: int | None = None,
        period_len: int | None = None,
        basis_num: int = DEFAULT_BASIS_NUM,
        freq: str | None = None,
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
        """Initialize the explicit TimeBase model."""
        super().__init__(
            h=h,
            input_size=input_size,
            period_len=period_len,
            freq=freq,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **trainer_kwargs,
        )
        self.core = TimeBaseCore(
            TimeBaseConfig(
                input_size=self.input_size,
                period_len=self.period_len,
                basis_num=int(basis_num),
                use_period_norm=use_period_norm,
            ),
            horizon=h,
        )
        self._set_regularization(
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
        )

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the TimeBase forward pass."""
        forecast, basis = self.core(windows_batch["insample_y"])
        return self._finalize_forecast(forecast, basis)


class TimeBaseTrend(_BaseTimeBaseModel):
    """TimeBase model with an added linear trend branch."""

    def __init__(
        self,
        h: int,
        input_size: int | None = None,
        period_len: int | None = None,
        basis_num: int = DEFAULT_BASIS_NUM,
        moving_avg_window: int | None = None,
        freq: str | None = None,
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
        """Initialize the explicit TimeBase model with trend decomposition."""
        super().__init__(
            h=h,
            input_size=input_size,
            period_len=period_len,
            freq=freq,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **trainer_kwargs,
        )

        self.moving_avg_window = self._resolve_moving_avg_window(moving_avg_window)
        self.decomp = SeriesDecomp(self.moving_avg_window)
        self.core = TimeBaseCore(
            TimeBaseConfig(
                input_size=self.input_size,
                period_len=self.period_len,
                basis_num=int(basis_num),
                use_period_norm=use_period_norm,
            ),
            horizon=h,
        )
        self.linear_trend = nn.Linear(self.input_size, h)
        self._set_regularization(
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
        )

    @staticmethod
    def _resolve_moving_avg_window(moving_avg_window: int | None) -> int:
        """Return a valid moving average window for trend decomposition."""
        resolved_window = (
            DEFAULT_MOVING_AVG_WINDOW
            if moving_avg_window is None
            else int(moving_avg_window)
        )
        if resolved_window % 2 == 0:
            msg = "moving_avg_window must be odd for moving average decomposition"
            raise ValueError(msg)
        return resolved_window

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the TimeBaseTrend forward pass."""
        seasonal, trend = self.decomp(windows_batch["insample_y"])
        seasonal_forecast, basis = self.core(seasonal)
        trend_forecast = self.linear_trend(trend)
        forecast = trend_forecast + seasonal_forecast
        return self._finalize_forecast(forecast, basis)
