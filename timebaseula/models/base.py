"""Shared NeuralForecast wrapper logic for explicit TimeBase models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows

from timebaseula.models.config import (
    ModelSettings,
    TimeBaseArchitectureConfig,
    TimeBaseRegularizationConfig,
)
from timebaseula.models.core import TimeBaseCore
from timebaseula.models.defaults import DEFAULT_LOSS
from timebaseula.models.factories import build_timebase_core


class TrainingLossNaNError(RuntimeError):
    """Raised when the training loss becomes NaN."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Loss is NaN, training stopped.")


class _BaseTimeBaseModel(BaseWindows):
    """Shared NeuralForecast wrapper logic for the public TimeBase models."""

    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    core: TimeBaseCore
    use_orthogonal: bool
    orthogonal_weight: float
    _last_orthogonal_loss: torch.Tensor | None

    def __init__(
        self,
        h: int,
        model_settings: ModelSettings,
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
    ) -> None:
        """Initialize the shared NeuralForecast wrapper state."""
        super().__init__(
            h=h,
            input_size=model_settings.input_size,
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
            **model_settings.trainer_kwargs,
        )
        self.freq = freq
        self.period_len = model_settings.period_len
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

    def _initialize_timebase_modules(
        self,
        architecture_config: TimeBaseArchitectureConfig,
        regularization_config: TimeBaseRegularizationConfig,
        horizon: int,
    ) -> None:
        """Build the shared forecast core and regularization state."""
        self.core = build_timebase_core(
            architecture_config=architecture_config,
            horizon=horizon,
        )
        self._set_regularization(
            use_orthogonal=regularization_config.use_orthogonal,
            orthogonal_weight=regularization_config.orthogonal_weight,
        )

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
