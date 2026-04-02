"""Public TimeBase model wrappers for NeuralForecast.

`TimeBaseTrend` intentionally uses a local pure-Torch decomposition helper instead of
importing `neuralforecast.models.dlinear.SeriesDecomp`. The local implementation keeps
this module decoupled from unrelated NeuralForecast model internals while preserving the
same seasonal-plus-trend contract. If upstream alignment becomes more valuable later,
maintainers can revert to the dependency with a small internal change.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from timebaseula.models.base import _BaseTimeBaseModel
from timebaseula.models.decomposition import SeriesDecomposition
from timebaseula.models.defaults import DEFAULT_BASIS_NUM, resolve_moving_avg_window
from timebaseula.models.factories import resolve_model_components


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
        components = resolve_model_components(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            freq=freq,
            use_period_norm=use_period_norm,
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
            trainer_kwargs=trainer_kwargs,
        )
        super().__init__(
            h=h,
            model_settings=components.model_settings,
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
        )
        self._initialize_timebase_modules(
            architecture_config=components.architecture_config,
            regularization_config=components.regularization_config,
            horizon=h,
        )
        self._initialize_output_adapter()

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the TimeBase forward pass."""
        forecast, basis = self.core(windows_batch["insample_y"])
        return self._finalize_forecast(forecast, basis)


class TimeBaseTrend(_BaseTimeBaseModel):
    """TimeBase model with an added linear trend branch.

    The trend decomposition intentionally relies on a local pure-Torch implementation
    instead of NeuralForecast's DLinear helper. This keeps the explicit TimeBase family
    decoupled from another model module while preserving a small revert path if the
    project later decides to reuse the upstream dependency again.
    """

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
        components = resolve_model_components(
            h=h,
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            freq=freq,
            use_period_norm=use_period_norm,
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
            trainer_kwargs=trainer_kwargs,
        )
        super().__init__(
            h=h,
            model_settings=components.model_settings,
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
        )
        self.moving_avg_window = resolve_moving_avg_window(moving_avg_window)
        self.decomp = SeriesDecomposition(self.moving_avg_window)
        self._initialize_timebase_modules(
            architecture_config=components.architecture_config,
            regularization_config=components.regularization_config,
            horizon=h,
        )
        self.linear_trend = nn.Linear(self.input_size, h)
        self._initialize_output_adapter()

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the TimeBaseTrend forward pass."""
        seasonal, trend = self.decomp(windows_batch["insample_y"])
        seasonal_forecast, basis = self.core(seasonal)
        trend_forecast = self.linear_trend(trend)
        forecast = trend_forecast + seasonal_forecast
        return self._finalize_forecast(forecast, basis)
