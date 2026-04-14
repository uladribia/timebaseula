"""Shared NeuralForecast wrapper logic for explicit TimeBase models.

This module keeps the public ``TimeBase`` and ``TimeBaseTrend`` API aligned with
NeuralForecast's long-format interface while switching the internal sampling strategy
from per-series windows to multivariate windows. The implementation follows the
shape contract of ``BaseMultivariate`` closely, but it keeps local control over
loss handling so the explicit TimeBase family continues to support point,
quantile, and distribution losses through the existing public API.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from neuralforecast.common._base_model import BaseModel
from neuralforecast.common._base_multivariate import BaseMultivariate, pl
from neuralforecast.common._scalers import TemporalNorm
from neuralforecast.tsdataset import TimeSeriesDataModule

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


class _BaseTimeBaseModel(BaseMultivariate):
    """Shared NeuralForecast wrapper logic for the public TimeBase models.

    The public TimeBase classes now use NeuralForecast's multivariate sampling
    strategy so each training window contains all active series together. This
    matches the original TimeBase ``individual=0`` behavior more closely while
    preserving the existing long-format ``unique_id`` API.
    """

    SAMPLING_TYPE = "multivariate"
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    core: TimeBaseCore
    use_orthogonal: bool
    orthogonal_weight: float
    output_adapter: nn.Module
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
        BaseModel.__init__(
            self,
            random_seed=random_seed,
            loss=DEFAULT_LOSS if loss is None else loss,
            valid_loss=valid_loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            futr_exog_list=None,
            hist_exog_list=None,
            stat_exog_list=None,
            max_steps=max_steps,
            early_stop_patience_steps=early_stop_patience_steps,
            **model_settings.trainer_kwargs,
        )

        self.h = h
        self.input_size = model_settings.input_size
        self.n_series = 1
        self.freq = freq
        self.period_len = model_settings.period_len
        self.batch_size = int(windows_batch_size)
        self.user_batch_size = int(batch_size)
        self.valid_batch_size = (
            self.user_batch_size if valid_batch_size is None else int(valid_batch_size)
        )
        self.inference_windows_batch_size = int(inference_windows_batch_size)
        self.start_padding_enabled = start_padding_enabled
        self.step_size = int(step_size)
        self.num_lr_decays = int(num_lr_decays)
        self.lr_decay_steps = (
            max(max_steps // self.num_lr_decays, 1) if self.num_lr_decays > 0 else 10e7
        )
        self.val_check_steps = int(val_check_steps)
        self.max_steps = int(max_steps)
        self.learning_rate = float(learning_rate)
        self.early_stop_patience_steps = int(early_stop_patience_steps)
        self.num_workers_loader = int(num_workers_loader)
        self.dataloader_kwargs = None
        self.drop_last_loader = drop_last_loader
        self.validation_step_outputs = []
        self.alias = None
        self.decompose_forecast = False
        self.val_size = 0
        self.test_size = 0
        self.padder = nn.ConstantPad1d(padding=(0, self.h), value=0.0)
        self.scaler = TemporalNorm(scaler_type=scaler_type, dim=2)

        self.use_orthogonal = False
        self.orthogonal_weight = 0.0
        self.output_adapter = nn.Identity()
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

    def _initialize_output_adapter(self) -> None:
        """Build the optional loss-aware output adapter."""
        output_width = self.h * self.loss.outputsize_multiplier
        if output_width == self.h:
            self.output_adapter = nn.Identity()
            return
        self.output_adapter = nn.Linear(self.h, output_width)

    def _set_active_n_series(self, dataset: Any) -> None:
        """Update the active series count from a NeuralForecast dataset."""
        self.n_series = int(dataset.n_groups)

    def _create_windows(
        self,
        batch: dict[str, torch.Tensor],
        step: str,
        w_idxs: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Create multivariate windows following NeuralForecast's joint sampling."""
        window_size = self.input_size + self.h
        temporal_cols = cast(Any, batch["temporal_cols"])
        temporal = batch["temporal"]

        if step == "train":
            if self.val_size + self.test_size > 0:
                cutoff = -self.val_size - self.test_size
                temporal = temporal[:, :, :cutoff]

            temporal = self.padder(temporal)
            windows = temporal.unfold(
                dimension=-1, size=window_size, step=self.step_size
            )

            available_idx = temporal_cols.get_loc("available_mask")
            sample_condition = windows[:, available_idx, :, -self.h :]
            sample_condition = torch.sum(sample_condition, dim=2)
            sample_condition = torch.sum(sample_condition, dim=0)
            available_condition = windows[:, available_idx, :, : -self.h]
            available_condition = torch.sum(available_condition, dim=2)
            available_condition = torch.sum(available_condition, dim=0)
            final_condition = (sample_condition > 0) & (available_condition > 0)
            windows = windows[:, :, final_condition, :]

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)
            if final_condition.sum() == 0:
                raise Exception("No windows available for training")

            n_windows = windows.shape[2]
            if self.batch_size is not None:
                if w_idxs is None:
                    sampled_idxs = np.random.choice(
                        n_windows,
                        size=self.batch_size,
                        replace=(n_windows < self.batch_size),
                    )
                else:
                    sampled_idxs = w_idxs
                windows = windows[:, :, sampled_idxs, :]

            windows = windows.permute(2, 1, 3, 0)
            return {
                "temporal": windows,
                "temporal_cols": temporal_cols,
                "static": static,
                "static_cols": static_cols,
            }

        if step in {"predict", "val"}:
            if step == "predict":
                predict_step_size = self.predict_step_size
                cutoff = -self.input_size - self.test_size
                temporal = batch["temporal"][:, :, cutoff:]
            else:
                predict_step_size = self.step_size
                cutoff = -self.input_size - self.val_size - self.test_size
                if self.test_size > 0:
                    temporal = batch["temporal"][:, :, cutoff : -self.test_size]
                else:
                    temporal = batch["temporal"][:, :, cutoff:]

            if (
                step == "predict"
                and self.test_size == 0
                and len(self.futr_exog_list) == 0
            ):
                temporal = self.padder(temporal)

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=predict_step_size
            )
            windows = windows.permute(2, 1, 3, 0)
            if w_idxs is not None:
                windows = windows[w_idxs]

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)
            return {
                "temporal": windows,
                "temporal_cols": temporal_cols,
                "static": static,
                "static_cols": static_cols,
            }

        msg = f"Unknown step {step}"
        raise ValueError(msg)

    def _inv_normalization(
        self,
        y_hat: torch.Tensor,
        temporal_cols: Any,
        y_idx: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Invert normalization for tensors with optional output dimensions."""
        del temporal_cols
        resolved_y_idx = int(y_idx)
        y_scale = self.scaler.x_scale[:, [resolved_y_idx], :].squeeze(1)
        y_loc = self.scaler.x_shift[:, [resolved_y_idx], :].squeeze(1)

        while y_scale.ndim < y_hat.ndim:
            y_scale = y_scale.unsqueeze(-1)
            y_loc = y_loc.unsqueeze(-1)

        y_hat = self.scaler.inverse_transform(z=y_hat, x_scale=y_scale, x_shift=y_loc)
        return y_hat, y_loc, y_scale

    def _set_regularization(
        self,
        use_orthogonal: bool,
        orthogonal_weight: float,
    ) -> None:
        """Store orthogonal regularization settings."""
        self.use_orthogonal = use_orthogonal
        self.orthogonal_weight = orthogonal_weight
        self._last_orthogonal_loss = None

    def _apply_output_adapter(self, forecast: torch.Tensor) -> torch.Tensor:
        """Project deterministic forecasts to the loss output width."""
        multiplier = self.loss.outputsize_multiplier
        if forecast.ndim == 2:
            if multiplier == 1:
                return forecast.unsqueeze(-1)
            adapted = self.output_adapter(forecast)
            return adapted.reshape(-1, self.h, multiplier)

        batch_size, _, n_series = forecast.shape
        if multiplier == 1:
            return forecast.unsqueeze(-1)
        adapted = self.output_adapter(forecast.permute(0, 2, 1))
        adapted = adapted.reshape(batch_size, n_series, self.h, multiplier)
        return adapted.permute(0, 2, 1, 3)

    def _finalize_forecast(
        self,
        forecast: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """Apply shared post-processing for a model forecast."""
        self._last_orthogonal_loss = (
            self._compute_orthogonal_loss(basis) if self.use_orthogonal else None
        )
        output = self._apply_output_adapter(forecast)
        return self.loss.domain_map(output)

    def _apply_distribution_sample_shape(
        self,
        distribution_output: tuple[torch.Tensor, ...],
        series_count: int,
        window_count: int,
    ) -> torch.Tensor:
        """Sample multivariate distribution outputs with a flattened series axis."""
        flattened_args = tuple(
            argument.permute(0, 2, 1).reshape(window_count * series_count, self.h)
            for argument in distribution_output
        )
        _, sample_mean, quants = self.loss.sample(distr_args=flattened_args)
        sample_mean = sample_mean.reshape(window_count, series_count, self.h, 1)
        quants = quants.reshape(window_count, series_count, self.h, -1)
        y_hat = torch.cat((sample_mean, quants), dim=-1).permute(0, 2, 1, 3)

        if self.loss.return_params:
            stacked_args = torch.stack(flattened_args, dim=-1)
            stacked_args = stacked_args.reshape(window_count, series_count, self.h, -1)
            stacked_args = stacked_args.permute(0, 2, 1, 3)
            y_hat = torch.cat((y_hat, stacked_args), dim=-1)
        return y_hat

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one training step with shared loss bookkeeping."""
        del batch_idx
        loss = super().training_step(batch, batch_idx=0)
        if self.use_orthogonal and self._last_orthogonal_loss is not None:
            loss = loss + self.orthogonal_weight * self._last_orthogonal_loss
        if torch.isnan(loss):
            raise TrainingLossNaNError
        return loss

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Predict on multivariate windows with loss-aware output formatting."""
        del batch_idx
        windows = self._create_windows(batch, step="predict")
        y_idx = int(batch["y_idx"])
        n_windows = len(windows["temporal"])
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0 or windows_batch_size >= n_windows:
            windows_batch_size = n_windows

        y_hats: list[torch.Tensor] = []
        for start in range(0, n_windows, windows_batch_size):
            stop = min(start + windows_batch_size, n_windows)
            w_idxs = np.arange(start, stop)
            windows = self._create_windows(batch, step="predict", w_idxs=w_idxs)
            windows = self._normalization(windows=windows, y_idx=y_idx)
            insample_y, insample_mask, _, _, hist_exog, futr_exog, stat_exog = (
                self._parse_windows(batch, windows)
            )
            windows_batch = {
                "insample_y": insample_y,
                "insample_mask": insample_mask,
                "futr_exog": futr_exog,
                "hist_exog": hist_exog,
                "stat_exog": stat_exog,
            }
            output_batch = self(windows_batch)
            series_count = insample_y.shape[-1]
            if self.loss.is_distribution_output:
                _, y_loc, y_scale = self._inv_normalization(
                    y_hat=torch.empty(
                        size=(insample_y.shape[0], self.h, series_count),
                        dtype=output_batch[0].dtype,
                        device=output_batch[0].device,
                    ),
                    temporal_cols=batch["temporal_cols"],
                    y_idx=y_idx,
                )
                distr_args = self.loss.scale_decouple(
                    output=output_batch,
                    loc=y_loc,
                    scale=y_scale,
                )
                y_hat = self._apply_distribution_sample_shape(
                    distribution_output=distr_args,
                    series_count=series_count,
                    window_count=insample_y.shape[0],
                )
            else:
                y_hat, _, _ = self._inv_normalization(
                    y_hat=output_batch,
                    temporal_cols=batch["temporal_cols"],
                    y_idx=y_idx,
                )
                if y_hat.ndim == 3:
                    y_hat = y_hat.unsqueeze(-1)
            y_hats.append(y_hat)

        return torch.cat(y_hats, dim=0)

    def fit(
        self,
        dataset: Any,
        val_size: int = 0,
        test_size: int = 0,
        random_seed: int | None = None,
        distributed_config: Any = None,
    ) -> Any:
        """Fit the model using multivariate joint windows across all active series."""
        if distributed_config is not None:
            msg = "multivariate TimeBase models do not support distributed training"
            raise ValueError(msg)
        self._set_active_n_series(dataset)
        return self._fit(
            dataset=dataset,
            batch_size=self.n_series,
            valid_batch_size=self.n_series,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            shuffle_train=False,
            distributed_config=None,
        )

    def predict(
        self,
        dataset: Any,
        test_size: int | None = None,
        step_size: int = 1,
        random_seed: int | None = None,
        **data_module_kwargs: Any,
    ) -> np.ndarray:
        """Predict with multivariate windows while preserving NeuralForecast output order."""
        del test_size
        self._set_active_n_series(dataset)
        self._check_exog(dataset)
        self._restart_seed(random_seed)
        data_module_kwargs = self._set_quantile_for_iqloss(**data_module_kwargs)

        self.predict_step_size = step_size
        self.decompose_forecast = False
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.n_series,
            batch_size=self.n_series,
            **data_module_kwargs,
        )

        pred_trainer_kwargs = self.trainer_kwargs.copy()
        if (pred_trainer_kwargs.get("accelerator", None) == "gpu") and (
            torch.cuda.device_count() > 1
        ):
            pred_trainer_kwargs["devices"] = [0]

        trainer = pl.Trainer(**pred_trainer_kwargs)
        forecast_batches = cast(
            list[torch.Tensor], trainer.predict(self, datamodule=datamodule)
        )
        if not forecast_batches:
            msg = "Prediction produced no forecast batches."
            raise RuntimeError(msg)
        forecasts = torch.cat(forecast_batches, dim=0).numpy()
        forecasts = np.transpose(forecasts, (2, 0, 1, 3))
        forecasts = forecasts.reshape(-1, forecasts.shape[-1])
        return forecasts
