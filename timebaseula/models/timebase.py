"""TimeBase model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import pandas as pd
import torch
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.dlinear import SeriesDecomp

from timebaseula.recommend import (
    DatasetProfile,
    candidate_periods_for_frequency,
    profile_dataset,
    recommend_timebase_kwargs,
    recommend_timebase_trend_kwargs,
    trim_frame_for_recommendation,
)

DEFAULT_LOSS = MAE()


class RecommendationDataset(Protocol):
    """Minimal dataset interface needed for auto-configuration."""

    temporal: torch.Tensor
    y_idx: int
    indptr: Any


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
        include_iteration_recommendation: bool = False,
    ) -> dict[str, Any]:
        """Return recommended initialization kwargs for TimeBase."""
        return recommend_timebase_kwargs(
            frame=frame,
            freq=freq,
            horizon=horizon,
            max_steps=max_steps,
            include_iteration_recommendation=include_iteration_recommendation,
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
        include_iteration_recommendation: bool = False,
    ) -> dict[str, Any]:
        """Return recommended initialization kwargs for TimeBaseTrend."""
        return recommend_timebase_trend_kwargs(
            frame=frame,
            freq=freq,
            horizon=horizon,
            max_steps=max_steps,
            include_iteration_recommendation=include_iteration_recommendation,
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


class AutoTimeBase(TimeBase):
    """TimeBase with dataset-aware defaults and an optional short search."""

    @classmethod
    def recommend_defaults(
        cls,
        frame: pd.DataFrame,
        freq: str,
        horizon: int,
        max_steps: int = 100,
        include_iteration_recommendation: bool = False,
        holdout_size: int = 0,
    ) -> dict[str, Any]:
        """Return recommendation-driven defaults for AutoTimeBase."""
        return recommend_timebase_kwargs(
            frame=frame,
            freq=freq,
            horizon=horizon,
            max_steps=max_steps,
            holdout_size=holdout_size,
            include_iteration_recommendation=include_iteration_recommendation,
        )

    def __init__(
        self,
        h: int,
        freq: str,
        input_size: int | None = None,
        period_len: int | None = None,
        basis_num: int | None = None,
        use_period_norm: bool = True,
        use_orthogonal: bool = False,
        orthogonal_weight: float = 0.0,
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        max_steps: int = 5000,
        learning_rate: float | None = None,
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
        search_enabled: bool = True,
        search_max_steps: int = 20,
        n_search_configs: int = 3,
        include_iteration_recommendation: bool = False,
        **trainer_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the auto-configuring TimeBase wrapper."""
        base_input_size = 16 if input_size is None else int(input_size)
        base_period_len = 7 if period_len is None else int(period_len)
        base_basis_num = 4 if basis_num is None else int(basis_num)
        base_learning_rate = 1e-3 if learning_rate is None else float(learning_rate)
        super().__init__(
            h=h,
            input_size=base_input_size,
            period_len=base_period_len,
            basis_num=base_basis_num,
            use_period_norm=use_period_norm,
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=base_learning_rate,
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
        self.freq = freq
        self.search_enabled = search_enabled
        self.search_max_steps = search_max_steps
        self.n_search_configs = n_search_configs
        self.include_iteration_recommendation = include_iteration_recommendation
        self.selected_config_: dict[str, Any] = {
            "input_size": base_input_size,
            "period_len": base_period_len,
            "basis_num": base_basis_num,
            "max_steps": max_steps,
            "learning_rate": base_learning_rate,
            "early_stop_patience_steps": early_stop_patience_steps,
            "val_check_steps": val_check_steps,
        }
        self.candidate_results_: list[dict[str, Any]] = []
        self.recommended_training_iterations_: int | None = None
        self.profile_: DatasetProfile | None = None

    def _dataset_to_frame(self, dataset: object) -> pd.DataFrame:
        """Convert a NeuralForecast dataset into a long-format frame."""
        recommendation_dataset = cast(RecommendationDataset, dataset)
        values = recommendation_dataset.temporal.numpy()
        time_col = int(recommendation_dataset.y_idx)
        rows: list[pd.DataFrame] = []
        for series_idx in range(len(recommendation_dataset.indptr) - 1):
            start = int(recommendation_dataset.indptr[series_idx])
            end = int(recommendation_dataset.indptr[series_idx + 1])
            length = end - start
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": [f"series_{series_idx}"] * length,
                        "ds": pd.RangeIndex(length),
                        "y": values[start:end, time_col],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def _recommendation_frame(
        self,
        dataset: object,
        val_size: int,
        test_size: int,
    ) -> pd.DataFrame:
        """Build the leakage-safe frame used for profiling and search."""
        frame = self._dataset_to_frame(dataset)
        return trim_frame_for_recommendation(frame, holdout_size=val_size + test_size)

    def _build_search_configs(
        self, recommendation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Create a compact local search around the heuristic recommendation."""
        base = {
            key: value
            for key, value in recommendation.items()
            if key != "recommended_training_iterations"
        }
        input_size = int(base["input_size"])
        period_candidates = candidate_periods_for_frequency(self.freq, input_size)
        candidate_configs = [base]

        smaller_input = max(self.h * 2, input_size - max(2, input_size // 4))
        larger_input = input_size + max(2, input_size // 4)
        if smaller_input != input_size:
            candidate_configs.append(
                {
                    **base,
                    "input_size": smaller_input,
                    "period_len": min(int(base["period_len"]), smaller_input),
                    "basis_num": max(2, int(base["basis_num"]) - 1),
                    "learning_rate": float(base["learning_rate"]) / 2.0,
                }
            )
        candidate_configs.append(
            {
                **base,
                "input_size": larger_input,
                "period_len": min(max(period_candidates), larger_input),
                "basis_num": int(base["basis_num"]) + 1,
                "learning_rate": float(base["learning_rate"]),
            }
        )
        unique_configs: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for config in candidate_configs:
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_configs.append(config)
        return unique_configs[: max(1, self.n_search_configs)]

    def _candidate_model_kwargs(
        self, config: dict[str, Any], max_steps: int
    ) -> dict[str, Any]:
        """Convert a selected config into model kwargs for a short run."""
        trainer_kwargs = {
            key: value
            for key, value in self.trainer_kwargs.items()
            if key != "callbacks"
        }
        return {
            **config,
            "max_steps": max_steps,
            "use_period_norm": self.core.use_period_norm,
            "use_orthogonal": self.use_orthogonal,
            "orthogonal_weight": self.orthogonal_weight,
            "loss": self.loss,
            "valid_loss": self.valid_loss,
            "num_lr_decays": getattr(self, "num_lr_decays", -1),
            "batch_size": getattr(self, "batch_size", 32),
            "valid_batch_size": getattr(self, "valid_batch_size", None),
            "windows_batch_size": getattr(self, "windows_batch_size", 1024),
            "inference_windows_batch_size": getattr(
                self, "inference_windows_batch_size", 1024
            ),
            "start_padding_enabled": getattr(self, "start_padding_enabled", False),
            "step_size": getattr(self, "step_size", 1),
            "scaler_type": getattr(self, "scaler_type", "identity"),
            "random_seed": getattr(self, "random_seed", 1),
            "num_workers_loader": getattr(self, "num_workers_loader", 0),
            "drop_last_loader": getattr(self, "drop_last_loader", False),
            "optimizer": getattr(self, "optimizer", None),
            "optimizer_kwargs": getattr(self, "optimizer_kwargs", None),
            "lr_scheduler": getattr(self, "lr_scheduler", None),
            "lr_scheduler_kwargs": getattr(self, "lr_scheduler_kwargs", None),
            **trainer_kwargs,
        }

    def _score_candidate(
        self,
        dataset: object,
        val_size: int,
        test_size: int,
        config: dict[str, Any],
    ) -> float:
        """Fit one short candidate model and return its validation score."""
        short_model = TimeBase(
            h=self.h, **self._candidate_model_kwargs(config, self.search_max_steps)
        )
        fitted = short_model.fit(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            random_seed=self.random_seed,
        )
        metrics = getattr(fitted, "metrics", {})
        score = metrics.get("ptl/val_loss", metrics.get("valid_loss"))
        return float(score)

    def _select_config(
        self,
        dataset: object,
        val_size: int,
        test_size: int,
    ) -> dict[str, Any]:
        """Choose the final configuration from heuristics and short search."""
        recommendation_frame = self._recommendation_frame(dataset, val_size, test_size)
        recommendation = self.recommend_defaults(
            recommendation_frame,
            freq=self.freq,
            horizon=self.h,
            max_steps=self.max_steps,
            holdout_size=0,
            include_iteration_recommendation=self.include_iteration_recommendation,
        )
        self.profile_ = profile_dataset(recommendation_frame, self.freq, self.h)
        if "recommended_training_iterations" in recommendation:
            self.recommended_training_iterations_ = int(
                recommendation["recommended_training_iterations"]
            )
        if not self.search_enabled or val_size <= 0:
            return {
                key: value
                for key, value in recommendation.items()
                if key != "recommended_training_iterations"
            }

        best_config = None
        best_score = float("inf")
        self.candidate_results_ = []
        for config in self._build_search_configs(recommendation):
            score = self._score_candidate(dataset, val_size, test_size, config)
            candidate_result = {**config, "validation_score": score}
            self.candidate_results_.append(candidate_result)
            if score < best_score:
                best_score = score
                best_config = config
        if best_config is None:
            msg = "AutoTimeBase search could not score any candidate configuration."
            raise RuntimeError(msg)
        return best_config

    def _apply_selected_config(self, config: dict[str, Any]) -> None:
        """Apply the selected configuration before the final fit."""
        self.input_size = int(config["input_size"])
        self.learning_rate = float(config["learning_rate"])
        self.max_steps = int(config["max_steps"])
        self.early_stop_patience_steps = int(config["early_stop_patience_steps"])
        self.val_check_steps = int(config["val_check_steps"])
        self.core = TimeBaseCore(
            TimeBaseConfig(
                period_len=int(config["period_len"]),
                basis_num=int(config["basis_num"]),
                use_period_norm=self.core.use_period_norm,
            ),
            input_size=self.input_size,
            pred_len=self.h,
        )
        self.selected_config_ = dict(config)
        self.hparams.update(
            {
                "freq": self.freq,
                "input_size": self.input_size,
                "period_len": int(config["period_len"]),
                "basis_num": int(config["basis_num"]),
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate,
                "early_stop_patience_steps": self.early_stop_patience_steps,
                "val_check_steps": self.val_check_steps,
            }
        )

    def fit(
        self,
        dataset: object,
        val_size: int = 0,
        test_size: int = 0,
        random_seed: int | None = None,
        distributed_config: object | None = None,
    ) -> object:
        """Select parameters safely from fit data, then train the final model."""
        selected_config = self._select_config(dataset, val_size, test_size)
        self._apply_selected_config(selected_config)
        return super().fit(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            distributed_config=distributed_config,
        )


class AutoTimeBaseTrend(TimeBaseTrend):
    """TimeBaseTrend with dataset-aware defaults and an optional short search."""

    @classmethod
    def recommend_defaults(
        cls,
        frame: pd.DataFrame,
        freq: str,
        horizon: int,
        max_steps: int = 100,
        include_iteration_recommendation: bool = False,
        holdout_size: int = 0,
    ) -> dict[str, Any]:
        """Return recommendation-driven defaults for AutoTimeBaseTrend."""
        return recommend_timebase_trend_kwargs(
            frame=frame,
            freq=freq,
            horizon=horizon,
            max_steps=max_steps,
            holdout_size=holdout_size,
            include_iteration_recommendation=include_iteration_recommendation,
        )

    def __init__(
        self,
        h: int,
        freq: str,
        input_size: int | None = None,
        period_len: int | None = None,
        basis_num: int | None = None,
        moving_avg_window: int | None = None,
        use_period_norm: bool = True,
        use_orthogonal: bool = False,
        orthogonal_weight: float = 0.0,
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        max_steps: int = 5000,
        learning_rate: float | None = None,
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
        search_enabled: bool = True,
        search_max_steps: int = 20,
        n_search_configs: int = 3,
        include_iteration_recommendation: bool = False,
        **trainer_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the auto-configuring TimeBaseTrend wrapper."""
        base_input_size = 16 if input_size is None else int(input_size)
        base_period_len = 7 if period_len is None else int(period_len)
        base_basis_num = 4 if basis_num is None else int(basis_num)
        base_window = 5 if moving_avg_window is None else int(moving_avg_window)
        base_learning_rate = 1e-3 if learning_rate is None else float(learning_rate)
        super().__init__(
            h=h,
            input_size=base_input_size,
            period_len=base_period_len,
            basis_num=base_basis_num,
            moving_avg_window=base_window,
            use_period_norm=use_period_norm,
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=base_learning_rate,
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
        self.freq = freq
        self.search_enabled = search_enabled
        self.search_max_steps = search_max_steps
        self.n_search_configs = n_search_configs
        self.include_iteration_recommendation = include_iteration_recommendation
        self.selected_config_: dict[str, Any] = {
            "input_size": base_input_size,
            "period_len": base_period_len,
            "basis_num": base_basis_num,
            "moving_avg_window": base_window,
            "max_steps": max_steps,
            "learning_rate": base_learning_rate,
            "early_stop_patience_steps": early_stop_patience_steps,
            "val_check_steps": val_check_steps,
        }
        self.candidate_results_: list[dict[str, Any]] = []
        self.recommended_training_iterations_: int | None = None
        self.profile_: DatasetProfile | None = None

    def _dataset_to_frame(self, dataset: object) -> pd.DataFrame:
        """Convert a NeuralForecast dataset into a long-format frame."""
        recommendation_dataset = cast(RecommendationDataset, dataset)
        values = recommendation_dataset.temporal.numpy()
        time_col = int(recommendation_dataset.y_idx)
        rows: list[pd.DataFrame] = []
        for series_idx in range(len(recommendation_dataset.indptr) - 1):
            start = int(recommendation_dataset.indptr[series_idx])
            end = int(recommendation_dataset.indptr[series_idx + 1])
            length = end - start
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": [f"series_{series_idx}"] * length,
                        "ds": pd.RangeIndex(length),
                        "y": values[start:end, time_col],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def _recommendation_frame(
        self,
        dataset: object,
        val_size: int,
        test_size: int,
    ) -> pd.DataFrame:
        """Build the leakage-safe frame used for profiling and search."""
        frame = self._dataset_to_frame(dataset)
        return trim_frame_for_recommendation(frame, holdout_size=val_size + test_size)

    def _build_search_configs(
        self, recommendation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Create a compact local search around the heuristic recommendation."""
        base = {
            key: value
            for key, value in recommendation.items()
            if key != "recommended_training_iterations"
        }
        input_size = int(base["input_size"])
        candidate_configs = [base]
        candidate_configs.append(
            {
                **base,
                "basis_num": int(base["basis_num"]) + 1,
                "learning_rate": float(base["learning_rate"]) / 2.0,
            }
        )
        wider_window = min(input_size - 1, int(base["moving_avg_window"]) + 2)
        if wider_window % 2 == 0:
            wider_window -= 1
        candidate_configs.append(
            {
                **base,
                "moving_avg_window": max(3, wider_window),
                "period_len": min(
                    max(candidate_periods_for_frequency(self.freq, input_size)),
                    input_size,
                ),
            }
        )
        unique_configs: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for config in candidate_configs:
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_configs.append(config)
        return unique_configs[: max(1, self.n_search_configs)]

    def _candidate_model_kwargs(
        self, config: dict[str, Any], max_steps: int
    ) -> dict[str, Any]:
        """Convert a selected config into model kwargs for a short run."""
        trainer_kwargs = {
            key: value
            for key, value in self.trainer_kwargs.items()
            if key != "callbacks"
        }
        return {
            **config,
            "max_steps": max_steps,
            "use_period_norm": self.core.use_period_norm,
            "use_orthogonal": self.use_orthogonal,
            "orthogonal_weight": self.orthogonal_weight,
            "loss": self.loss,
            "valid_loss": self.valid_loss,
            "num_lr_decays": getattr(self, "num_lr_decays", -1),
            "batch_size": getattr(self, "batch_size", 32),
            "valid_batch_size": getattr(self, "valid_batch_size", None),
            "windows_batch_size": getattr(self, "windows_batch_size", 1024),
            "inference_windows_batch_size": getattr(
                self, "inference_windows_batch_size", 1024
            ),
            "start_padding_enabled": getattr(self, "start_padding_enabled", False),
            "step_size": getattr(self, "step_size", 1),
            "scaler_type": getattr(self, "scaler_type", "identity"),
            "random_seed": getattr(self, "random_seed", 1),
            "num_workers_loader": getattr(self, "num_workers_loader", 0),
            "drop_last_loader": getattr(self, "drop_last_loader", False),
            "optimizer": getattr(self, "optimizer", None),
            "optimizer_kwargs": getattr(self, "optimizer_kwargs", None),
            "lr_scheduler": getattr(self, "lr_scheduler", None),
            "lr_scheduler_kwargs": getattr(self, "lr_scheduler_kwargs", None),
            **trainer_kwargs,
        }

    def _score_candidate(
        self,
        dataset: object,
        val_size: int,
        test_size: int,
        config: dict[str, Any],
    ) -> float:
        """Fit one short candidate model and return its validation score."""
        short_model = TimeBaseTrend(
            h=self.h,
            **self._candidate_model_kwargs(config, self.search_max_steps),
        )
        fitted = short_model.fit(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            random_seed=self.random_seed,
        )
        metrics = getattr(fitted, "metrics", {})
        score = metrics.get("ptl/val_loss", metrics.get("valid_loss"))
        return float(score)

    def _select_config(
        self,
        dataset: object,
        val_size: int,
        test_size: int,
    ) -> dict[str, Any]:
        """Choose the final configuration from heuristics and short search."""
        recommendation_frame = self._recommendation_frame(dataset, val_size, test_size)
        recommendation = self.recommend_defaults(
            recommendation_frame,
            freq=self.freq,
            horizon=self.h,
            max_steps=self.max_steps,
            holdout_size=0,
            include_iteration_recommendation=self.include_iteration_recommendation,
        )
        self.profile_ = profile_dataset(recommendation_frame, self.freq, self.h)
        if "recommended_training_iterations" in recommendation:
            self.recommended_training_iterations_ = int(
                recommendation["recommended_training_iterations"]
            )
        if not self.search_enabled or val_size <= 0:
            return {
                key: value
                for key, value in recommendation.items()
                if key != "recommended_training_iterations"
            }

        best_config = None
        best_score = float("inf")
        self.candidate_results_ = []
        for config in self._build_search_configs(recommendation):
            score = self._score_candidate(dataset, val_size, test_size, config)
            candidate_result = {**config, "validation_score": score}
            self.candidate_results_.append(candidate_result)
            if score < best_score:
                best_score = score
                best_config = config
        if best_config is None:
            msg = (
                "AutoTimeBaseTrend search could not score any candidate configuration."
            )
            raise RuntimeError(msg)
        return best_config

    def _apply_selected_config(self, config: dict[str, Any]) -> None:
        """Apply the selected configuration before the final fit."""
        self.input_size = int(config["input_size"])
        self.learning_rate = float(config["learning_rate"])
        self.max_steps = int(config["max_steps"])
        self.early_stop_patience_steps = int(config["early_stop_patience_steps"])
        self.val_check_steps = int(config["val_check_steps"])
        self.moving_avg_window = int(config["moving_avg_window"])
        self.decomp = SeriesDecomp(self.moving_avg_window)
        self.core = TimeBaseCore(
            TimeBaseConfig(
                period_len=int(config["period_len"]),
                basis_num=int(config["basis_num"]),
                use_period_norm=self.core.use_period_norm,
            ),
            input_size=self.input_size,
            pred_len=self.h,
        )
        self.linear_trend = nn.Linear(self.input_size, self.h)
        self.selected_config_ = dict(config)
        self.hparams.update(
            {
                "freq": self.freq,
                "input_size": self.input_size,
                "period_len": int(config["period_len"]),
                "basis_num": int(config["basis_num"]),
                "moving_avg_window": self.moving_avg_window,
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate,
                "early_stop_patience_steps": self.early_stop_patience_steps,
                "val_check_steps": self.val_check_steps,
            }
        )

    def fit(
        self,
        dataset: object,
        val_size: int = 0,
        test_size: int = 0,
        random_seed: int | None = None,
        distributed_config: object | None = None,
    ) -> object:
        """Select parameters safely from fit data, then train the final model."""
        selected_config = self._select_config(dataset, val_size, test_size)
        self._apply_selected_config(selected_config)
        return super().fit(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            distributed_config=distributed_config,
        )
