"""TimeBase models and thin Nixtla-style auto wrappers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from os import cpu_count
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.dlinear import SeriesDecomp
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

DEFAULT_LOSS = MAE()
DEFAULT_SEARCH_ALG = BasicVariantGenerator(random_state=1)
DEFAULT_BASIS_NUM = 6
DEFAULT_MOVING_AVG_WINDOW = 25

SearchSpace = dict[str, object]
AutoConfig = SearchSpace | Callable[[object], SearchSpace]


def _normalize_frequency(freq: str | None) -> str | None:
    """Normalize a frequency string when one is provided."""
    if freq is None:
        return None
    return freq.upper()


def _default_input_size(horizon: int) -> int:
    """Return a simple horizon-based default input size."""
    return max(2 * horizon, 8)


def _default_period_len(
    horizon: int,
    input_size: int,
    freq: str | None,
) -> int:
    """Return a simple seasonal default for the explicit models."""
    normalized_freq = _normalize_frequency(freq)
    if normalized_freq == "D":
        return min(7, input_size)
    if normalized_freq in {"M", "ME", "MS"}:
        return min(12, input_size)
    return min(max(2, horizon), input_size)


def _default_moving_avg_window() -> int:
    """Return the default decomposition window."""
    return DEFAULT_MOVING_AVG_WINDOW


def _default_trainer_kwargs(
    trainer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Force CPU-first trainer defaults while preserving user overrides."""
    resolved = dict(trainer_kwargs)
    resolved.setdefault("accelerator", "cpu")
    resolved.setdefault("devices", 1)
    return resolved


def _period_candidates(freq: str | None, horizon: int) -> list[int]:
    """Return a compact set of candidate period lengths for auto models."""
    normalized_freq = _normalize_frequency(freq)
    if normalized_freq == "D":
        candidates = [7, 14, 28]
    elif normalized_freq in {"M", "ME", "MS"}:
        candidates = [3, 6, 12]
    else:
        candidates = [max(2, horizon // 2), max(2, horizon), max(2, 2 * horizon)]
    return sorted({int(candidate) for candidate in candidates if int(candidate) >= 2})


@dataclass(frozen=True)
class TimeBaseConfig:
    """Configuration for the TimeBase core components."""

    period_len: int
    basis_num: int
    use_period_norm: bool


class TrainingLossNaNError(RuntimeError):
    """Raised when the training loss becomes NaN."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Loss is NaN, training stopped.")


class TimeBaseCore(nn.Module):
    """Core TimeBase operations for basis extraction and segment forecasting."""

    def __init__(self, config: TimeBaseConfig, input_size: int, pred_len: int) -> None:
        """Initialize the TimeBase core components."""
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
                batch_size,
                self.period_len,
                self.seg_num_y,
            )

        forecast = forecast_segments.permute(0, 2, 1).reshape(batch_size, -1)
        return forecast[:, : self.pred_len].contiguous(), basis


class BaseTimeBaseWindows(BaseWindows):
    """Shared NeuralForecast wrapper logic for TimeBase-style models."""

    SAMPLING_TYPE = "windows"
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
        """Initialize the TimeBase wrapper with deterministic defaults."""
        resolved_input_size = (
            _default_input_size(h) if input_size is None else int(input_size)
        )
        resolved_period_len = (
            _default_period_len(h, resolved_input_size, freq)
            if period_len is None
            else int(period_len)
        )
        resolved_trainer_kwargs = _default_trainer_kwargs(trainer_kwargs)

        super().__init__(
            h=h,
            input_size=resolved_input_size,
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
            **resolved_trainer_kwargs,
        )
        self.freq = freq
        self.core = TimeBaseCore(
            TimeBaseConfig(
                period_len=resolved_period_len,
                basis_num=int(basis_num),
                use_period_norm=use_period_norm,
            ),
            input_size=resolved_input_size,
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
    """TimeBase with trend decomposition, compatible with BaseWindows."""

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
        """Initialize the TimeBaseTrend wrapper with deterministic defaults."""
        resolved_input_size = (
            _default_input_size(h) if input_size is None else int(input_size)
        )
        resolved_period_len = (
            _default_period_len(h, resolved_input_size, freq)
            if period_len is None
            else int(period_len)
        )
        resolved_window = (
            _default_moving_avg_window()
            if moving_avg_window is None
            else int(moving_avg_window)
        )
        if resolved_window % 2 == 0:
            msg = "moving_avg_window must be odd for moving average decomposition"
            raise ValueError(msg)

        resolved_trainer_kwargs = _default_trainer_kwargs(trainer_kwargs)

        super().__init__(
            h=h,
            input_size=resolved_input_size,
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
            **resolved_trainer_kwargs,
        )
        self.freq = freq
        self.moving_avg_window = resolved_window
        self.decomp = SeriesDecomp(self.moving_avg_window)
        self.core = TimeBaseCore(
            TimeBaseConfig(
                period_len=resolved_period_len,
                basis_num=int(basis_num),
                use_period_norm=use_period_norm,
            ),
            input_size=resolved_input_size,
            pred_len=h,
        )
        self.linear_trend = nn.Linear(resolved_input_size, h)
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


class AutoTimeBase(BaseAuto):
    """Nixtla-style auto wrapper for TimeBase."""

    default_config: ClassVar[SearchSpace] = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "period_len": None,
        "basis_num": tune.choice([4, 6, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "scaler_type": tune.choice(["identity", "robust", "standard"]),
        "max_steps": tune.choice([100, 300, 500]),
        "batch_size": tune.choice([32, 64, 128]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h: int,
        loss: nn.Module = DEFAULT_LOSS,
        valid_loss: nn.Module | None = None,
        config: SearchSpace | None = None,
        search_alg: object | None = None,
        num_samples: int = 10,
        refit_with_val: bool = False,
        cpus: int | None = None,
        gpus: int = 0,
        verbose: bool = False,
        alias: str | None = None,
        backend: str = "ray",
        callbacks: list[Any] | None = None,
        freq: str | None = None,
    ) -> None:
        """Initialize the Nixtla-style auto wrapper for TimeBase."""
        resolved_search_alg = DEFAULT_SEARCH_ALG if search_alg is None else search_alg
        resolved_config = (
            self.get_default_config(h=h, backend=backend, freq=freq)
            if config is None
            else config
        )
        super().__init__(
            cls_model=TimeBase,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=resolved_config,
            search_alg=resolved_search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpu_count() if cpus is None else cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )
        self.freq = freq

    @classmethod
    def get_default_config(
        cls,
        h: int,
        backend: str,
        freq: str | None = None,
        n_series: int | None = None,
    ) -> AutoConfig:
        """Return the default Nixtla-style search space for TimeBase."""
        del n_series
        config = cls.default_config.copy()
        input_size_multipliers = cast(list[int], config["input_size_multiplier"])
        config["input_size"] = tune.choice(
            [max(8, h * multiplier) for multiplier in input_size_multipliers]
        )
        config["period_len"] = tune.choice(_period_candidates(freq=freq, horizon=h))
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config


class AutoTimeBaseTrend(BaseAuto):
    """Nixtla-style auto wrapper for TimeBaseTrend."""

    default_config: ClassVar[SearchSpace] = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "period_len": None,
        "basis_num": tune.choice([4, 6, 8]),
        "moving_avg_window": tune.choice([3, 5, 7, 11, 25]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "scaler_type": tune.choice(["identity", "robust", "standard"]),
        "max_steps": tune.choice([100, 300, 500]),
        "batch_size": tune.choice([32, 64, 128]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h: int,
        loss: nn.Module = DEFAULT_LOSS,
        valid_loss: nn.Module | None = None,
        config: SearchSpace | None = None,
        search_alg: object | None = None,
        num_samples: int = 10,
        refit_with_val: bool = False,
        cpus: int | None = None,
        gpus: int = 0,
        verbose: bool = False,
        alias: str | None = None,
        backend: str = "ray",
        callbacks: list[Any] | None = None,
        freq: str | None = None,
    ) -> None:
        """Initialize the Nixtla-style auto wrapper for TimeBaseTrend."""
        resolved_search_alg = DEFAULT_SEARCH_ALG if search_alg is None else search_alg
        resolved_config = (
            self.get_default_config(h=h, backend=backend, freq=freq)
            if config is None
            else config
        )
        super().__init__(
            cls_model=TimeBaseTrend,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=resolved_config,
            search_alg=resolved_search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpu_count() if cpus is None else cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )
        self.freq = freq

    @classmethod
    def get_default_config(
        cls,
        h: int,
        backend: str,
        freq: str | None = None,
        n_series: int | None = None,
    ) -> AutoConfig:
        """Return the default Nixtla-style search space for TimeBaseTrend."""
        del n_series
        config = cls.default_config.copy()
        input_size_multipliers = cast(list[int], config["input_size_multiplier"])
        config["input_size"] = tune.choice(
            [max(8, h * multiplier) for multiplier in input_size_multipliers]
        )
        config["period_len"] = tune.choice(_period_candidates(freq=freq, horizon=h))
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config
