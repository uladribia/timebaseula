"""NeuralForecast auto wrappers for the TimeBase family."""

from __future__ import annotations

from os import cpu_count
from typing import Any

import torch
from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.losses.pytorch import MAE
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from timebaseula.models.timebase import TimeBase, TimeBaseTrend

DEFAULT_SCALER_CHOICES = ("identity",)
DEFAULT_STEP_SIZE_MULTIPLIERS = (1,)
TIMEBASE_INPUT_SIZE_MULTIPLIERS = (2, 3, 4, 5)
TIMEBASETREND_INPUT_SIZE_MULTIPLIERS = (3, 4, 5, 6)


def _resolve_cpus(cpus: int | None) -> int:
    """Return a concrete CPU count for BaseAuto."""
    return 1 if cpus is None else cpus


class AutoTimeBase(BaseAuto):
    """Automatic hyperparameter tuning wrapper for :class:`TimeBase`."""

    default_config: dict[str, Any] = {
        "input_size_multiplier": TIMEBASE_INPUT_SIZE_MULTIPLIERS,
        "h": None,
        "basis_num": tune.choice([6, 8, 10]),
        "period_len": tune.choice([7, 14, 28]),
        "learning_rate": tune.loguniform(5e-4, 1e-2),
        "scaler_type": tune.choice(list(DEFAULT_SCALER_CHOICES)),
        "step_size_multiplier": DEFAULT_STEP_SIZE_MULTIPLIERS,
        "max_steps": tune.choice([100, 140, 220]),
        "batch_size": tune.choice([32, 64]),
        "windows_batch_size": tune.choice([256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h: int,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples: int = 10,
        refit_with_val: bool = False,
        cpus: int | None = cpu_count(),
        gpus: int = torch.cuda.device_count(),
        verbose: bool = False,
        alias: str | None = None,
        backend: str = "ray",
        callbacks=None,
    ) -> None:
        """Initialize the auto wrapper around :class:`TimeBase`."""
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super().__init__(
            cls_model=TimeBase,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=_resolve_cpus(cpus),
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h: int, backend: str, n_series=None):
        """Return the default search space for automatic TimeBase tuning."""
        del n_series
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [max(h * multiplier, 8) for multiplier in TIMEBASE_INPUT_SIZE_MULTIPLIERS]
        )
        config["step_size"] = tune.choice(
            [h * multiplier for multiplier in DEFAULT_STEP_SIZE_MULTIPLIERS]
        )
        del config["input_size_multiplier"], config["step_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config


class AutoTimeBaseTrend(BaseAuto):
    """Automatic hyperparameter tuning wrapper for :class:`TimeBaseTrend`."""

    default_config: dict[str, Any] = {
        "input_size_multiplier": TIMEBASETREND_INPUT_SIZE_MULTIPLIERS,
        "h": None,
        "basis_num": tune.choice([6, 8, 10]),
        "period_len": tune.choice([7, 14, 28]),
        "moving_avg_window": tune.choice([21, 29, 35]),
        "learning_rate": tune.loguniform(5e-4, 1e-2),
        "scaler_type": tune.choice(list(DEFAULT_SCALER_CHOICES)),
        "step_size_multiplier": DEFAULT_STEP_SIZE_MULTIPLIERS,
        "max_steps": tune.choice([120, 180, 260]),
        "batch_size": tune.choice([32, 64]),
        "windows_batch_size": tune.choice([256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h: int,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples: int = 10,
        refit_with_val: bool = False,
        cpus: int | None = cpu_count(),
        gpus: int = torch.cuda.device_count(),
        verbose: bool = False,
        alias: str | None = None,
        backend: str = "ray",
        callbacks=None,
    ) -> None:
        """Initialize the auto wrapper around :class:`TimeBaseTrend`."""
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super().__init__(
            cls_model=TimeBaseTrend,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=_resolve_cpus(cpus),
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h: int, backend: str, n_series=None):
        """Return the default search space for automatic TimeBaseTrend tuning."""
        del n_series
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [
                max(h * multiplier, 8)
                for multiplier in TIMEBASETREND_INPUT_SIZE_MULTIPLIERS
            ]
        )
        config["step_size"] = tune.choice(
            [h * multiplier for multiplier in DEFAULT_STEP_SIZE_MULTIPLIERS]
        )
        del config["input_size_multiplier"], config["step_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config
