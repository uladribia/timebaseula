"""Factory helpers for shared TimeBase model construction."""

from __future__ import annotations

from typing import Any

from timebaseula.models.config import (
    ModelSettings,
    ResolvedTimeBaseComponents,
    TimeBaseArchitectureConfig,
    TimeBaseRegularizationConfig,
)
from timebaseula.models.core import TimeBaseConfig, TimeBaseCore
from timebaseula.models.defaults import (
    _default_input_size,
    _default_period_len,
    _default_trainer_kwargs,
)


def resolve_model_settings(
    h: int,
    input_size: int | None,
    period_len: int | None,
    freq: str | None,
    trainer_kwargs: dict[str, Any],
) -> ModelSettings:
    """Resolve shared constructor defaults for public TimeBase models."""
    resolved_input_size = (
        _default_input_size(h) if input_size is None else int(input_size)
    )
    resolved_period_len = (
        _default_period_len(h, resolved_input_size, freq)
        if period_len is None
        else int(period_len)
    )
    return ModelSettings(
        input_size=resolved_input_size,
        period_len=resolved_period_len,
        trainer_kwargs=_default_trainer_kwargs(trainer_kwargs),
    )


def resolve_model_components(
    h: int,
    input_size: int | None,
    period_len: int | None,
    basis_num: int,
    freq: str | None,
    use_period_norm: bool,
    use_orthogonal: bool,
    orthogonal_weight: float,
    trainer_kwargs: dict[str, Any],
) -> ResolvedTimeBaseComponents:
    """Resolve the shared internal components for explicit TimeBase models."""
    model_settings = resolve_model_settings(
        h=h,
        input_size=input_size,
        period_len=period_len,
        freq=freq,
        trainer_kwargs=trainer_kwargs,
    )
    return ResolvedTimeBaseComponents(
        model_settings=model_settings,
        architecture_config=TimeBaseArchitectureConfig(
            input_size=model_settings.input_size,
            period_len=model_settings.period_len,
            basis_num=int(basis_num),
            use_period_norm=use_period_norm,
        ),
        regularization_config=TimeBaseRegularizationConfig(
            use_orthogonal=use_orthogonal,
            orthogonal_weight=orthogonal_weight,
        ),
    )


def build_timebase_core(
    architecture_config: TimeBaseArchitectureConfig,
    horizon: int,
) -> TimeBaseCore:
    """Build the pure TimeBase core from resolved architecture settings."""
    return TimeBaseCore(
        config=TimeBaseConfig(
            input_size=architecture_config.input_size,
            period_len=architecture_config.period_len,
            basis_num=architecture_config.basis_num,
            use_period_norm=architecture_config.use_period_norm,
        ),
        horizon=horizon,
    )
