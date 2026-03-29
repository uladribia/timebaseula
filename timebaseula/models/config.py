"""Internal configuration objects for TimeBase models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSettings:
    """Resolved settings shared by the public explicit models."""

    input_size: int
    period_len: int
    trainer_kwargs: dict[str, Any]


@dataclass(frozen=True)
class TimeBaseArchitectureConfig:
    """Architecture settings for the segmented TimeBase core."""

    input_size: int
    period_len: int
    basis_num: int
    use_period_norm: bool


@dataclass(frozen=True)
class TimeBaseRegularizationConfig:
    """Regularization settings for the explicit TimeBase models."""

    use_orthogonal: bool
    orthogonal_weight: float


@dataclass(frozen=True)
class ResolvedTimeBaseComponents:
    """Resolved shared components used to build explicit TimeBase models."""

    model_settings: ModelSettings
    architecture_config: TimeBaseArchitectureConfig
    regularization_config: TimeBaseRegularizationConfig
