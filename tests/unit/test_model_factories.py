"""Tests for TimeBase internal configuration factories."""

from __future__ import annotations

from timebaseula.models.config import (
    TimeBaseArchitectureConfig,
    TimeBaseRegularizationConfig,
)
from timebaseula.models.core import TimeBaseCore
from timebaseula.models.factories import build_timebase_core, resolve_model_components


def test_resolve_model_components_builds_shared_architecture_settings() -> None:
    """Shared factory helpers should resolve explicit-model defaults once."""
    components = resolve_model_components(
        h=12,
        input_size=None,
        period_len=None,
        basis_num=4,
        freq="D",
        use_period_norm=False,
        use_orthogonal=True,
        orthogonal_weight=0.25,
        trainer_kwargs={"devices": 2},
    )

    assert components.model_settings.input_size == 24
    assert components.model_settings.period_len == 7
    assert components.model_settings.trainer_kwargs == {
        "accelerator": "cpu",
        "devices": 2,
    }
    assert components.architecture_config == TimeBaseArchitectureConfig(
        input_size=24,
        period_len=7,
        basis_num=4,
        use_period_norm=False,
    )
    assert components.regularization_config == TimeBaseRegularizationConfig(
        use_orthogonal=True,
        orthogonal_weight=0.25,
    )


def test_build_timebase_core_creates_core_from_architecture_config() -> None:
    """The core factory should build a TimeBaseCore instance from config objects."""
    architecture_config = TimeBaseArchitectureConfig(
        input_size=24,
        period_len=6,
        basis_num=4,
        use_period_norm=True,
    )

    core = build_timebase_core(architecture_config=architecture_config, horizon=12)

    assert isinstance(core, TimeBaseCore)
    assert core.input_size == 24
    assert core.period_len == 6
    assert core.basis_num == 4
