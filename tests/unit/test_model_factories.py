"""Tests for TimeBase internal configuration factories."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from timebaseula.models.config import (
    TimeBaseArchitectureConfig,
    TimeBaseRegularizationConfig,
)
from timebaseula.models.core import TimeBaseCore
from timebaseula.models.factories import build_timebase_core, resolve_model_components


@given(
    h=st.integers(min_value=1, max_value=32),
    input_size=st.one_of(st.none(), st.integers(min_value=1, max_value=64)),
    period_len=st.one_of(st.none(), st.integers(min_value=1, max_value=32)),
    basis_num=st.integers(min_value=1, max_value=12),
    freq=st.one_of(st.none(), st.sampled_from(["D", "M", "ME", "MS"])),
    use_period_norm=st.booleans(),
    use_orthogonal=st.booleans(),
    orthogonal_weight=st.floats(
        min_value=0, max_value=5, allow_nan=False, allow_infinity=False, width=32
    ),
    devices=st.integers(min_value=1, max_value=8),
)
def test_resolve_model_components_builds_shared_architecture_settings(
    h: int,
    input_size: int | None,
    period_len: int | None,
    basis_num: int,
    freq: str | None,
    use_period_norm: bool,
    use_orthogonal: bool,
    orthogonal_weight: float,
    devices: int,
) -> None:
    """Shared factory helpers should resolve explicit-model defaults once."""
    components = resolve_model_components(
        h=h,
        input_size=input_size,
        period_len=period_len,
        basis_num=basis_num,
        freq=freq,
        use_period_norm=use_period_norm,
        use_orthogonal=use_orthogonal,
        orthogonal_weight=orthogonal_weight,
        trainer_kwargs={"devices": devices},
    )

    expected_input_size = max(2 * h, 8) if input_size is None else input_size
    if period_len is None:
        if freq == "D":
            expected_period_len = min(7, expected_input_size)
        elif freq in {"M", "ME", "MS"}:
            expected_period_len = min(12, expected_input_size)
        else:
            expected_period_len = min(max(2, h), expected_input_size)
    else:
        expected_period_len = period_len

    assert components.model_settings.input_size == expected_input_size
    assert components.model_settings.period_len == expected_period_len
    assert components.model_settings.trainer_kwargs == {
        "accelerator": "cpu",
        "devices": devices,
    }
    assert components.architecture_config == TimeBaseArchitectureConfig(
        input_size=expected_input_size,
        period_len=expected_period_len,
        basis_num=basis_num,
        use_period_norm=use_period_norm,
    )
    assert components.regularization_config == TimeBaseRegularizationConfig(
        use_orthogonal=use_orthogonal,
        orthogonal_weight=orthogonal_weight,
    )


@given(
    input_size=st.integers(min_value=1, max_value=64),
    period_len=st.integers(min_value=1, max_value=32),
    basis_num=st.integers(min_value=1, max_value=12),
    use_period_norm=st.booleans(),
    horizon=st.integers(min_value=1, max_value=32),
)
def test_build_timebase_core_creates_core_from_architecture_config(
    input_size: int,
    period_len: int,
    basis_num: int,
    use_period_norm: bool,
    horizon: int,
) -> None:
    """The core factory should build a TimeBaseCore instance from config objects."""
    architecture_config = TimeBaseArchitectureConfig(
        input_size=input_size,
        period_len=period_len,
        basis_num=basis_num,
        use_period_norm=use_period_norm,
    )

    core = build_timebase_core(architecture_config=architecture_config, horizon=horizon)

    assert isinstance(core, TimeBaseCore)
    assert core.input_size == input_size
    assert core.period_len == period_len
    assert core.basis_num == basis_num
    assert core.pred_len == horizon
