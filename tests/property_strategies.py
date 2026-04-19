"""Shared Hypothesis strategies for the repository test suite.

This module centralizes reusable property-based testing helpers so individual test
files can stay focused on model contracts instead of strategy plumbing.

Hypothesis rewrites ``@st.composite`` helpers into strategy factories at runtime,
but ``ty`` still sees the original helper signature with its internal ``draw``
parameter. The local ``# type: ignore[missing-argument]`` below keeps that
workaround in one place instead of repeating it throughout the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from hypothesis import strategies as st
import torch
from neuralforecast.losses.pytorch import DistributionLoss

from timebaseula.models.core import TimeBaseConfig, TimeBaseCore
from timebaseula.models.timebase import TimeBase, TimeBaseTrend

T = TypeVar("T")


@dataclass(frozen=True)
class CoreCase:
    """Inputs and expectations for a univariate TimeBase core forward pass."""

    core: TimeBaseCore
    series: torch.Tensor
    horizon: int
    period_len: int


@dataclass(frozen=True)
class CoreMultivariateCase:
    """Inputs and expectations for a multivariate TimeBase core forward pass."""

    core: TimeBaseCore
    series: torch.Tensor
    horizon: int
    period_len: int
    basis_num: int


@dataclass(frozen=True)
class ModelCase:
    """Generated model arguments plus an input tensor batch."""

    h: int
    input_size: int
    period_len: int
    basis_num: int
    insample_y: torch.Tensor


@dataclass(frozen=True)
class DistributionCase:
    """Distribution loss setup for explicit model output-contract tests."""

    model_cls: type[TimeBase] | type[TimeBaseTrend]
    loss: DistributionLoss
    parameter_count: int


def _composite_strategy(
    factory: Any, /, *args: object, **kwargs: object
) -> st.SearchStrategy[T]:
    """Call a Hypothesis composite factory with the local `ty` workaround.

    Hypothesis injects the internal ``draw`` argument when the strategy is built,
    but ``ty`` validates the undecorated helper signature and reports a false
    missing-argument error. Keeping the ignore here avoids repeating it across
    every test file.
    """
    return factory(*args, **kwargs)  # type: ignore[missing-argument]


def finite_float32(
    *,
    min_value: float,
    max_value: float,
) -> st.SearchStrategy[float]:
    """Return bounded finite float32-compatible values."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )


def odd_integers(min_value: int, max_value: int) -> st.SearchStrategy[int]:
    """Return odd integers in the inclusive range."""
    first = min_value if min_value % 2 == 1 else min_value + 1
    last = max_value if max_value % 2 == 1 else max_value - 1
    if first > last:
        raise ValueError("No odd integers available in the provided range.")
    return st.integers(min_value=0, max_value=(last - first) // 2).map(
        lambda offset: first + 2 * offset
    )


def even_integers(min_value: int, max_value: int) -> st.SearchStrategy[int]:
    """Return even integers in the inclusive range."""
    first = min_value if min_value % 2 == 0 else min_value + 1
    last = max_value if max_value % 2 == 0 else max_value - 1
    if first > last:
        raise ValueError("No even integers available in the provided range.")
    return st.integers(min_value=0, max_value=(last - first) // 2).map(
        lambda offset: first + 2 * offset
    )


@st.composite
def _tensor_2d(
    draw: st.DrawFn,
    *,
    time_steps: int | None = None,
    time_steps_min: int = 1,
    time_steps_max: int = 12,
    batch_size_min: int = 1,
    batch_size_max: int = 4,
    value_min: float = -100,
    value_max: float = 100,
) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=batch_size_min, max_value=batch_size_max))
    resolved_time_steps = time_steps
    if resolved_time_steps is None:
        resolved_time_steps = draw(
            st.integers(min_value=time_steps_min, max_value=time_steps_max)
        )
    values = draw(
        st.lists(
            st.lists(
                finite_float32(min_value=value_min, max_value=value_max),
                min_size=resolved_time_steps,
                max_size=resolved_time_steps,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


@st.composite
def _tensor_3d(
    draw: st.DrawFn,
    *,
    time_steps: int | None = None,
    time_steps_min: int = 1,
    time_steps_max: int = 10,
    batch_size_min: int = 1,
    batch_size_max: int = 3,
    n_series_min: int = 1,
    n_series_max: int = 4,
    value_min: float = -100,
    value_max: float = 100,
) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=batch_size_min, max_value=batch_size_max))
    n_series = draw(st.integers(min_value=n_series_min, max_value=n_series_max))
    resolved_time_steps = time_steps
    if resolved_time_steps is None:
        resolved_time_steps = draw(
            st.integers(min_value=time_steps_min, max_value=time_steps_max)
        )
    values = draw(
        st.lists(
            st.lists(
                st.lists(
                    finite_float32(min_value=value_min, max_value=value_max),
                    min_size=n_series,
                    max_size=n_series,
                ),
                min_size=resolved_time_steps,
                max_size=resolved_time_steps,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


def tensor_2d(
    *,
    time_steps: int | None = None,
    time_steps_min: int = 1,
    time_steps_max: int = 12,
    batch_size_min: int = 1,
    batch_size_max: int = 4,
    value_min: float = -100,
    value_max: float = 100,
) -> st.SearchStrategy[torch.Tensor]:
    """Return bounded univariate float tensor batches."""
    return _composite_strategy(
        _tensor_2d,
        time_steps=time_steps,
        time_steps_min=time_steps_min,
        time_steps_max=time_steps_max,
        batch_size_min=batch_size_min,
        batch_size_max=batch_size_max,
        value_min=value_min,
        value_max=value_max,
    )


def tensor_3d(
    *,
    time_steps: int | None = None,
    time_steps_min: int = 1,
    time_steps_max: int = 10,
    batch_size_min: int = 1,
    batch_size_max: int = 3,
    n_series_min: int = 1,
    n_series_max: int = 4,
    value_min: float = -100,
    value_max: float = 100,
) -> st.SearchStrategy[torch.Tensor]:
    """Return bounded multivariate float tensor batches."""
    return _composite_strategy(
        _tensor_3d,
        time_steps=time_steps,
        time_steps_min=time_steps_min,
        time_steps_max=time_steps_max,
        batch_size_min=batch_size_min,
        batch_size_max=batch_size_max,
        n_series_min=n_series_min,
        n_series_max=n_series_max,
        value_min=value_min,
        value_max=value_max,
    )


@st.composite
def _core_univariate_case(draw: st.DrawFn) -> CoreCase:
    input_size = draw(st.integers(min_value=1, max_value=24))
    period_len = draw(st.integers(min_value=1, max_value=min(input_size, 8)))
    basis_num = draw(st.integers(min_value=1, max_value=6))
    horizon = draw(st.integers(min_value=1, max_value=12))
    series = draw(tensor_2d(time_steps=input_size, batch_size_max=4))
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            use_period_norm=True,
        ),
        horizon=horizon,
    )
    return CoreCase(core=core, series=series, horizon=horizon, period_len=period_len)


@st.composite
def _core_multivariate_case(draw: st.DrawFn) -> CoreMultivariateCase:
    input_size = draw(st.integers(min_value=1, max_value=18))
    period_len = draw(st.integers(min_value=1, max_value=min(input_size, 6)))
    basis_num = draw(st.integers(min_value=1, max_value=5))
    horizon = draw(st.integers(min_value=1, max_value=10))
    series = draw(
        tensor_3d(
            time_steps=input_size,
            batch_size_max=3,
            n_series_max=4,
        )
    )
    torch.manual_seed(0)
    core = TimeBaseCore(
        config=TimeBaseConfig(
            input_size=input_size,
            period_len=period_len,
            basis_num=basis_num,
            use_period_norm=draw(st.booleans()),
        ),
        horizon=horizon,
    )
    return CoreMultivariateCase(
        core=core,
        series=series,
        horizon=horizon,
        period_len=period_len,
        basis_num=basis_num,
    )


def core_univariate_cases() -> st.SearchStrategy[CoreCase]:
    """Return valid univariate core-forward cases."""
    return _composite_strategy(_core_univariate_case)


def core_multivariate_cases() -> st.SearchStrategy[CoreMultivariateCase]:
    """Return valid multivariate core-forward cases."""
    return _composite_strategy(_core_multivariate_case)


@st.composite
def _model_case(draw: st.DrawFn, *, multivariate: bool) -> ModelCase:
    h = draw(st.integers(min_value=1, max_value=10))
    input_size = draw(st.integers(min_value=1, max_value=24))
    period_len = draw(st.integers(min_value=1, max_value=min(input_size, 8)))
    basis_num = draw(st.integers(min_value=1, max_value=6))
    if multivariate:
        insample_y = draw(
            tensor_3d(
                time_steps=input_size,
                value_min=-50,
                value_max=50,
            )
        )
    else:
        insample_y = draw(
            tensor_2d(
                time_steps=input_size,
                value_min=-50,
                value_max=50,
            )
        )
    return ModelCase(
        h=h,
        input_size=input_size,
        period_len=period_len,
        basis_num=basis_num,
        insample_y=insample_y,
    )


@st.composite
def _padded_model_case(draw: st.DrawFn) -> ModelCase:
    h = draw(st.integers(min_value=1, max_value=10))
    period_len = draw(st.integers(min_value=2, max_value=8))
    quotient = draw(st.integers(min_value=1, max_value=3))
    remainder = draw(st.integers(min_value=1, max_value=period_len - 1))
    input_size = quotient * period_len + remainder
    basis_num = draw(st.integers(min_value=1, max_value=6))
    insample_y = draw(
        tensor_2d(
            time_steps=input_size,
            value_min=-50,
            value_max=50,
        )
    )
    return ModelCase(
        h=h,
        input_size=input_size,
        period_len=period_len,
        basis_num=basis_num,
        insample_y=insample_y,
    )


@st.composite
def _distribution_case(draw: st.DrawFn) -> DistributionCase:
    model_cls = draw(st.sampled_from([TimeBase, TimeBaseTrend]))
    loss_name, parameter_count = draw(
        st.sampled_from(
            [
                ("Normal", 2),
                ("Poisson", 1),
                ("StudentT", 3),
                ("NegativeBinomial", 2),
                ("Tweedie", 1),
            ]
        )
    )
    return DistributionCase(
        model_cls=model_cls,
        loss=DistributionLoss(loss_name),
        parameter_count=parameter_count,
    )


def univariate_model_cases() -> st.SearchStrategy[ModelCase]:
    """Return valid univariate explicit-model cases."""
    return _composite_strategy(_model_case, multivariate=False)


def multivariate_model_cases() -> st.SearchStrategy[ModelCase]:
    """Return valid multivariate explicit-model cases."""
    return _composite_strategy(_model_case, multivariate=True)


def padded_model_cases() -> st.SearchStrategy[ModelCase]:
    """Return valid explicit-model cases with non-divisible input windows."""
    return _composite_strategy(_padded_model_case)


def distribution_cases() -> st.SearchStrategy[DistributionCase]:
    """Return supported distribution-loss cases for explicit models."""
    return _composite_strategy(_distribution_case)
